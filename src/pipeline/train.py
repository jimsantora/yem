import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from PIL.ImageOps import exif_transpose
from tqdm.auto import tqdm
from typing import Dict, Any
import itertools
from pathlib import Path

class DreamBoothDataset(Dataset):
    def __init__(
        self,
        instance_data_root: str,
        instance_prompt: str,
        tokenizer=None,
        size: int = 1024,
        center_crop: bool = False,
    ):
        self.instance_prompt = instance_prompt
        self.instance_data_root = Path(instance_data_root)
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_images = [Image.open(path) for path in list(Path(instance_data_root).iterdir())]
        self.num_instance_images = len(self.instance_images)
        
        self._length = self.num_instance_images

        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = self.instance_images[index % self.num_instance_images]
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["pixel_values"] = self.image_transforms(instance_image)
        example["instance_prompt"] = self.instance_prompt

        return example

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    
    prompts = [example["instance_prompt"] for example in examples]

    batch = {
        "pixel_values": pixel_values,
        "prompts": prompts,
    }
    return batch

def run_training(args: Dict[str, Any], components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the training loop."""
    dataset = DreamBoothDataset(
        instance_data_root=args["data"],
        instance_prompt=args["prompt"],
        size=args["resolution"],
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Better for MPS
    )
    
    # Training loop components
    transformer = components["transformer"]
    vae = components["vae"]
    scheduler = components["scheduler"]
    optimizer = components["optimizer"]
    accelerator = components["accelerator"]
    
    progress_bar = tqdm(range(config["training"]["num_epochs"]))
    global_step = 0
    
    transformer.train()
    
    for epoch in progress_bar:
        for batch in dataloader:
            with accelerator.accumulate(transformer):
                # Convert images to latent space
                pixel_values = batch["pixel_values"].to(dtype=torch.float32)
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(model_input)
                timesteps = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (model_input.shape[0],),
                    device=model_input.device
                )

                # Add noise according to scheduler
                noisy_model_input = scheduler.add_noise(model_input, noise, timesteps)

                # Predict noise
                noise_pred = transformer(
                    noisy_model_input, 
                    timesteps,
                    return_dict=False
                )[0]

                # Calculate loss
                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), 1.0)
                    
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.set_postfix(loss=loss.detach().item())
                global_step += 1

            # Clear MPS cache periodically
            if global_step % 10 == 0:
                torch.mps.empty_cache()

    return components