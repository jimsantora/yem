from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from PIL.ImageOps import exif_transpose
from tqdm.auto import tqdm

class TrainingDataset(Dataset):
    """Training dataset optimized for Apple Silicon memory usage."""
    def __init__(self, data_root: Path, prompt: str, resolution: int = 1024):
        self.data_root = Path(data_root)
        self.prompt = prompt
        self.image_paths = list(self.data_root.glob("*.[jJ][pP][gG]"))
        
        # Optimize transforms for MPS
        self.transforms = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image = exif_transpose(image)
        
        if not image.mode == "RGB":
            image = image.convert("RGB")
            
        return {
            "pixel_values": self.transforms(image),
            "prompt": self.prompt
        }

def train_step(batch, components):
    """Single training step optimized for MPS memory usage."""
    vae = components['vae']
    transformer = components['transformer']
    scheduler = components['scheduler']
    
    # Move batch to MPS
    pixel_values = batch["pixel_values"].to("mps")
    
    # Get latents
    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
    
    # Sample noise and timesteps
    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0, 
        scheduler.config.num_train_timesteps, 
        (latents.shape[0],), 
        device="mps"
    )
    
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    
    # Predict noise
    noise_pred = transformer(noisy_latents, timesteps).sample
    
    return torch.nn.functional.mse_loss(noise_pred, noise)

def run(args, config):
    """Training loop with Apple Silicon optimizations."""
    dataset = TrainingDataset(args.data, args.prompt, args.resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # Better for MPS
    )
    
    components = config.get('components', {})
    
    optimizer = torch.optim.AdamW(
        components['transformer'].parameters(),
        lr=1e-4,
        weight_decay=1e-2
    )
    
    progress_bar = tqdm(range(args.num_epochs))
    
    for epoch in progress_bar:
        for batch in dataloader:
            loss = train_step(batch, components)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Explicitly clear MPS cache periodically
            if epoch % 10 == 0:
                torch.mps.empty_cache()
            
            progress_bar.set_postfix(loss=loss.item())
    
    return components