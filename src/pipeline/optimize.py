import torch
import psutil
import subprocess
from typing import Dict

class AppleSiliconOptimizer:
    """Handles optimization for Apple Silicon devices."""
    
    def __init__(self):
        self.memory = psutil.virtual_memory()
    
    def get_thermal_status(self) -> str:
        """Check thermal state of the M-series chip."""
        try:
            result = subprocess.run(['pmset', '-g', 'therm'], capture_output=True, text=True)
            return "normal" if "CPU_Speed_Limit=100" in result.stdout else "throttled"
        except:
            return "unknown"
    
    def get_memory_status(self) -> Dict[str, float]:
        """Get current memory usage."""
        self.memory = psutil.virtual_memory()
        return {
            "available_gb": self.memory.available / (1024 ** 3),
            "percent_used": self.memory.percent
        }
    
    def determine_batch_size(self, resolution: int) -> int:
        """Calculate optimal batch size based on memory and resolution."""
        available_gb = self.get_memory_status()["available_gb"]
        
        # Memory estimation per image
        memory_per_image = (resolution ** 2 * 4) / (1024 ** 3)  # GB per image
        safe_memory = available_gb * 0.7  # Use 70% of available memory
        
        # Calculate but cap based on resolution
        if resolution <= 512:
            return min(int(safe_memory / memory_per_image), 4)
        elif resolution <= 768:
            return min(int(safe_memory / memory_per_image), 2)
        return 1
    
    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model for MPS inference."""
        # Convert to float16 for better MPS performance
        model = model.half()
        
        # Clear memory
        torch.mps.empty_cache()
        
        return model

def run(args, config):
    """Main optimization function."""
    optimizer = AppleSiliconOptimizer()
    
    # Optimize batch size if auto
    if args.batch_size == "auto":
        args.batch_size = optimizer.determine_batch_size(args.resolution)
    
    # Get components
    components = config.get('components', {})
    
    # Check thermal state
    thermal_status = optimizer.get_thermal_status()
    if thermal_status == "throttled":
        print("Warning: Device is thermally throttled. Performance may be reduced.")
    
    # Optimize transformer
    if 'transformer' in components:
        components['transformer'] = optimizer.optimize_model(components['transformer'])
    
    # Clean up
    torch.mps.empty_cache()
    
    return components