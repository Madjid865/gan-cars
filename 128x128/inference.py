"""
inference.py - Image generation with trained GAN
Compatible with Generator that doesn't use img_size parameter
"""

import torch
from train_gan import Generator
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List
import os


class CarGenerator:
    """
    Wrapper class for generating car images using a trained GAN model
    """
    
    def __init__(self, checkpoint_path: str, latent_dim: int = 100, device: str = None):
        """
        Initialize the car generator
        
        Args:
            checkpoint_path: Path to the model checkpoint
            latent_dim: Dimension of the latent space
            device: Device to use ('cuda' or 'cpu')
        """
        self.latent_dim = latent_dim
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"🚀 Loading Car Generator...")
        
        # Initialize generator
        self.generator = Generator(latent_dim=latent_dim)
        
        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'G' in checkpoint:
                    self.generator.load_state_dict(checkpoint['G'])
                elif 'generator' in checkpoint:
                    self.generator.load_state_dict(checkpoint['generator'])
                else:
                    self.generator.load_state_dict(checkpoint)
            else:
                self.generator.load_state_dict(checkpoint)
                
            print(f"✅ Generator loaded from {checkpoint_path}")
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            raise
        
        # Move to device and set to eval mode
        self.generator.to(self.device)
        self.generator.eval()
        
        print(f"🔧 Device: {self.device}")
    
    def generate(self, num_images: int = 1, seed: int = None) -> List[Image.Image]:
        """
        Generate car images
        
        Args:
            num_images: Number of images to generate
            seed: Random seed for reproducibility
            
        Returns:
            List of PIL Images
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        images = []
        
        with torch.no_grad():
            for _ in range(num_images):
                # Generate random latent vector
                z = torch.randn(1, self.latent_dim, 1, 1, device=self.device)
                
                # Generate image
                fake_img = self.generator(z)
                
                # Convert to numpy and then PIL
                img_tensor = (fake_img[0] + 1) / 2
                img_tensor = img_tensor.clamp(0, 1)
                img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)
                
                # Upscale to 256x256 for better viewing
                pil_img = pil_img.resize((256, 256), Image.LANCZOS)
                images.append(pil_img)
                
        return images
    
    def generate_grid(self, num_images: int = 16, nrow: int = 4, seed: int = None) -> Image.Image:
        """
        Generate a grid of car images
        
        Args:
            num_images: Total number of images
            nrow: Number of images per row
            seed: Random seed
            
        Returns:
            PIL Image containing the grid
        """
        images = self.generate(num_images, seed)
        
        # Calculate grid dimensions
        ncol = nrow
        nrows = (num_images + nrow - 1) // nrow
        
        # Get image size
        img_size = images[0].size[0]
        
        # Create grid
        grid = Image.new('RGB', (ncol * img_size, nrows * img_size))
        
        for idx, img in enumerate(images):
            row = idx // ncol
            col = idx % ncol
            grid.paste(img, (col * img_size, row * img_size))
            
        return grid
    
    def interpolate(self, seed1: int, seed2: int, steps: int = 8) -> List[Image.Image]:
        """
        Interpolate between two latent vectors
        
        Args:
            seed1: First seed
            seed2: Second seed
            steps: Number of interpolation steps
            
        Returns:
            List of PIL Images showing the interpolation
        """
        # Generate two latent vectors
        torch.manual_seed(seed1)
        z1 = torch.randn(1, self.latent_dim, 1, 1, device=self.device)
        
        torch.manual_seed(seed2)
        z2 = torch.randn(1, self.latent_dim, 1, 1, device=self.device)
        
        images = []
        
        with torch.no_grad():
            for i in range(steps):
                # Linear interpolation
                alpha = i / (steps - 1)
                z = (1 - alpha) * z1 + alpha * z2
                
                # Generate image
                fake_img = self.generator(z)
                
                # Convert to PIL
                img_tensor = (fake_img[0] + 1) / 2
                img_tensor = img_tensor.clamp(0, 1)
                img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)
                
                # Upscale to 256x256 for better viewing
                pil_img = pil_img.resize((256, 256), Image.LANCZOS)
                images.append(pil_img)
                
        return images


def find_best_checkpoint(checkpoint_dir: str = "checkpoints") -> str:
    """
    Find the best checkpoint file in the directory
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to the best checkpoint
    """
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Look for .pth files
    pth_files = list(checkpoint_path.glob("*.pth"))
    
    if not pth_files:
        raise FileNotFoundError(f"No .pth files found in {checkpoint_dir}")
    
    # Return the first one found (or implement logic to find "best")
    return str(pth_files[0])


if __name__ == "__main__":
    # Example usage
    checkpoint = find_best_checkpoint("checkpoints")
    gen = CarGenerator(checkpoint)
    
    # Generate a single image
    images = gen.generate(num_images=1, seed=42)
    images[0].save("generated_car.png")
    print("✅ Saved generated_car.png")
    
    # Generate a grid
    grid = gen.generate_grid(num_images=16, nrow=4, seed=123)
    grid.save("generated_grid.png")
    print("✅ Saved generated_grid.png")
    
    # Generate interpolation
    interp_images = gen.interpolate(seed1=100, seed2=200, steps=8)
    for i, img in enumerate(interp_images):
        img.save(f"interpolation_{i:02d}.png")
    print(f"✅ Saved {len(interp_images)} interpolation images")