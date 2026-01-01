"""
Advanced features for the Car GAN Interface
Impressive capabilities for presentations and demos
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
from typing import List


def generate_variations(generator, seed: int, num_variations: int = 16, 
                       variation_strength: float = 0.3, device: str = "cpu", latent_dim: int = 100) -> List[Image.Image]:
    """
    Generate variations around a specific seed
    Perfect for exploring the neighborhood of a good design
    """
    generator.eval()
    
    # Create base latent vector
    torch.manual_seed(seed)
    base_z = torch.randn(1, latent_dim, 1, 1, device=device)
    
    images = []
    with torch.no_grad():
        for i in range(num_variations):
            # Add controlled noise to create variation
            noise = torch.randn(1, latent_dim, 1, 1, device=device) * variation_strength
            varied_z = base_z + noise
            
            # Generate
            fake_img = generator(varied_z)
            
            # Convert to PIL
            img_tensor = (fake_img[0] + 1) / 2
            img_tensor = img_tensor.clamp(0, 1)
            img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            
            # Upscale
            pil_img = pil_img.resize((256, 256), Image.LANCZOS)
            images.append(pil_img)
            
    return images


def create_training_comparison(generator, checkpoint_paths: List[str], seed: int, 
                               device: str = "cpu", latent_dim: int = 100) -> List[Image.Image]:
    """
    Compare same seed across different training checkpoints
    Shows evolution of training!
    """
    torch.manual_seed(seed)
    z = torch.randn(1, latent_dim, 1, 1, device=device)
    
    images = []
    
    for ckpt_path in checkpoint_paths:
        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location=device)
        if 'G' in ckpt:
            generator.load_state_dict(ckpt['G'])
        else:
            generator.load_state_dict(ckpt)
        
        generator.eval()
        
        with torch.no_grad():
            fake_img = generator(z)
            
            # Convert to PIL
            img_tensor = (fake_img[0] + 1) / 2
            img_tensor = img_tensor.clamp(0, 1)
            img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            pil_img = pil_img.resize((256, 256), Image.LANCZOS)
            
            # Add label
            draw = ImageDraw.Draw(pil_img)
            epoch = ckpt.get('epoch', '?')
            label = f"Epoch {epoch}"
            draw.text((10, 10), label, fill=(255, 255, 0))
            
            images.append(pil_img)
    
    return images


def generate_latent_grid(generator, rows: int = 8, cols: int = 8, 
                         seed_start: int = 0, device: str = "cpu", latent_dim: int = 100) -> Image.Image:
    """
    Create a grid showing systematic exploration of latent space
    Visual map of the design space!
    """
    generator.eval()
    
    # Create seeds for grid
    torch.manual_seed(seed_start)
    
    # Generate boundary vectors
    z_row_start = torch.randn(1, latent_dim, 1, 1, device=device)
    z_row_end = torch.randn(1, latent_dim, 1, 1, device=device)
    z_col_start = torch.randn(1, latent_dim, 1, 1, device=device)
    z_col_end = torch.randn(1, latent_dim, 1, 1, device=device)
    
    # Create grid
    cell_size = 64
    grid_img = Image.new('RGB', (cols * cell_size, rows * cell_size))
    
    with torch.no_grad():
        for i in range(rows):
            for j in range(cols):
                # Bilinear interpolation
                alpha_row = i / (rows - 1) if rows > 1 else 0
                alpha_col = j / (cols - 1) if cols > 1 else 0
                
                z_row = (1 - alpha_row) * z_row_start + alpha_row * z_row_end
                z_col = (1 - alpha_col) * z_col_start + alpha_col * z_col_end
                z = (z_row + z_col) / 2
                
                # Generate
                fake_img = generator(z)
                
                # Convert to PIL
                img_tensor = (fake_img[0] + 1) / 2
                img_tensor = img_tensor.clamp(0, 1)
                img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)
                
                # Paste into grid
                grid_img.paste(pil_img, (j * cell_size, i * cell_size))
    
    return grid_img


def create_gif_from_images(images: List[Image.Image], output_path: str, duration: int = 200):
    """
    Create animated GIF from list of images
    Perfect for sharing interpolations!
    """
    if not images:
        return False
    
    # Save as GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=True
    )
    return True


def generate_random_walk(generator, start_seed: int, steps: int = 20, 
                        step_size: float = 0.1, device: str = "cpu", latent_dim: int = 100) -> List[Image.Image]:
    """
    Random walk through latent space
    Creates smooth continuous generation - hypnotic for demos!
    """
    generator.eval()
    
    # Start position
    torch.manual_seed(start_seed)
    z = torch.randn(1, latent_dim, 1, 1, device=device)
    
    images = []
    
    with torch.no_grad():
        for _ in range(steps):
            # Generate current
            fake_img = generator(z)
            
            # Convert to PIL
            img_tensor = (fake_img[0] + 1) / 2
            img_tensor = img_tensor.clamp(0, 1)
            img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            pil_img = pil_img.resize((256, 256), Image.LANCZOS)
            images.append(pil_img)
            
            # Random step
            direction = torch.randn(1, latent_dim, 1, 1, device=device)
            direction = direction / direction.norm() * step_size
            z = z + direction
    
    return images


def generate_mega_showcase(generator, num_cars: int = 64, device: str = "cpu", latent_dim: int = 100) -> Image.Image:
    """
    Generate massive grid of cars
    Visual impact for presentations!
    """
    generator.eval()
    
    # Calculate grid size
    grid_size = int(np.sqrt(num_cars))
    cell_size = 64
    
    grid_img = Image.new('RGB', (grid_size * cell_size, grid_size * cell_size))
    
    with torch.no_grad():
        for i in range(grid_size):
            for j in range(grid_size):
                # Generate random car
                z = torch.randn(1, latent_dim, 1, 1, device=device)
                fake_img = generator(z)
                
                # Convert to PIL
                img_tensor = (fake_img[0] + 1) / 2
                img_tensor = img_tensor.clamp(0, 1)
                img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)
                
                # Paste into grid
                grid_img.paste(pil_img, (j * cell_size, i * cell_size))
    
    return grid_img
