"""
app_enhanced.py - Enhanced Car GAN Interface
Professional interface with advanced features and error handling
"""

import gradio as gr
import torch
from pathlib import Path
import random
import os
import sys
from inference import CarGenerator, find_best_checkpoint
from advanced_features import (
    generate_variations, generate_mega_showcase, create_gif_from_images
)
import tempfile


# Model download link
CHECKPOINT_DOWNLOAD_LINK = "https://drive.google.com/file/d/1g-AfY58T2zF_sSTQl3wEbNdUg_tbF9s0/view?usp=sharing"
CHECKPOINT_DIR = "checkpoints"


def check_checkpoint_exists():
    """Check if checkpoint directory exists and contains .pth files"""
    checkpoint_path = Path(CHECKPOINT_DIR)
    
    if not checkpoint_path.exists():
        return False, "Checkpoint directory not found"
    
    pth_files = list(checkpoint_path.glob("*.pth"))
    
    if not pth_files:
        return False, "No checkpoint files found"
    
    return True, f"Found {len(pth_files)} checkpoint file(s)"


def create_error_message():
    """Create a formatted error message with download instructions"""
    return f"""
    ⚠️ **CHECKPOINT FILE NOT FOUND**
    
    The model checkpoint file (.pth) is missing from the `{CHECKPOINT_DIR}/` directory.
    
    **📥 How to fix this:**
    
    1. Download the pre-trained model from Google Drive:
       👉 [{CHECKPOINT_DOWNLOAD_LINK}]({CHECKPOINT_DOWNLOAD_LINK})
    
    2. Place the downloaded `.pth` file in the following directory:
       📁 `{os.path.abspath(CHECKPOINT_DIR)}/`
    
    3. Restart the application
    
    **Alternative:** If you're training your own model, make sure the checkpoint 
    is saved in the correct directory before launching the interface.
    """


# Initialize the generator
print("🚀 Loading Car Generator...")
car_gen = None
checkpoint_error = None

try:
    exists, message = check_checkpoint_exists()
    
    if not exists:
        checkpoint_error = create_error_message()
        print(f"❌ {message}")
        print(f"\n📥 Please download the checkpoint from: {CHECKPOINT_DOWNLOAD_LINK}")
        print(f"📁 Place it in: {os.path.abspath(CHECKPOINT_DIR)}/")
    else:
        checkpoint = find_best_checkpoint(CHECKPOINT_DIR)
        car_gen = CarGenerator(checkpoint)
        print(f"✅ Loaded checkpoint: {checkpoint}")
        
except Exception as e:
    checkpoint_error = create_error_message()
    print(f"❌ Error loading checkpoint: {e}")


def generate_single_car(seed=None, use_random_seed=True):
    """Generate a single car image"""
    if car_gen is None:
        return None, checkpoint_error
    
    if use_random_seed:
        seed = random.randint(0, 999999)
    
    try:
        images = car_gen.generate(num_images=1, seed=seed)
        return images[0], f"✅ Generated car with seed: {seed}"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def generate_multiple_cars(num_images=4, grid_cols=2, seed=None, use_random_seed=True):
    """Generate multiple cars in a grid"""
    if car_gen is None:
        return None, checkpoint_error
    
    if use_random_seed:
        seed = random.randint(0, 999999)
    
    try:
        grid = car_gen.generate_grid(num_images=num_images, nrow=grid_cols, seed=seed)
        return grid, f"✅ Generated {num_images} cars with seed: {seed}"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def interpolate_cars(start_seed, end_seed, steps=8):
    """Create smooth transition between two cars"""
    if car_gen is None:
        return None, checkpoint_error
    
    try:
        images = car_gen.interpolate(start_seed, end_seed, steps)
        return images, f"✅ Created {steps} interpolation steps from seed {start_seed} to {end_seed}"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def interpolate_and_export_gif(start_seed, end_seed, steps=16, duration=200):
    """Create interpolation and export as GIF"""
    if car_gen is None:
        return None, None, checkpoint_error
    
    try:
        images = car_gen.interpolate(start_seed, end_seed, steps)
        
        # Create temporary GIF file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.gif')
        create_gif_from_images(images, temp_file.name, duration)
        
        return images, temp_file.name, f"✅ Created GIF with {steps} frames!"
    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"


def create_variations(base_seed, num_vars=16, strength=0.3):
    """Generate variations around a seed"""
    if car_gen is None:
        return None, checkpoint_error
    
    try:
        images = generate_variations(
            car_gen.generator, base_seed, num_vars, strength, 
            car_gen.device, car_gen.latent_dim
        )
        return images, f"✅ Generated {num_vars} variations of seed {base_seed}"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def create_mega_grid(num_cars=64):
    """Generate massive showcase grid"""
    if car_gen is None:
        return None, checkpoint_error
    
    try:
        grid = generate_mega_showcase(
            car_gen.generator, num_cars,
            car_gen.device, car_gen.latent_dim
        )
        return grid, f"✅ Generated {num_cars} unique cars!"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def random_seed_generator():
    """Generate a random seed"""
    return random.randint(0, 999999)


# ============================================
# GRADIO INTERFACE
# ============================================

with gr.Blocks(title="🚗 AI Car Generator") as app:
    
    gr.Markdown(
        """
        # 🚗 AI Car Generator
        ### Generate realistic car images using our trained DCGAN model
        """
    )
    
    # Show error message if checkpoint not found
    if checkpoint_error:
        with gr.Row():
            gr.Markdown(
                f"""
                <div style="background-color: #fff3cd; border: 2px solid #ffc107; 
                            border-radius: 8px; padding: 20px; margin: 10px 0;">
                {checkpoint_error}
                </div>
                """,
                elem_classes="error-message"
            )
    
    with gr.Tabs():
        
        # ==================== TAB 1: Single Generation ====================
        with gr.Tab("🎨 Single Car"):
            gr.Markdown("### Generate one car at a time")
            
            with gr.Row():
                with gr.Column(scale=1):
                    single_use_random = gr.Checkbox(label="Use Random Seed", value=True)
                    single_seed = gr.Number(label="Seed", value=42, precision=0)
                    single_random_btn = gr.Button("🎲 Random Seed", size="sm")
                    single_generate_btn = gr.Button("✨ Generate Car", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    single_output = gr.Image(label="Generated Car", height=500, width=500)
                    single_status = gr.Textbox(label="Status", interactive=False)
            
            single_random_btn.click(fn=random_seed_generator, outputs=single_seed)
            single_generate_btn.click(
                fn=generate_single_car,
                inputs=[single_seed, single_use_random],
                outputs=[single_output, single_status]
            )
        
        # ==================== TAB 2: Batch Generation ====================
        with gr.Tab("🎯 Batch Generation"):
            gr.Markdown("### Generate multiple cars in a grid")
            
            with gr.Row():
                with gr.Column(scale=1):
                    batch_num = gr.Slider(minimum=1, maximum=64, step=1, value=16, label="Number of Cars")
                    batch_cols = gr.Slider(minimum=1, maximum=8, step=1, value=4, label="Grid Columns")
                    batch_use_random = gr.Checkbox(label="Use Random Seed", value=True)
                    batch_seed = gr.Number(label="Seed", value=42, precision=0)
                    batch_random_btn = gr.Button("🎲 Random Seed", size="sm")
                    batch_generate_btn = gr.Button("✨ Generate Grid", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    batch_output = gr.Image(label="Generated Grid", height=600, width=600)
                    batch_status = gr.Textbox(label="Status", interactive=False)
            
            batch_random_btn.click(fn=random_seed_generator, outputs=batch_seed)
            batch_generate_btn.click(
                fn=generate_multiple_cars,
                inputs=[batch_num, batch_cols, batch_seed, batch_use_random],
                outputs=[batch_output, batch_status]
            )
        
        # ==================== TAB 3: Interpolation ====================
        with gr.Tab("🔄 Interpolation"):
            gr.Markdown("### Create smooth transitions between two cars")
            
            with gr.Row():
                with gr.Column(scale=1):
                    interp_start = gr.Number(label="Start Seed", value=100, precision=0)
                    interp_start_random = gr.Button("🎲 Random Start", size="sm")
                    
                    interp_end = gr.Number(label="End Seed", value=200, precision=0)
                    interp_end_random = gr.Button("🎲 Random End", size="sm")
                    
                    interp_steps = gr.Slider(minimum=3, maximum=20, step=1, value=8, label="Steps")
                    interp_generate_btn = gr.Button("🔄 Create Interpolation", variant="primary", size="lg")
                    
                    gr.Markdown("---")
                    gr.Markdown("### 🎬 Export as GIF")
                    gif_steps = gr.Slider(minimum=8, maximum=30, step=1, value=16, label="Frames")
                    gif_duration = gr.Slider(minimum=50, maximum=500, step=50, value=200, label="Frame Duration (ms)")
                    gif_generate_btn = gr.Button("🎞️ Create Animated GIF", variant="secondary")
                
                with gr.Column(scale=2):
                    interp_output = gr.Gallery(label="Interpolation Sequence", height=600, columns=4)
                    gif_output = gr.File(label="Download GIF")
                    interp_status = gr.Textbox(label="Status", interactive=False)
            
            interp_start_random.click(fn=random_seed_generator, outputs=interp_start)
            interp_end_random.click(fn=random_seed_generator, outputs=interp_end)
            interp_generate_btn.click(
                fn=interpolate_cars,
                inputs=[interp_start, interp_end, interp_steps],
                outputs=[interp_output, interp_status]
            )
            gif_generate_btn.click(
                fn=interpolate_and_export_gif,
                inputs=[interp_start, interp_end, gif_steps, gif_duration],
                outputs=[interp_output, gif_output, interp_status]
            )
        
        # ==================== TAB 4: Variations ====================
        with gr.Tab("🎲 Variations"):
            gr.Markdown("### Explore variations around a favorite design")
            
            with gr.Row():
                with gr.Column(scale=1):
                    var_seed = gr.Number(label="Base Seed", value=442, precision=0)
                    var_random_btn = gr.Button("🎲 Random Seed", size="sm")
                    
                    var_num = gr.Slider(minimum=4, maximum=36, step=4, value=16, 
                                       label="Number of Variations")
                    var_strength = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.3,
                                            label="Variation Strength",
                                            info="Higher = more different")
                    var_generate_btn = gr.Button("✨ Generate Variations", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    var_output = gr.Gallery(label="Variations", height=600, columns=4)
                    var_status = gr.Textbox(label="Status", interactive=False)
            
            var_random_btn.click(fn=random_seed_generator, outputs=var_seed)
            var_generate_btn.click(
                fn=create_variations,
                inputs=[var_seed, var_num, var_strength],
                outputs=[var_output, var_status]
            )
        
        # ==================== TAB 5: Mega Showcase ====================
        with gr.Tab("🎆 Mega Showcase"):
            gr.Markdown("### Generate massive grid of unique cars")
            
            with gr.Row():
                with gr.Column(scale=1):
                    mega_num = gr.Slider(minimum=16, maximum=100, step=4, value=64,
                                        label="Number of Cars")
                    mega_generate_btn = gr.Button("🎆 Generate Showcase", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    mega_output = gr.Image(label="Mega Showcase", height=700)
                    mega_status = gr.Textbox(label="Status", interactive=False)
            
            mega_generate_btn.click(
                fn=create_mega_grid,
                inputs=[mega_num],
                outputs=[mega_output, mega_status]
            )
        
        # ==================== TAB 6: Info ====================
        with gr.Tab("ℹ️ Info"):
            gr.Markdown(
                f"""
                ## 📊 Model Information
                
                **Architecture:** DCGAN (Deep Convolutional GAN)
                - **Generator:** Transforms random noise into 128x128 car images
                - **Discriminator:** Learns to distinguish real from fake
                - **Training:** Adversarial learning process
                
                ## 🎮 Features
                
                ### 🎨 Single Car
                Generate one car at a time with full control over the random seed.
                
                ### 🎯 Batch Generation
                Create multiple cars in a customizable grid layout.
                
                ### 🔄 Interpolation
                - Smoothly morph between two different car designs
                - Export as animated GIF for sharing
                
                ### 🎲 Variations
                - Start with a car you like (save its seed!)
                - Generate similar designs by exploring nearby latent space
                - Control variation strength
                
                ### 🎆 Mega Showcase
                - Generate 64-100 unique cars at once
                - Shows variety and quality of the model
                
                ## 💡 Tips
                
                - **Seeds** control randomness - same seed = same car
                - Try different seeds to get diverse results
                - Interpolation shows smooth transitions in latent space
                - Download images you like by right-clicking
                
                ## 📥 Model Download
                
                If you're missing the checkpoint file, download it here:
                
                **[Download Pre-trained Model]({CHECKPOINT_DOWNLOAD_LINK})**
                
                Place the `.pth` file in: `{os.path.abspath(CHECKPOINT_DIR)}/`
                
                ## 👥 Team
                
                This project was created by:
                - Madjid (Group B)
                - Nassim (Group B)
                - Hazem (Group C)
                - Kim (Group A)
                """
            )
    
    gr.Markdown(
        """
        ---
        Made with ❤️ using Gradio | Powered by PyTorch DCGAN
        """
    )


if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Launching Car Generator Interface...")
    print("="*50 + "\n")
    
    if checkpoint_error:
        print("⚠️  WARNING: No checkpoint file found!")
        print(f"📥 Download from: {CHECKPOINT_DOWNLOAD_LINK}")
        print(f"📁 Place in: {os.path.abspath(CHECKPOINT_DIR)}/")
        print("\nThe interface will launch, but generation will not work until you add the checkpoint.\n")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7863,
        share=False,
        show_error=True
    )