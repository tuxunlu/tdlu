import os
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import functional as TF
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Worker function must be at the top level for multiprocessing
def process_single_image(img_path):
    """
    Process a single image and return its sum, squared sum, and pixel count.
    """
    try:
        img = Image.open(img_path)
        
        # Convert to Tensor. PILToTensor preserves bit-depth (e.g. 16-bit).
        # convert_image_dtype scales values to [0.0, 1.0] based on dtype.
        img_t = transforms.PILToTensor()(img)
        img_t = TF.convert_image_dtype(img_t, torch.float32)
        
        # img_t is shape (C, H, W). 
        # We permute to (H, W, C) so that when we reshape to (-1, C), 
        # the channels stay grouped correctly for each pixel.
        img_t = img_t.permute(1, 2, 0)
        
        # Reshape to (Num_Pixels, Channels)
        # img_t.shape[-1] will be 1 for grayscale, 3 for RGB
        pixels = img_t.numpy().reshape(-1, img_t.shape[-1])
        
        c_sum = np.sum(pixels, axis=0)
        c_sum_sq = np.sum(pixels ** 2, axis=0)
        count = pixels.shape[0]
        
        return c_sum, c_sum_sq, count
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def compute_mean_std_parallel(dataset_path, extension="*.png", num_workers=None):
    """
    Computes dataset stats in parallel, adapting to 1 or 3 channels automatically.
    """
    search_path = os.path.join(dataset_path, extension)
    image_paths = glob.glob(search_path)
    
    if not image_paths:
        print(f"No images found in {search_path}")
        return

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    print(f"Found {len(image_paths)} images.")
    print(f"Processing with {num_workers} workers...")

    # Accumulators (Initialized to None to adapt to image channels)
    total_sum = None
    total_sum_sq = None
    total_count = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_single_image, image_paths), total=len(image_paths)))
        
        for res in results:
            if res is None:
                continue
                
            c_sum, c_sum_sq, count = res
            
            # Initialize accumulators on first successful result
            if total_sum is None:
                total_sum = np.zeros_like(c_sum)
                total_sum_sq = np.zeros_like(c_sum_sq)
            
            total_sum += c_sum
            total_sum_sq += c_sum_sq
            total_count += count

    if total_count == 0:
        print("Error: Total pixel count is zero. All images failed to process.")
        return

    # Final Calculations
    mean = total_sum / total_count
    std = np.sqrt((total_sum_sq / total_count) - (mean ** 2))
    
    print("\n" + "="*30)
    print(f"Dataset: {os.path.basename(dataset_path)}")
    print(f"Images Processed: {len(image_paths)}")
    print(f"Channels Detected: {len(mean)}")
    print("="*30)
    print(f"Mean: {mean}")
    print(f"Std:  {std}")
    print("="*30)
    
    print("\nFor PyTorch transforms.Normalize:")
    # Format output based on number of channels
    mean_str = ", ".join([f"{m:.4f}" for m in mean])
    std_str = ", ".join([f"{s:.4f}" for s in std])
    print(f"mean=[{mean_str}]")
    print(f"std=[{std_str}]")
    
    # Tip for Grayscale -> RGB usage
    if len(mean) == 1:
        print("\nNote: If you plan to use a 3-channel model (like ResNet) with these images repeated,")
        print("you can repeat these stats: ")
        print(f"mean=[{mean[0]:.4f}, {mean[0]:.4f}, {mean[0]:.4f}]")
        print(f"std=[{std[0]:.4f}, {std[0]:.4f}, {std[0]:.4f}]")

if __name__ == "__main__":
    folder_path = "/beacon-scratch/tuxunlu/git/tdlu/KOMEN/WUSTL_png_nomarker_16"
    
    if os.path.exists(folder_path):
        compute_mean_std_parallel(folder_path)
    else:
        print(f"Folder '{folder_path}' not found.")