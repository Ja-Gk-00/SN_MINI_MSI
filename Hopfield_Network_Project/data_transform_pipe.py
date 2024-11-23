import os
import glob
import pickle
from PIL import Image
import numpy as np

def preprocess_image(image_path, target_size=(200, 300), threshold=128, add_noise=False, noise_level=0.05):
    """
    Preprocesses an image: loads, converts to grayscale, resizes, binarizes, and optionally adds noise.

    Parameters:
    - image_path (str): Path to the image file.
    - target_size (tuple): Desired image size in pixels, as (width, height).
    - threshold (int): Threshold value for binarization (0-255).
    - add_noise (bool): Whether to add noise to the bitmap.
    - noise_level (float): Probability of flipping each bit when adding noise.

    Returns:
    - numpy.ndarray: Flattened binary bitmap of the image.
    """
    image = Image.open(image_path).convert('L')  
    image = image.resize(target_size)            
    image_array = np.array(image)               
    bitmap_array = (image_array > threshold).astype(int)

    if add_noise:
        bitmap_array = add_noise_to_bitmap(bitmap_array, noise_level)

    return bitmap_array.flatten() 

def add_noise_to_bitmap(bitmap_array, noise_level=0.05):
    """
    Adds salt-and-pepper noise to a binary bitmap array.

    Parameters:
    - bitmap_array (numpy.ndarray): Binary bitmap array (2D).
    - noise_level (float): Probability of flipping each bit (0 <= noise_level <= 1).

    Returns:
    - numpy.ndarray: Noisy bitmap array of the same shape.
    """
    noisy_bitmap = bitmap_array.copy()
    random_noise = np.random.rand(*bitmap_array.shape)
    flip_indices = random_noise < noise_level
    noisy_bitmap[flip_indices] = 1 - noisy_bitmap[flip_indices]
    return noisy_bitmap

def process_folder(input_folder, output_folder=None, pickle_file=None, target_size=(200, 300), threshold=128, add_noise=False, noise_level=0.05):
    """
    Processes all images in a folder and saves the results.

    Parameters:
    - input_folder (str): Path to the folder containing input images.
    - output_folder (str or None): Path to the folder to save bitmaps (as .npy files). 
                                   If None, files will not be saved.
    - pickle_file (str or None): Path to save all bitmaps in a single pickled file.
                                 If None, pickling is skipped.
    - target_size (tuple): Desired size of the output bitmaps (width, height).
    - threshold (int): Threshold for binarization.
    - add_noise (bool): Whether to add noise to the bitmaps.
    - noise_level (float): Probability of flipping each bit when adding noise.

    Returns:
    - List of NumPy arrays containing the processed bitmaps.
    """
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder '{input_folder}' does not exist.")
    
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder) 
    
    bitmap_list = []
    image_paths = glob.glob(os.path.join(input_folder, '*.*')) 
    
    for image_path in image_paths:
        try:
            bitmap = preprocess_image(
                image_path, 
                target_size=target_size, 
                threshold=threshold, 
                add_noise=add_noise, 
                noise_level=noise_level
            )
            bitmap_list.append(bitmap)
            
            if output_folder:
                file_name = os.path.splitext(os.path.basename(image_path))[0] + '.npy'
                output_path = os.path.join(output_folder, file_name)
                np.save(output_path, bitmap)
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    if pickle_file:
        with open(pickle_file, 'wb') as f:
            pickle.dump(bitmap_list, f)
    
    return bitmap_list

