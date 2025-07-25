import cv2
import csv
import os
import numpy as np
import skimage
from image_utils import preprocess_image, align_images
import traceback
from skimage.metrics import structural_similarity as ssim

def create_ssim_mask_rgb(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Creates a binary mask highlighting differences between two aligned RGB images using SSIM.
    
    Args:
        img1: First RGB image (e.g., golden model).
        img2: Second RGB image (e.g., PUI image).
    
    Returns:
        Binary mask (uint8) where 255 = changed region, 0 = unchanged.
    """
    # Ensure both images are float and in range [0, 1] for SSIM
    if img1.dtype != np.float32:
        img1 = img1.astype(np.float32) / 255.0
    if img2.dtype != np.float32:
        img2 = img2.astype(np.float32) / 255.0

    # Compute SSIM over RGB channels
    score, diff = ssim(img1, img2, full=True, channel_axis=-1, data_range=1.0)

    # Convert SSIM map to uint8 for thresholding
    diff = (diff * 255).astype(np.uint8)

    # Threshold: invert high-similarity zones (keep only significant changes)
    _, mask = cv2.threshold(diff, 240, 255, cv2.THRESH_BINARY_INV)

    # Morphological closing to remove small holes inside detected regions
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


# def create_ssim_mask_rgb(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
#     """Creates a high-quality difference mask from two pre-aligned and pre-processed RGB images."""
#     (score, diff) = structural_similarity(img1, img2, full=True, channel_axis=-1)
#     diff = (diff * 255).astype("uint8")
#     mask = cv2.threshold(diff, 240, 255, cv2.THRESH_BINARY_INV)[1]
#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     return mask

def process_csv_and_generate_masks(input_csv, output_csv, mask_dir):
    """Reads an input CSV, generates masks for each pair, and writes to an output CSV."""
    os.makedirs(mask_dir, exist_ok=True)
    
    with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['mask']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        # reader = csv.reader(infile)
        # writer = csv.writer(outfile)
        
        # # Read header and write new header with 'mask' column
        # header = next(reader)
        # writer.writerow(header + ['mask'])


        print(f"Processing {input_csv}...")
        for i, row in enumerate(reader):
            if i>0:
                break 
            # golden_path, pui_path, label = row
            golden_path = row['image1']
            pui_path = row['image2']
            label = int(row['label'])
            
            mask_path = "" # Default to empty path
                      
            try:
                # Define a unique name for the mask file
                pui_filename = os.path.basename(pui_path)
                mask_filename = f"mask_{pui_filename}"
                mask_path = os.path.join(mask_dir, mask_filename)

                # Load original images
                golden_img_orig = cv2.imread(golden_path)
                pui_img_orig = cv2.imread(pui_path)

                if golden_img_orig is None or pui_img_orig is None:
                    print(f"Warning: Could not load images for row {i+1}. Skipping.")
                    continue
                    
                # Full pipeline: Align -> Preprocess -> Create Mask
                aligned_pui, golden_img = align_images(pui_img_orig, golden_img_orig)
                proc_aligned_pui = preprocess_image(aligned_pui)
                proc_golden_img = preprocess_image(golden_img)
                final_mask = create_ssim_mask_rgb(proc_golden_img, proc_aligned_pui)
                    
                # Save the generated mask
                cv2.imwrite(mask_path, final_mask)
                    

            except Exception as e:
                print(f"Error processing row {i+1}: {e}")
                traceback.print_exc()
                mask_path = ""
        
        row['mask'] = mask_path
        writer.writerow(row)

            
        if (i + 1) % 10 == 0:
            print(f"  ...processed {i+1} rows.")
    
    print(f"Finished processing. New CSV saved to {output_csv}")

# train csv
process_csv_and_generate_masks(input_csv='data/pairs/FCSN_train_pairs.csv',
        output_csv='data/pairs/train_pairs_with_masks.csv',
        mask_dir='data/masks/train')
# test csv
process_csv_and_generate_masks(input_csv='data/pairs/FCSN_train_pairs.csv',
        output_csv='data/pairs/train_pairs_with_masks.csv',
        mask_dir='data/masks/train')
