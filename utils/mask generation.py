import csv

# IMPORTANT: Make sure this path is correct for your file
CSV_FILE_PATH = 'data/pairs/FCSN_train_pairs.csv' 

print(f"--- Debugging CSV File: {CSV_FILE_PATH} ---")

try:
    with open(CSV_FILE_PATH, 'r', newline='') as f:
        reader = csv.reader(f)
        
        # Read the header
        header = next(reader)
        print(f"Header found: {header} (Columns: {len(header)})")
        print("-" * 40)

        # Check every subsequent row
        found_problem = False
        for i, row in enumerate(reader):
            num_columns = len(row)
            if num_columns != 3:
                print(f"!!! PROBLEM FOUND ON LINE {i+2} !!!")
                print(f"    Expected 3 columns, but the reader found {num_columns}.")
                print(f"    Row Content: {row}")
                print("-" * 40)
                found_problem = True
    
    if not found_problem:
        print("Success! All rows have exactly 3 columns.")

except FileNotFoundError:
    print(f"ERROR: The file was not found at '{CSV_FILE_PATH}'. Please check the path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

print("--- Debugging Complete ---")

# import cv2
# import csv
# import os
# import numpy as np
# import skimage
# from skimage.metrics import structural_similarity


# from image_utils import preprocess_image, align_images

# def create_ssim_mask_rgb(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
#     """Creates a high-quality difference mask from two pre-aligned and pre-processed RGB images."""
#     (score, diff) = structural_similarity(img1, img2, full=True, channel_axis=-1)
#     diff = (diff * 255).astype("uint8")
#     mask = cv2.threshold(diff, 240, 255, cv2.THRESH_BINARY_INV)[1]
#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     return mask

# def process_csv_and_generate_masks(input_csv, output_csv, mask_dir):
#     """Reads an input CSV, generates masks for each pair, and writes to an output CSV."""
#     os.makedirs(mask_dir, exist_ok=True)
    
#     with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
#         reader = csv.reader(infile)
#         writer = csv.writer(outfile)
        
#         # Read header and write new header with 'mask' column
#         header = next(reader)
#         writer.writerow(header + ['mask'])
        
#         print(f"Processing {input_csv}...")
#         for i, row in enumerate(reader):
#             golden_path, pui_path, label = row
#             label = int(label)
            
#             mask_path = "" # Default to empty path
            
#             # We only generate masks for pairs that represent a change (label=1)
#             if label == 1:
#                 try:
#                     # Define a unique name for the mask file
#                     pui_filename = os.path.basename(pui_path)
#                     mask_filename = f"mask_{pui_filename}"
#                     mask_path = os.path.join(mask_dir, mask_filename)

#                     # Load original images
#                     golden_img_orig = cv2.imread(golden_path)
#                     pui_img_orig = cv2.imread(pui_path)

#                     if golden_img_orig is None or pui_img_orig is None:
#                         print(f"Warning: Could not load images for row {i+1}. Skipping.")
#                         continue
                    
#                     # Full pipeline: Align -> Preprocess -> Create Mask
#                     aligned_pui, golden_img = align_images(pui_img_orig, golden_img_orig)
#                     proc_aligned_pui = preprocess_image(aligned_pui)
#                     proc_golden_img = preprocess_image(golden_img)
#                     final_mask = create_ssim_mask_rgb(proc_golden_img, proc_aligned_pui)
                    
#                     # Save the generated mask
#                     cv2.imwrite(mask_path, final_mask)
                    
#                 except Exception as e:
#                     print(f"Error processing row {i+1}: {e}. Mask path will be empty.")
#                     mask_path = "" # Reset path on error
            
#             # For clean pairs (label=0), the mask path remains empty
#             writer.writerow(row + [mask_path])
            
#             if (i + 1) % 10 == 0:
#                 print(f"  ...processed {i+1} rows.")
    
#     print(f"Finished processing. New CSV saved to {output_csv}")

# # train csv
# process_csv_and_generate_masks(input_csv='data/pairs/FCSN_train_pairs.csv',
#         output_csv='data/pairs/train_pairs_with_masks.csv',
#         mask_dir='data/masks/train')
# # test csv
# process_csv_and_generate_masks(input_csv='data/pairs/FCSN_train_pairs.csv',
#         output_csv='data/pairs/train_pairs_with_masks.csv',
#         mask_dir='data/masks/train')
