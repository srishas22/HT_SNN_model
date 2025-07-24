import pandas as pd
import numpy as np
from typing import Tuple
import cv2
import os 

class SiameseDataset:
    # Set default target_size here.
    # Note: For InceptionV3, it should be (299, 299).
    # Your detector.py handles this by passing specific shapes for each model.
    def __init__(self, csv_path, image_dir: str = '', target_size: Tuple[int, int] = (224, 224)):
        self.pairs = pd.read_csv(csv_path)
        self.image_dir = image_dir # Use this for absolute paths if your CSV has relative ones
        self.target_size = target_size
        # Removed: self.transform as it's not used for Keras/TF

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # This __getitem__ method is typically used by PyTorch DataLoaders.
        # Since your detector.py uses get_data(), this method might not be directly called.
        # However, it's good practice to keep it consistent or remove if unused.
        # For simplicity, let's make it consistent with the Keras/TF pipeline.

        row = self.pairs.iloc[idx]
        img1_path = os.path.join(self.image_dir, row['image1']) # Assume image1 is GM
        img2_path = os.path.join(self.image_dir, row['image2']) # Assume image2 is PUI
        label = row['label']

        try:
            # Load images as BGR (default for cv2.imread)
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)

            if img1 is None:
                raise FileNotFoundError(f"Image not found at {img1_path}")
            if img2 is None:
                raise FileNotFoundError(f"Image not found at {img2_path}")

            # Ensure 3 channels (if original was grayscale/RGBA, convert to BGR 3-channel)
            if img1.ndim == 2: img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            elif img1.shape[2] == 4: img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2BGR)
            if img2.ndim == 2: img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
            elif img2.shape[2] == 4: img2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2BGR)

            # Resize images to target_size
            img1 = cv2.resize(img1, (self.target_size[1], self.target_size[0])) # cv2.resize expects (width, height)
            img2 = cv2.resize(img2, (self.target_size[1], self.target_size[0]))

            # Normalize pixel values to [0, 1] for Keras/TF models
            img1 = img1.astype(np.float32) / 255.0
            img2 = img2.astype(np.float32) / 255.0

        except FileNotFoundError:
            print(f"Warning: Image file not found for pair at index {idx}. Skipping this item.")
            return None # Return None if image is not found, get_data will handle it.
        except Exception as e:
            print(f"Error processing image pair at index {idx}: {e}")
            return None

        # Return GM (img1) and PUI (img2) in order, and label
        return img1, img2, label

    def get_data(self):
        x1_list, x2_list, y_list = [], [], []

        for idx in range(len(self)): # Iterate using self.__getitem__ for consistency
            item = self.__getitem__(idx)
            if item is not None: # Only append if __getitem__ successfully loaded the pair
                img1, img2, label = item
                x1_list.append(img1)
                x2_list.append(img2)
                y_list.append(label)
            # If item is None, it means an image was not found or error occurred, and it was skipped.

        x1_array = np.array(x1_list)
        x2_array = np.array(x2_list)
        y_array = np.array(y_list)

        return x1_array, x2_array, y_array
