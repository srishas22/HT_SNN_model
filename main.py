from PIL import Image
import matplotlib.pyplot as plt
import os

from core.detector import HardwareTrojanDetector
from utils.visualization import visualize_results
from utils.data_loader import SiameseDataset


if __name__ == "__main__":
    detector = HardwareTrojanDetector()

    print("Training Siamese Neural Network...")
    results = detector.train_model(epochs=10, batch_size=32)

    print("\nTest Accuracy:", results['test_accuracy'])
    full_dataset = SiameseDataset('data/pairs/train_pairs.csv')
    dataset = SiameseDataset('data/pairs/test_pairs.csv')

    # Access the underlying DataFrame
    df = dataset.pairs

    # Get the first PUI and GM image paths
    pui_path = df.loc[0, 'image2']  # PCB Under Inspection
    gm_path = df.loc[0, 'image1']   # Golden Model

    #detection_results = detector.detect_hardware_trojans(pui_path, gm_path)

    #print("\nDetection Summary:", detection_results['status'])
    #print("Detected Trojan Regions:", detection_results['regions'])

    #visualize_results(pui_path, gm_path, detection_results)
