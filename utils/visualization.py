import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

def visualize_results(pui_image_path: str, gm_image_path: str, results: Dict[str, any]) -> None:
    pui_image = cv2.imread(pui_image_path, cv2.IMREAD_GRAYSCALE)
    gm_image = cv2.imread(gm_image_path, cv2.IMREAD_GRAYSCALE)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(gm_image, cmap='gray')
    axes[0].set_title('Golden Model (GM)')
    axes[0].axis('off')

    axes[1].imshow(pui_image, cmap='gray')
    axes[1].set_title('PCB Under Inspection (PUI)')
    axes[1].axis('off')

    result_image = pui_image.copy()
    if len(result_image.shape) == 2:
        result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2RGB)

    for region_info in results['regions']:
        x, y, w, h = region_info['region']
        confidence = region_info['confidence']
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(result_image, f'HT: {confidence:.2f}', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    axes[2].imshow(result_image)
    axes[2].set_title(f'Detection Results\nTrojans: {len(results["regions"])}')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
