# detector.py

from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import cv2
from tensorflow import keras # Assuming this refers to tf.keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.image_utils import (
    preprocess_image,
    align_images,
    identify_suspicious_regions,
    crop_and_normalize_regions
)
from utils.data_loader import SiameseDataset


class HardwareTrojanDetector:
    # Default input shape for most ImageNet models (ResNet, VGG, EfficientNetB0)
    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3)):
        self.input_shape = input_shape
        self.siamese_model = None
        self.detection_threshold = 0.5

    def train_model(self, epochs: int = 20, batch_size: int = 32) -> Dict[str, Any]:
        """Enhanced training method optimized for hardware trojan detection"""
        from sklearn.utils.class_weight import compute_class_weight
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        import numpy as np
        
        # Load datasets from the custom SiameseDataset class
        full_dataset = SiameseDataset('data/pairs/train_pairs.csv')
        test_dataset = SiameseDataset('data/pairs/test_pairs.csv')
        
        # Split into training and validation with stratification to preserve class balance
        val_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        
        x1, x2, y = full_dataset.get_data()  # returns np arrays
        
        # Stratified split to maintain class balance
        from sklearn.model_selection import train_test_split
        x1_train, x1_val, x2_train, x2_val, y_train, y_val = train_test_split(
            x1, x2, y, test_size=0.3, random_state=42, stratify=y
        )
        
        x1_test, x2_test, y_test = test_dataset.get_data()
        
        print(f"Training samples: {len(y_train)}")
        print(f"Validation samples: {len(y_val)}")
        print(f"Test samples: {len(y_test)}")
        print(f"Training positive pairs: {np.sum(y_train)} ({np.mean(y_train)*100:.1f}%)")
        print(f"Validation positive pairs: {np.sum(y_val)} ({np.mean(y_val)*100:.1f}%)")
        print(f"Test positive pairs: {np.sum(y_test)} ({np.mean(y_test)*100:.1f}%)")
        
        # --- MODIFIED: Import all four specific model builders. No data_augmentation_for_trojans import. ---
        from core.siamese import (
            # build_resnet50_siamese_network,
            # build_vgg16_siamese_network,
            # build_inceptionv3_siamese_network,
            # build_efficientnetb0_siamese_network,
            # get_trojan_detection_callbacks,
            build_fully_convolutional_siamese_network
        )
        
        # Instantiate all four models ---
        models = {
            # 'ResNet50_Siamese': build_resnet50_siamese_network(input_shape=self.input_shape, embedding_dim=128),
            # 'VGG16_Siamese': build_vgg16_siamese_network(input_shape=self.input_shape, embedding_dim=128),
            # # IMPORTANT: InceptionV3 typically expects 299x299 input.
            # # Ensure your data pipeline or preprocessing aligns with this.
            # 'InceptionV3_Siamese': build_inceptionv3_siamese_network(input_shape=(299, 299, 3), embedding_dim=128),
            # 'EfficientNetB0_Siamese': build_efficientnetb0_siamese_network(input_shape=self.input_shape, embedding_dim=128),
            'FCSN_Model': build_fully_convolutional_siamese_network(input_shape=self.input_shape)
        
        }
                
        # Calculate class weights - Keep this for informational purposes, or remove if truly balanced
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        print(f"Class weights: {class_weight_dict}")
        
        # --- MODIFIED: Direct assignment of training data
        train_data = [x1_train, x2_train]
        train_labels = y_train
        # --- END MODIFIED SECTION ---
        
        # Dictionary to store results
        model_results = {}
        
        # Train and evaluate each model
        for model_name, model in models.items():
            print(f"\n{'='*50}")
            print(f"Training {model_name} Siamese Network")
            print(f"{'='*50}")
            
            # Set the current model for the current training loop iteration
            self.siamese_model = model
            
            # Get optimized callbacks
            #callbacks = get_trojan_detection_callbacks(model_name=f'trojan_{model_name.lower()}')
            
            # Train the model with enhanced configuration
            print(f"Starting training for {model_name} model...")
            
            history = self.siamese_model.fit(
                train_data, train_labels, # Now directly using the original train_data and train_labels
                validation_data=([x1_val, x2_val], y_val),
                batch_size=batch_size,
                epochs=epochs,
                #callbacks=callbacks,
                # REMOVED: class_weight=class_weight_dict, # Remove this if classes are truly balanced
                verbose=1,
                shuffle=True
            )
            
            # Comprehensive evaluation on test set
            print(f"Evaluating {model_name} model...")
            test_preds = self.siamese_model.predict([x1_test, x2_test])
            test_probs = test_preds.flatten()
            
            # Use optimized threshold (you might want to tune this)
            optimal_threshold = 0.5  # Standard for sigmoid output, can be tuned using ROC curve
            binary_preds = (test_probs > optimal_threshold).astype(int)
            
            # Calculate comprehensive metrics
            test_accuracy = accuracy_score(y_test, binary_preds)
            test_precision = precision_score(y_test, binary_preds, zero_division=0)
            test_recall = recall_score(y_test, binary_preds, zero_division=0)
            test_f1 = f1_score(y_test, binary_preds, zero_division=0)
            
            # Calculate AUC if both classes are present
            test_auc = 0
            if len(np.unique(y_test)) > 1:
                test_auc = roc_auc_score(y_test, test_probs)
            
            # Store results
            model_results[model_name] = {
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1,
                'test_auc': test_auc,
                'history': history,
                'predictions': test_probs,
                'threshold': optimal_threshold
            }
            
            print(f"\n{model_name} Model Results:")
            print(f"Test Accuracy:  {test_accuracy:.4f}")
            print(f"Test Precision: {test_precision:.4f}")
            print(f"Test Recall:    {test_recall:.4f}")
            print(f"Test F1-Score:  {test_f1:.4f}")
            print(f"Test AUC:       {test_auc:.4f}")
            
            # Print confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, binary_preds)
            print(f"Confusion Matrix:")
            print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
            print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        # Display final comparison
        print(f"\n{'='*80}")
        print("FINAL COMPARISON OF ALL MODELS")
        print(f"{'='*80}")
        print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1':<8} {'AUC':<8}")
        print(f"{'-'*85}") # Adjusted separator length
        
        for model_name, results in model_results.items():
            print(f"{model_name:<25} {results['test_accuracy']:<10.4f} "
                f"{results['test_precision']:<11.4f} {results['test_recall']:<8.4f} "
                f"{results['test_f1']:<8.4f} {results['test_auc']:<8.4f}")
        
        # Find best performing model based on F1-score (better for imbalanced data)
        best_model = max(model_results.items(), key=lambda x: x[1]['test_f1'])
        print(f"\nBest performing model: {best_model[0]} with F1-score: {best_model[1]['test_f1']:.4f}")
        
        # Alternative: Find best model based on AUC
        best_auc_model = max(model_results.items(), key=lambda x: x[1]['test_auc'])
        print(f"Best AUC model: {best_auc_model[0]} with AUC: {best_auc_model[1]['test_auc']:.4f}")
        
        # Set the best model as the final siamese_model (based on F1-score)
        self.siamese_model = models[best_model[0]]
        self.detection_threshold = best_model[1]['threshold']
        print(f"Final model set to: {best_model[0]} with threshold: {self.detection_threshold}")
        
        # Generate ROC curve for the best model
        try:
            from sklearn.metrics import roc_curve
            import matplotlib.pyplot as plt
            
            best_preds = best_model[1]['predictions']
            fpr, tpr, thresholds = roc_curve(y_test, best_preds)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'{best_model[0]} (AUC = {best_model[1]["test_auc"]:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve for Hardware Trojan Detection')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f'roc_curve_{best_model[0].lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Find optimal threshold using Youden's J statistic
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold_roc = thresholds[optimal_idx]
            print(f"Optimal threshold from ROC: {optimal_threshold_roc:.4f}")
            
        except ImportError:
            print("Matplotlib not available, skipping ROC curve generation")
        
        # Return comprehensive results
        return {
            'test_accuracy': best_model[1]['test_accuracy'],
            'test_f1': best_model[1]['test_f1'],
            'test_auc': best_model[1]['test_auc'],
            'best_model_name': best_model[0],
            'all_model_results': model_results,
            'class_distribution': {
                'train_positive_ratio': np.mean(y_train),
                'val_positive_ratio': np.mean(y_val),
                'test_positive_ratio': np.mean(y_test)
            }
        }

    def evaluate_model_detailed(self, test_data_path: str = None) -> Dict[str, Any]:
        """Detailed evaluation method for hardware trojan detection"""
        if self.siamese_model is None:
            raise RuntimeError("No Siamese model has been trained. Please call train_model() first.")

        if test_data_path:
            test_dataset = SiameseDataset(test_data_path)
            x1_test, x2_test, y_test = test_dataset.get_data()
        else:
            # Use existing test data if available
            test_dataset = SiameseDataset('data/pairs/test_pairs.csv')
            x1_test, x2_test, y_test = test_dataset.get_data()
        
        # Get predictions
        predictions = self.siamese_model.predict([x1_test, x2_test])
        pred_probs = predictions.flatten()
        binary_preds = (pred_probs > self.detection_threshold).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix, classification_report
        )
        
        metrics = {
            'accuracy': accuracy_score(y_test, binary_preds),
            'precision': precision_score(y_test, binary_preds, zero_division=0),
            'recall': recall_score(y_test, binary_preds, zero_division=0),
            'f1_score': f1_score(y_test, binary_preds, zero_division=0),
            'auc': roc_auc_score(y_test, pred_probs) if len(np.unique(y_test)) > 1 else 0,
            'confusion_matrix': confusion_matrix(y_test, binary_preds),
            'classification_report': classification_report(y_test, binary_preds)
        }
        
        print("Detailed Evaluation Results:")
        print("=" * 50)
        for metric, value in metrics.items():
            if metric not in ['confusion_matrix', 'classification_report']:
                print(f"{metric.capitalize()}: {value:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        
        print(f"\nClassification Report:")
        print(metrics['classification_report'])
        
        return metrics