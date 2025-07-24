# core/siamese.py

import tensorflow as tf
from tensorflow import keras # Ensure keras is from tensorflow
from keras import layers # Ensure layers is from tensorflow.keras
from keras import backend as K
# Import all required pre-trained models
from keras.applications import ResNet50, VGG16, InceptionV3, EfficientNetB0
from typing import Tuple, List, Dict, Any
import numpy as np


# --- Utility functions ---

# Callbacks for Trojan Detection
def get_trojan_detection_callbacks(model_name: str = "siamese") -> List[keras.callbacks.Callback]:
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001),
        keras.callbacks.ModelCheckpoint(
            filepath=f'models/{model_name}_best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    return callbacks

# --- Separate Builders for Each CNN-based Siamese Model ---

def _create_siamese_head(embedding_network: keras.Model, input_shape: Tuple[int, int, int], embedding_dim: int, name_suffix: str = "") -> keras.Model:
    """Helper function to create the Siamese twin head (distance and prediction)."""
    # Define inputs consistent with your x1 (GM) and x2 (PUI) data order
    input_gm_model = layers.Input(shape=input_shape, name=f"input_gm_{name_suffix}") # First input: Golden Model
    input_pui_model = layers.Input(shape=input_shape, name=f"input_pui_{name_suffix}") # Second input: PCB Under Inspection

    processed_gm = embedding_network(input_gm_model)
    processed_pui = embedding_network(input_pui_model)

    # This line is already correct with output_shape=(embedding_dim,)
    distance = layers.Lambda(lambda x: tf.math.abs(x[0] - x[1]), output_shape=(embedding_dim,), name=f'l1_distance_{name_suffix}')([processed_gm, processed_pui])
    prediction = layers.Dense(1, activation='sigmoid', name=f'similarity_output_{name_suffix}')(distance)

    siamese_model = keras.Model(inputs=[input_gm_model, input_pui_model], outputs=prediction, name=f'Siamese_{name_suffix}')
    siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return siamese_model

def build_resnet50_siamese_network(input_shape: Tuple[int, int, int] = (224, 224, 3), embedding_dim: int = 128, trainable_base_layers: int = 0) -> keras.Model:
    """Builds a Siamese network with ResNet50 as the backbone."""
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    if trainable_base_layers > 0:
        for layer in base_model.layers[-trainable_base_layers:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    embedding_output = layers.Dense(embedding_dim, activation='relu', name='resnet_embedding_output')(x)
    embedding_network = keras.Model(inputs=base_model.input, outputs=embedding_output, name='ResNet50_Embedding')

    # FIX: Pass embedding_dim here
    model = _create_siamese_head(embedding_network, input_shape, embedding_dim, name_suffix='ResNet50')
    print("\n--- Built ResNet50 Siamese Network ---")
    embedding_network.summary()
    model.summary()
    return model

def build_vgg16_siamese_network(input_shape: Tuple[int, int, int] = (224, 224, 3), embedding_dim: int = 128, trainable_base_layers: int = 0) -> keras.Model:
    """Builds a Siamese network with VGG16 as the backbone."""
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    if trainable_base_layers > 0:
        for layer in base_model.layers[-trainable_base_layers:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True # VGG doesn't have BN, but good practice

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    embedding_output = layers.Dense(embedding_dim, activation='relu', name='vgg16_embedding_output')(x)
    embedding_network = keras.Model(inputs=base_model.input, outputs=embedding_output, name='VGG16_Embedding')

    # FIX: Pass embedding_dim here
    model = _create_siamese_head(embedding_network, input_shape, embedding_dim, name_suffix='VGG16')
    print("\n--- Built VGG16 Siamese Network ---")
    embedding_network.summary()
    model.summary()
    return model

def build_inceptionv3_siamese_network(input_shape: Tuple[int, int, int] = (299, 299, 3), embedding_dim: int = 128, trainable_base_layers: int = 0) -> keras.Model:
    """Builds a Siamese network with InceptionV3 as the backbone."""
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    if trainable_base_layers > 0:
        for layer in base_model.layers[-trainable_base_layers:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    embedding_output = layers.Dense(embedding_dim, activation='relu', name='inception_embedding_output')(x)
    embedding_network = keras.Model(inputs=base_model.input, outputs=embedding_output, name='InceptionV3_Embedding')

    # FIX: Pass embedding_dim here
    model = _create_siamese_head(embedding_network, input_shape, embedding_dim, name_suffix='InceptionV3')
    print("\n--- Built InceptionV3 Siamese Network ---")
    embedding_network.summary()
    model.summary()
    return model

def build_efficientnetb0_siamese_network(input_shape: Tuple[int, int, int] = (224, 224, 3), embedding_dim: int = 128, trainable_base_layers: int = 0) -> keras.Model:
    """Builds a Siamese network with EfficientNetB0 as the backbone."""
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    if trainable_base_layers > 0:
        for layer in base_model.layers[-trainable_base_layers:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    embedding_output = layers.Dense(embedding_dim, activation='relu', name='efficientnet_embedding_output')(x)
    embedding_network = keras.Model(inputs=base_model.input, outputs=embedding_output, name='EfficientNetB0_Embedding')

    # FIX: Pass embedding_dim here
    model = _create_siamese_head(embedding_network, input_shape, embedding_dim, name_suffix='EfficientNetB0')
    print("\n--- Built EfficientNetB0 Siamese Network ---")
    embedding_network.summary()
    model.summary()
    return model