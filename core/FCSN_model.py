# FCSN model 
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.applications import ResNet50
from typing import Tuple, List

def fcsn_decoder_block(
    input_tensor: tf.Tensor,
    skip_features: tf.Tensor,
    filters: int
) -> tf.Tensor:
    """
    Decoder block for the U-Net architecture.
    Upsamples the input tensor and concatenates it with skip connection features.

    Args:
        input_tensor (tf.Tensor): The input tensor from the previous decoder block.
        skip_features (tf.Tensor): The skip connection features from the encoder.
        filters (int): The number of filters for the convolutional layers.

    Returns:
        tf.Tensor: The output tensor of the decoder block.
    """
    # Upsample the input tensor
    x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(input_tensor)
    
    # Concatenate with the skip features from the encoder
    # Ensure dimensions are compatible; cropping/padding might be needed in some cases
    x = layers.Concatenate()([x, skip_features])
    
    # Apply convolutional layers
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    
    return x

def build_fcsn_unet(
    input_shape: Tuple[int, int, int] = (224, 224, 3)
) -> keras.Model:
    """
    Builds a Fully Convolutional Siamese Network (FCSN) with a U-Net style decoder.
    This model is designed for pixel-level change detection, making it suitable for
    detecting hardware trojans by comparing a golden reference image with an image
    under inspection.

    Args:
        input_shape (Tuple[int, int, int]): The shape of the input images (height, width, channels).

    Returns:
        keras.Model: A compiled Keras model that takes two images and outputs a change mask.
    """
    # --- 1. Define the Shared Encoder (ResNet50 Backbone) ---
    # We use a pre-trained ResNet50 and extract feature maps from different stages
    # to serve as skip connections for the decoder.
    base_encoder = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_encoder.trainable = False # Freeze the encoder weights

    # Define the layers from which we will extract features (skip connections)
    # These names correspond to the output of different residual blocks in ResNet50
    skip_connection_layers = [
        'conv1_relu',           # 112x112
        'conv2_block3_out',     # 56x56
        'conv3_block4_out',     # 28x28
        'conv4_block6_out',     # 14x14
    ]
    encoder_outputs = [base_encoder.get_layer(name).output for name in skip_connection_layers]
    
    # The final output of the encoder is the "bottleneck"
    bottleneck = base_encoder.output # 7x7

    # Create the shared encoder model
    encoder = keras.Model(inputs=base_encoder.input, outputs=encoder_outputs + [bottleneck], name="resnet_encoder")

    # --- 2. Define the Siamese Branches ---
    input_golden = layers.Input(shape=input_shape, name="input_golden_model")
    input_test = layers.Input(shape=input_shape, name="input_test_model")

    # Process each input through the shared encoder
    features_golden = encoder(input_golden)
    features_test = encoder(input_test)

    # --- 3. Feature Fusion Layer ---
    # Fuse the features from both branches using absolute difference.
    # This is done for each level of the feature hierarchy.
    fused_features = []
    for f_golden, f_test in zip(features_golden, features_test):
        diff = layers.Lambda(lambda x: tf.math.abs(x[0] - x[1]))([f_golden, f_test])
        fused_features.append(diff)

    # Separate the fused skip features from the fused bottleneck feature
    fused_skips = fused_features[:-1]
    fused_bottleneck = fused_features[-1]

    # --- 4. Decoder Path (U-Net Style) ---
    # The decoder upsamples the bottleneck features and combines them with
    # the fused skip connection features to reconstruct the high-resolution change mask.
    
    # Decoder Block 1
    d1 = fcsn_decoder_block(fused_bottleneck, fused_skips[3], filters=512)
    
    # Decoder Block 2
    d2 = fcsn_decoder_block(d1, fused_skips[2], filters=256)

    # Decoder Block 3
    d3 = fcsn_decoder_block(d2, fused_skips[1], filters=128)
    
    # Decoder Block 4
    d4 = fcsn_decoder_block(d3, fused_skips[0], filters=64)
    
    # Final upsampling to restore original image dimensions
    final_upsample = layers.Conv2DTranspose(32, (2, 2), strides=2, padding='same')(d4)

    # --- 5. Output Layer ---
    # A 1x1 convolution with a single filter and sigmoid activation
    # produces the final binary change mask.
    output_mask = layers.Conv2D(1, (1, 1), activation='sigmoid', name='change_mask_output')(final_upsample)

    # --- Create and Compile the Model ---
    fcsn_model = keras.Model(inputs=[input_golden, input_test], outputs=output_mask, name='FCSN_U-Net')

    fcsn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2, name='mean_iou')]
    )

    return fcsn_model