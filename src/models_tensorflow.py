"""MobileNetV2 modified for 13-channel Sentinel-2 input and 10-class EuroSAT output.

A learnable 1x1 conv projects 13 bands to 3 channels, feeding the standard
ImageNet-pretrained MobileNetV2 backbone. This keeps the bulk of the
pretrained weights usable without per-layer surgery.
"""
from __future__ import annotations

from tensorflow.keras import Model, layers
from tensorflow.keras.applications import MobileNetV2


def build_mobilenetv2_13ch(num_classes: int = 10, input_size: int = 64) -> Model:
    base = MobileNetV2(
        input_shape=(input_size, input_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = True

    inputs = layers.Input(shape=(input_size, input_size, 13))
    x = layers.Conv2D(3, kernel_size=1, padding="same", name="band_projection")(inputs)
    x = base(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs, name="mobilenetv2_13ch")
