"""CNN architecture for the hotdog / not-hotdog classifier.

Transfer learning: a MobileNetV2 base pretrained on ImageNet acts as a frozen feature
extractor, and we train only a small classification head on top. A from-scratch CNN
couldn't learn from our ~500 images (it collapsed to 50% accuracy); reusing features
already learned from millions of images is what makes a small dataset like this work.
"""

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

from preprocessing import IMG_SIZE

INPUT_SHAPE = (*IMG_SIZE, 3)  # 224x224 RGB


def build_model():
    # Pretrained base without its classifier head. Downloads ImageNet weights on first run.
    base = MobileNetV2(input_shape=INPUT_SHAPE, include_top=False, weights="imagenet")
    base.trainable = False  # freeze: keep the learned features, train only our head

    model = models.Sequential(
        [
            base,
            # Average each feature map into a single value -> a compact 1280-length vector.
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            # Single sigmoid unit -> probability the image is a hotdog (class 1).
            layers.Dense(1, activation="sigmoid"),
        ],
        name="not_hotdog_mobilenetv2",
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    build_model().summary()
