"""CNN architecture for the hotdog / not-hotdog classifier.

A small VGG-style convolutional network: stacked Conv2D + MaxPooling blocks that grow
wider as the spatial size shrinks, followed by a dense head with dropout and a single
sigmoid output for binary classification.
"""

from tensorflow.keras import layers, models

from preprocessing import IMG_SIZE

INPUT_SHAPE = (*IMG_SIZE, 3)  # 224x224 RGB

# Input:         224 x 224 x 3      (original image, 3 RGB channels)
# After block 1: 112 x 112 x 32     (halved by MaxPooling, 32 feature maps)
# After block 2:  56 x  56 x 64     (halved again, 64 feature maps)
# After block 3:  28 x  28 x 128    (halved again, 128 feature maps)

# relu(x) = max(0, x)
# → anything negative becomes zero, positive stays the same
# → used in hidden layers: only positive evidence (pattern found) passes through

# sigmoid(x) = 1 / (1 + e^(-x))
# → squashes any number into a range between 0 and 1
# → used in the final layer: outputs a probability
# → close to 1 = hotdog, close to 0 = not hotdog

def build_model():
    # Linear stack of layers, one after another
    # https://keras.io/api/models/sequential/
    model = models.Sequential(name="not_hotdog_cnn")

    # input tensor shape (224x224x3)
    # https://keras.io/api/layers/core_layers/input/
    model.add(layers.Input(shape=INPUT_SHAPE))

    # Conv2D: slides 32 learnable 3x3 filters to detect visual patterns; relu adds non-linearity
    # https://keras.io/api/layers/convolution_layers/convolution2d/
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))

    # MaxPooling2D: keeps the max of each 2x2 region, halving width/height
    # https://keras.io/api/layers/pooling_layers/max_pooling2d/
    model.add(layers.MaxPooling2D((2, 2)))

    # Filters double each block (32 -> 64 -> 128): deeper layers learn richer features
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # turns the 3D tensor (28x28x128) into a 1D vector of 100,352 values for the dense head
    # https://keras.io/api/layers/reshaping_layers/flatten/
    model.add(layers.Flatten())

    # 28 positions (width)
    # × 28 positions (height)
    # × 128 feature maps
    # = 100,352 numbers

    # Dense(128): fully connected layer, each of the 100,352 inputs connects to all 128 neurons
    # learns which combinations of patterns are most useful for the final decision
    # https://keras.io/api/layers/core_layers/dense/
    model.add(layers.Dense(128, activation="relu"))

    # randomly zeroes 50% of units at train time to reduce overfitting
    # https://keras.io/api/layers/regularization_layers/dropout/
    model.add(layers.Dropout(0.5))

    # Single sigmoid unit -> probability the image is a hotdog (class 1)

    model.add(layers.Dense(1, activation="sigmoid"))

    # https://keras.io/api/models/model_training_apis/#compile-method
    model.compile(
        optimizer="adam",                # adaptive gradient-descent optimizer
        loss="binary_crossentropy",      # standard loss for a two-class / single-sigmoid setup
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    # prints the layer table with output shapes and parameter counts.
    build_model().summary()
