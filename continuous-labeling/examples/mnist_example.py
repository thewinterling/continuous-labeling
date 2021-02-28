#!/usr/bin/env python3

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


class Parameters(object):
    NUMBER_OF_CLASSES = 10
    INPUT_SHAPE = (28, 28, 1)
    NORMALIZATION_FACTOR = 255
    BATCH_SIZE = 128
    EPOCHS = 15
    VALIDATION_SPLIT = 0.1


def get_all_mnist_data():
    """Returns train and test data including labels in one-hot encoding style
    directly useable by the keras model.

    Args:
        None
    Returns:
        Train and test data, including labels.
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / Parameters.NORMALIZATION_FACTOR
    x_test = x_test.astype("float32") / Parameters.NORMALIZATION_FACTOR
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Use one-hot encoding for the net.
    y_train = keras.utils.to_categorical(y_train, Parameters.NUMBER_OF_CLASSES)
    y_test = keras.utils.to_categorical(y_test, Parameters.NUMBER_OF_CLASSES)

    return (x_train, y_train), (x_test, y_test)


def get_simple_keras_model():
    """Returns the raw keras model, meaning the architecture only.

    Args:
        None
    Returns:
        The keras model to be used for training.
    """
    return keras.Sequential(
        [
            keras.Input(shape=Parameters.INPUT_SHAPE),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(Parameters.NUMBER_OF_CLASSES, activation="softmax"),
        ]
    )


def get_sample_model_config():
    """Returns a sample model config to be used in the `compile` method.

    Args:
        None
    Returns:
        Config, dict
    """
    return {
        "loss": "categorical_crossentropy",
        "optimizer": "adam",
        "metrics": "accuracy",
    }


def keras_example():
    """Main function that will be used in example if invoked as
    `python3 -m lit --example mnist`

    Args:
        None
    Returns:
        None
    """
    (x_train, y_train), (x_test, y_test) = get_all_mnist_data()

    model = get_simple_keras_model()
    model.summary()
    model.compile(**get_sample_model_config())
    model.fit(
        x_train,
        y_train,
        batch_size=Parameters.BATCH_SIZE,
        epochs=Parameters.EPOCHS,
        validation_split=Parameters.VALIDATION_SPLIT,
    )

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])


if __name__ == "__main__":
    keras_example()
