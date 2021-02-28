#!/usr/bin/env python3

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


class Setup(object):
    NUMBER_OF_CLASSES = 10
    INPUT_SHAPE = (28, 28, 1)
    NORMALIZATION_FACTOR = 255
    BATCH_SIZE = 128
    EPOCHS = 15
    VALIDATION_SPLIT = 0.1

    def __init__(self):
        self.valid_config = {
            "training_parameters": {
                "compile": {
                    "loss": "categorical_crossentropy",
                    "optimizer": "adam",
                    "metrics": "accuracy",
                }
            }
        }

        self.raw_model = keras.Sequential(
            [
                keras.Input(shape=Setup.INPUT_SHAPE),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(Setup.NUMBER_OF_CLASSES, activation="softmax"),
            ]
        )

        self.mnist = dict(train=None, test=None)
        self._load_data()

    def _load_data(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        x_train = x_train.astype("float32") / Setup.NORMALIZATION_FACTOR
        x_test = x_test.astype("float32") / Setup.NORMALIZATION_FACTOR
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

        # Use one-hot encoding for the net.
        y_train = keras.utils.to_categorical(y_train, Setup.NUMBER_OF_CLASSES)
        y_test = keras.utils.to_categorical(y_test, Setup.NUMBER_OF_CLASSES)

        self.mnist["train"] = dict(data=x_train, label=y_train)
        self.mnist["test"] = dict(data=x_test, label=y_test)