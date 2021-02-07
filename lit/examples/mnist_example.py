#!/usr/bin/env python3

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def keras_example():
    NUMBER_OF_CLASSES = 10
    INPUT_SHAPE = (28, 28, 1)
    NORMALIZATION_FACTOR = 255
    BATCH_SIZE = 128
    EPOCHS = 15
    VALIDATION_SPLIT = 0.1

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / NORMALIZATION_FACTOR
    x_test = x_test.astype("float32") / NORMALIZATION_FACTOR
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Use one-hot encoding for the net.
    y_train = keras.utils.to_categorical(y_train, NUMBER_OF_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUMBER_OF_CLASSES)

    model = keras.Sequential(
        [
            keras.Input(shape=INPUT_SHAPE),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(NUMBER_OF_CLASSES, activation="softmax"),
        ]
    )
    model.summary()
    model.compile(
        **{
            "loss": "categorical_crossentropy",
            "optimizer": "adam",
            "metrics": "accuracy",
        }
    )
    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
    )

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])


if __name__ == "__main__":
    keras_example()
