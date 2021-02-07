#!/usr/bin/env python3

import numpy as np


class Experiment(object):
    def __init__(
        self, keras_model, training_data: dict, unlabeled_data: np.array, config: dict
    ):
        self._model = keras_model
        self._training_data = training_data
        self._unlabeled_data = unlabeled_data
        self._config = config

    def prepare_data(self):
        # Data specifics
        pass

    def train(self):
        self._model.compile(**self._config["training_parameters"]["compile"])
        self._model.fit(
            self._training_data["data"],
            self._training_data["label"],
            batch_size=128,
            epochs=15,
            validation_split=0.1,
        )

    def predict_unlabled(self):
        pass

    def select_by_method(self, method: str):
        pass

    def print_or_write_results_to_yaml(self):
        pass
