#!/usr/bin/env python3

from lit.labeling_iteratively.experiment import Experiment
from lit.examples import mnist_example


class CustomExperiment(Experiment):
    def prepare_data(self):
        (x_train, y_train), (x_test, y_test) = mnist_example.get_all_mnist_data()