#!/usr/bin/env python3

import unittest

from labeling_iteratively.experiment import Experiment
from tests.test_setup import Setup


class ExperimentTestFixture(unittest.TestCase):
    def setUp(self):
        self.setup = Setup()

        self.unittest_model = "blub"
        self.unittest_train_data = "tbd"
        self.unittest_unlabeled_data = "tbd"
        self.unittest_valid_config = {
            "training_parameters": {
                "compile": {
                    "loss": "categorical_crossentropy",
                    "optimizer": "adam",
                    "metrics": "accuracy",
                }
            }
        }

    def tearDown(self):
        pass

    def test_experiment_train_shall_fail_when_invalid_optimizer_is_given(self):
        # model.fit() will raise an Exception for several cases (given from the keras docs):
        # ValueError: In case of invalid arguments for
        # `optimizer`, `loss` or `metrics`.
        # Note that the invalid values are already set in `.compile()` but will only have
        # an effect in the `.fit()` method!
        invalid_config = self.setup.valid_config
        invalid_optimizer = "this_is_not_a_valid_optimizer"
        invalid_config["training_parameters"]["compile"]["loss"] = invalid_optimizer

        # We treat the test data as the unlabeled data for this test, as we don't
        # evaluate on the performance anyway.
        unlabeled = self.setup.mnist["test"]["data"]

        experiment = Experiment(
            self.setup.raw_model,
            self.setup.mnist["train"],
            unlabeled,
            invalid_config,
        )
        experiment.prepare_data()

        self.assertRaises(ValueError, experiment.train)


if __name__ == "__main__":
    unittest.main()
