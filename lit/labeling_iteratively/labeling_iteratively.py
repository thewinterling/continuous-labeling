#!/usr/bin/env python3

import logging
import yaml

from lit.labeling_iteratively.experiment import Experiment


class LabelingIteratively(object):
    """High level procedure:
    * initialize with
        * a classification model (currently supported: keras),
        * (labeled) training data as well as unlabeled data for prediction,
        * a config dict.
    * train the model using the training data
    * predict the unlabeled data
    * select the samples that should be labeled in the next iteration
    * write the id's of those to a yaml file or print it directly

    When using this class, all of the above is handeled by the public 'run' method.
    """

    def __init__(self, model, training_data: str, unlabeled_data: str, config: str):
        """Class constructor to set the intial state.
        Args:
            model: The machine learning model to be used.
            training_data: Already labeled data that can be used for training.
            unlabeled_data: Data used in the prediction.
            config: Additional config info for the sample selection step.
        """
        config_dict = self._load_and_check_for_valid_config(config)
        self._experiment = Experiment(model, training_data, unlabeled_data, config_dict)

    def _load_and_check_for_valid_config(self, config):
        """Load yaml and check if all required fields in the config are available.
        If anything is not as expected fail early.
        """
        # TODO: check that the dict contains valid values.
        try:
            with open(config) as yfile:
                data = yaml.load(yfile)
                return data
        except:
            logging.fatal("Couldn't read the config yaml, exit now.")
            exit(0)

    def run(self):
        """Entrypoint for predicting the unlabeled data and selecting
        the n samples to be manually labeled in the next iteration.
        Args:
            None
        Returns:
            None. Writes the samples in a yaml file according to
            <experiment-name>_iteration_<iteration-count>.yaml
            or prints it to the console.
        """
        self._experiment.prepare_data()
        self._experiment.train()
        self._experiment.predict_unlabled()
        self._experiment.select_by_method()
        self._experiment.print_or_write_results_to_yaml()
