#!/usr/bin/env python3

import argparse
import pathlib

from lit.labeling_iteratively.labeling_iteratively import LabelingIteratively
from lit.examples import mnist_example


def _parse_args():
    """Define and parse the command line arguments.
    Args:
        None
    Return:
        argparse.Namespace Command line arguments specified by the user.
    """

    BASE_PATH = pathlib.Path(__file__).parent.absolute().parents[0]
    CONFIG_PATH = pathlib.Path.joinpath(BASE_PATH, "config", "default.yaml")

    parser = argparse.ArgumentParser(description="LIT - labeling iteratively")
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="The initial model to be used throughout the process. Defaulting to 'None'.",
    )
    parser.add_argument(
        "-t",
        "--train",
        default=None,
        help="Path to the data that should be used for training the model. Defaulting to 'None'.",
    )
    parser.add_argument(
        "-u",
        "--unlabeled",
        default=None,
        help="Path to the so far unlabeled data. Defaulting to 'None'.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=str(CONFIG_PATH),
        help="Path to the config file. Defaulting to {c}".format(c=str(CONFIG_PATH)),
    )
    parser.add_argument(
        "-e",
        "--example",
        default=None,
        help="If this flas is set, instead run the examples. Defaulting to 'None'.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.example is not None:
        mnist_example.keras_example()
        exit(0)

    lit = LabelingIteratively(args.model, args.train, args.unlabeled, args.config)
    lit.run()
