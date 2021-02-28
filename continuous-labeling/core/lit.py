#!/usr/bin/env python3

import argparse
import pathlib

from labeling_iteratively.labeling_iteratively import LabelingIteratively


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
        required=True,
        help="The initial model to be used throughout the process.",
    )
    parser.add_argument(
        "-t",
        "--train",
        required=True,
        help="Path to the data that should be used for training the model.",
    )
    parser.add_argument(
        "-u",
        "--unlabeled",
        required=True,
        help="Path to the so far unlabeled data.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=str(CONFIG_PATH),
        help="Path to the config file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(args)
    lit = LabelingIteratively(args.model, args.train, args.unlabeled, args.config)
    lit.run()
