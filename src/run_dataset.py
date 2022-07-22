#!/usr/bin/env python3
"""
Dataset generation script.
"""
import logging

from src.config.directories import directories
from src.constants import DATASET
from src.data import build_dataset
from src.in_out import save_dataset
from src import config

logger = logging.getLogger(__name__)


def main():
    """ Build a dataset and save it as intermediate data. """
    logger.info("Building dataset...")
    dataset = build_dataset(config)
    save_path = directories.intermediate_data_dir / DATASET
    save_dataset(dataset, path=save_path)


if __name__ == '__main__':
    main()