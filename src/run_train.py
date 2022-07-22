import time
import logging

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src import config
from src.config.directories import directories
from src.constants.filenames import DATASET
from src.in_out import load_dataset, save_training_output, save_rai_training_output
from src.train import train_proxy, train_rai

logger = logging.getLogger(__name__)

def main():
    """ Load a model and a dataset, then run a training job. """
    #train pg
    start = time.time()
    logger.info("Starting training job for proxy variable...")
    model = LGBMClassifier(**config.model['params'])
    dataset_path = directories.intermediate_data_dir / DATASET
    dataset = load_dataset(dataset_path)
    training_output = train_proxy(model, dataset, config)
    run_duration = time.time() - start
    logger.info("Training job for proxy variable done...")
    logger.info(f"Took {run_duration:.2f}s to execute")
    logger.info("Storing model and evaluation metrics")
    save_training_output(training_output, directory=directories.artefacts_dir / config.model['name'])

    #train rai
    start = time.time()
    logger.info("Starting training job for toy variable...")
    rai_training_output = train_rai(training_output['model'],model, dataset, config)
    run_duration = time.time() - start
    logger.info("Training job for proxy variable done...")
    logger.info(f"Took {run_duration:.2f}s to execute")
    logger.info("Storing model and evaluation metrics")
    save_rai_training_output(rai_training_output, directory=directories.artefacts_dir / config.model['name'])

if __name__ == '__main__':
    main()