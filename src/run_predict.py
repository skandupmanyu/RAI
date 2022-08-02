
import time
import logging

from src.predict.main import predict, save_predictions
from src import directories, config


logger = logging.getLogger(__name__)


def main():
    start = time.time()
    logger.info("Starting prediction job...")

    predictions = predict(config)
    logger.debug("Saving predictions on disk...")
    save_predictions(predictions)

    run_duration = time.time() - start
    logger.debug("Prediction job done.")
    logger.info(f"Prediction took {run_duration:.2f}s to complete.")


if __name__ == '__main__':
    main()