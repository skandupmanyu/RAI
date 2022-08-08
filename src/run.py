
import time
import logging

from src.run_dataset import main as run_dataset
from src.run_train import main as run_train
from src.predict import main as run_predict

logger = logging.getLogger(__name__)

def main():
    run_dataset()
    run_train()
    run_predict()


if __name__ == '__main__':
    main()