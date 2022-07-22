import logging
import argparse

from src.run_dataset import main as run_dataset
from src.run_train import main as run_train
from src.run import main as run_all


logger = logging.getLogger(__name__)

_PIPELINES_REGISTRY = {
    'dataset': run_dataset,
    'train': run_train,
    'run': run_all
}


def main():
    args = parse_cli()

    pipeline = _PIPELINES_REGISTRY[args.action]
    pipeline()




def parse_cli():
    parser = argparse.ArgumentParser(
        description= "Responsible AI proxy model"
    )

    parser.add_argument("action", choices=_PIPELINES_REGISTRY)
    args = parser.parse_args()

    return args