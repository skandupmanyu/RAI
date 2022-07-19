import yaml

from pathlib import Path
from dataclasses import dataclass

from src.config.directories import directories
from src.constants import *
from src.utils.utils import *


@dataclass(frozen=True)
class Config:
    pg_target: str
    rai_target: str
    pos_rate: float
    latest_year_dataset: float
    sample_size: float
    random_state_sample: float
    bias_ranker: dict
    features: dict
    model: dict
    test_cutoff: float
    split_random_size: float

def load_config_file(path):
    cfg = yaml.load(Path(path).read_text(), Loader=yaml.FullLoader)
    return cfg

def get_config() -> Config:
    config_gen = load_config_file(directories.project_config / "config.yml")

    return Config(**config_gen)