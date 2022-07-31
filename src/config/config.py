import yaml

from pathlib import Path
from dataclasses import dataclass



@dataclass(frozen=True)
class Config:
    input_data: str
    data_type: str
    load_data: dict
    model_name: str
    pg_target: str
    rai_target: str
    threshold_rai_target: float
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
    config_gen = load_config_file(Path(__file__).parents[2].resolve()/'config'/ "config.yml")
    return Config(**config_gen)