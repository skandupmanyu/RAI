from pathlib import Path
from src.exceptions import DirectoryNotFoundError

class _Directories:
    def __init__(self, config):
        self.project_root = Path(__file__).parents[2].resolve()
        self.package = self.project_root / 'src'
        self.config = self.package / 'config'
        self.data = self.project_root / 'data'
        self.raw_data_dir = self.data / 'raw'
        self.intermediate_data_dir = self.data / 'intermediate'
        self.model_intermediary_data_dir = self.intermediate_data_dir / config.model_name
        self.project_config = self.project_root / 'config'
        self.artefacts_dir = self.data / 'artefacts'
        self.model = self.artefacts_dir / config.model_name


        for dir_path in vars(self).values():
            try:
                dir_path.mkdir(exist_ok=True, parents=True)
            except Exception as e:
                raise DirectoryNotFoundError("Cannot create the directory") from e


def get_directories(config) -> _Directories:
    return _Directories(config)
