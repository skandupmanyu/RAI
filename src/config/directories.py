from pathlib import Path
from src.exceptions import DirectoryNotFoundError

class _Directories:
    def __init__(self):
        self.project_root = Path(__file__).parents[2].resolve()
        self.package = self.project_root / 'src'
        self.config = self.package / 'config'
        self.data = self.project_root / 'data'
        self.raw_data_dir = self.data / 'raw'
        self.intermediate_data_dir = self.data / 'intermediate'
        self.project_config = self.project_root / 'config'
        self.artefacts_dir = self.data / 'artefacts'


        for dir_path in vars(self).values():
            try:
                dir_path.mkdir(exist_ok=True, parents=True)
            except Exception as e:
                raise DirectoryNotFoundError("Cannot create the directory") from e



directories = _Directories()