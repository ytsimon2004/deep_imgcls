from pathlib import Path
from typing import NamedTuple
import polars as pl

__all__ = [
    'CACHE_DIRECTORY',
    'TRAIN_DIR',
    'TEST_DIR',
    #
    'ImageClsDir'
]

CACHE_DIRECTORY = Path.home() / '.cache' / 'comvis' / 'imgcls'
TRAIN_DIR = CACHE_DIRECTORY / 'train'
TEST_DIR = CACHE_DIRECTORY / 'test'


class ImageClsDir(NamedTuple):
    """Class for folder structure for train/test dataset and model/prediction output

    DataFolder Structure ::

        <root_dir>
        ├── dataset.yml (1)
        ├── runs/ (2)
        │   └── detect
        │         ├── predict*
        │         │    ├── labels/
        │         │    ├── test_*.png
        │         │    └── test_set.csv (5)
        │         ├── train*
        │         │    ├── <yolo outputs>
        │         │    └── weights/ (6)
        │         └── *yolov8s.pt
        │
        ├── test/ (3)
        │   ├── img/
        │   ├── img_png/
        │   └── test_set.csv
        │
        └── train (4)
            ├── img/
            ├── img_png/
            ├── seg/
            ├── seg_png/
            └── train_set.csv

        ---------------------------

        (1) config yaml file for the custom path info/image labels
        (2) directory for model/train/evaluation output files
        (3) directory for test dataset
        (4) directory for train dataset
        (5) output results of classification
        (6) store the fine-tuned model weights

    """
    root_dir: Path

    @staticmethod
    def ensure_dir(p: Path):
        """auto mkdir, use for custom dir that fit for yolo pipeline"""
        p.mkdir(exist_ok=True)
        return p

    # ============= #
    # Train Dataset #
    # ============= #

    @property
    def train_data_dir(self) -> Path:
        return self.root_dir / 'train'

    @property
    def train_image_source(self) -> Path:
        return self.train_data_dir / 'img'

    @property
    def train_image_png(self) -> Path:
        return self.ensure_dir(self.train_data_dir / 'img_png')

    @property
    def train_seg_source(self) -> Path:
        return self.train_data_dir / 'seg'

    @property
    def train_seg_png(self) -> Path:
        return self.ensure_dir(self.train_data_dir / 'seg_png')

    @property
    def train_dataframe(self) -> pl.DataFrame:
        return pl.read_csv(self.train_data_dir / 'train_set.csv')

    # ============ #
    # Test Dataset #
    # ============ #

    @property
    def test_data_dir(self) -> Path:
        return self.root_dir / 'test'

    @property
    def test_image_source(self) -> Path:
        return self.test_data_dir / 'img'

    @property
    def test_image_png(self) -> Path:
        return self.ensure_dir(self.test_data_dir / 'img_png')

    # ================ #
    # Model Train/Eval #
    # ================ #

    @property
    def run_dir(self) -> Path:
        return self.root_dir / 'runs' / 'detect'

    def get_predict_dir(self, name: str) -> Path:
        return self.run_dir / name

    def get_predict_label_dir(self, name: str) -> Path:
        return self.get_predict_dir(name) / 'labels'

    def get_train_dir(self, name: str = 'train') -> Path:
        return self.run_dir / name

    def get_model_weights(self, name: str = 'train') -> Path:
        return self.run_dir / name / 'weights'

    # =================================== #
    # Adversarial Attack (For DataLoader) #
    # =================================== #

    @property
    def attack_dir(self) -> Path:
        return self.ensure_dir(self.root_dir / 'attack_dataset')

    @property
    def attack_train_dir(self) -> Path:
        return self.ensure_dir(self.attack_dir / 'train')

    @property
    def attack_test_dir(self) -> Path:
        return self.ensure_dir(self.attack_dir / 'test')



