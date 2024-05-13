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
    """Class for folder structure for train/test data

    DataFolder Structure ::

        <root_dir>
        ├── dataset.yml (1)
        ├── run (2)
        │   ├── predict
        │   ├── test_set.csv
        │   ├── train
        │   └── *yolov8s.pt
        │
        ├── test (3)
        │   ├── img
        │   ├── img_png
        │   └── test_set.csv
        │
        └── train (4)
            ├── img
            ├── img_png
            ├── seg
            ├── seg_png
            └── train_set.csv

        ---------------------------

        (1) config yaml file for the custom path info/image labels
        (2) directory for model/train/evaluation output files
        (3) directory for test dataset
        (4) directory for train dataset

    """
    root_dir: Path

    @staticmethod
    def ensure_dir(p: Path):
        """auto mkdir, use for custom dir that fit for yolo pipeline"""
        p.mkdir(exist_ok=True)
        return p

    @property
    def train_dir(self) -> Path:
        return self.root_dir / 'train'

    @property
    def train_img(self) -> Path:
        return self.train_dir / 'img'

    @property
    def train_img_png(self) -> Path:
        return self.ensure_dir(self.train_dir / 'img_png')

    @property
    def train_seg(self) -> Path:
        return self.train_dir / 'seg'

    @property
    def train_seg_png(self) -> Path:
        return self.ensure_dir(self.train_dir / 'seg_png')

    @property
    def train_dataframe(self) -> pl.DataFrame:
        return pl.read_csv(self.train_dir / 'train_set.csv')

    # ============ #
    # Test Dataset #
    # ============ #

    @property
    def test_dir(self) -> Path:
        return self.root_dir / 'test'

    @property
    def test_img(self) -> Path:
        return self.test_dir / 'img'

    @property
    def test_img_png(self) -> Path:
        return self.ensure_dir(self.test_dir / 'img_png')

    # ================ #
    # Model Train/Eval #
    # ================ #

    @property
    def run_dir(self) -> Path:
        return self.root_dir / 'run'

    @property
    def predict_dir(self) -> Path:
        return self.run_dir / 'predict'

    @property
    def predict_label_dir(self) -> Path:
        return self.predict_dir / 'labels'
