from typing import NamedTuple
import tensorflow as tf
import numpy as np
import polars as pl
from typing_extensions import Self

from src.imgcls.io import TRAIN_DIR, TEST_DIR


class ImageClassification(NamedTuple):
    train_data: pl.DataFrame
    """
    ::
    
        ┌─────┬───────────┬─────────┬──────┬───┬───────┬───────────┬─────────────────┬─────────────────────┐
        │ Id  ┆ aeroplane ┆ bicycle ┆ bird ┆ … ┆ train ┆ tvmonitor ┆ img             ┆ seg                 │
        │ --- ┆ ---       ┆ ---     ┆ ---  ┆   ┆ ---   ┆ ---       ┆ ---             ┆ ---                 │
        │ i64 ┆ i64       ┆ i64     ┆ i64  ┆   ┆ i64   ┆ i64       ┆ object          ┆ object              │
        ╞═════╪═══════════╪═════════╪══════╪═══╪═══════╪═══════════╪═════════════════╪═════════════════════╡
        │ 0   ┆ 0         ┆ 0       ┆ 0    ┆ … ┆ 0     ┆ 0         ┆ [[[ 10   8  13] ┆ [[0 0 0 ... 0 0 0]  │
        │     ┆           ┆         ┆      ┆   ┆       ┆           ┆   [ 16  14  19] ┆  [0 0 0 ... 0…      │
        │     ┆           ┆         ┆      ┆   ┆       ┆           ┆ …               ┆                     │
        │ 1   ┆ 0         ┆ 0       ┆ 0    ┆ … ┆ 0     ┆ 0         ┆ [[[ 60  54  64] ┆ [[0 0 0 ... 0 0 0]  │
        │     ┆           ┆         ┆      ┆   ┆       ┆           ┆   [ 70  57  64] ┆  [0 0 0 ... 0…      │
        │     ┆           ┆         ┆      ┆   ┆       ┆           ┆ …               ┆                     │
        │ …   ┆ …         ┆ …       ┆ …    ┆ … ┆ …     ┆ …         ┆ …               ┆ …                   │
        │ 747 ┆ 0         ┆ 0       ┆ 0    ┆ … ┆ 0     ┆ 1         ┆ [[[ 52  70  84] ┆ [[ 0  0  0 ... 20   │
        │     ┆           ┆         ┆      ┆   ┆       ┆           ┆   [ 49  67  81] ┆ 20 20]              │
        │     ┆           ┆         ┆      ┆   ┆       ┆           ┆ …               ┆  [ 0  0…            │
        │ 748 ┆ 0         ┆ 0       ┆ 1    ┆ … ┆ 0     ┆ 0         ┆ [[[ 51  53  52] ┆ [[0 0 0 ... 0 0 0]  │
        │     ┆           ┆         ┆      ┆   ┆       ┆           ┆   [ 68  70  67] ┆  [0 0 0 ... 0…      │
        │     ┆           ┆         ┆      ┆   ┆       ┆           ┆ …               ┆                     │
        └─────┴───────────┴─────────┴──────┴───┴───────┴───────────┴─────────────────┴─────────────────────┘
    """

    test_data: pl.DataFrame
    """
    ::
    
        ┌─────┬───────────┬─────────┬──────┬───┬───────┬───────────┬─────────────────┬─────┐
        │ Id  ┆ aeroplane ┆ bicycle ┆ bird ┆ … ┆ train ┆ tvmonitor ┆ img             ┆ seg │
        │ --- ┆ ---       ┆ ---     ┆ ---  ┆   ┆ ---   ┆ ---       ┆ ---             ┆ --- │
        │ i64 ┆ i64       ┆ i64     ┆ i64  ┆   ┆ i64   ┆ i64       ┆ object          ┆ i32 │
        ╞═════╪═══════════╪═════════╪══════╪═══╪═══════╪═══════════╪═════════════════╪═════╡
        │ 0   ┆ -1        ┆ -1      ┆ -1   ┆ … ┆ -1    ┆ -1        ┆ [[[139 130 115] ┆ -1  │
        │     ┆           ┆         ┆      ┆   ┆       ┆           ┆   [136 127 112] ┆     │
        │     ┆           ┆         ┆      ┆   ┆       ┆           ┆ …               ┆     │
        │ 1   ┆ -1        ┆ -1      ┆ -1   ┆ … ┆ -1    ┆ -1        ┆ [[[146  95  92] ┆ -1  │
        │     ┆           ┆         ┆      ┆   ┆       ┆           ┆   [131  87  84] ┆     │
        │     ┆           ┆         ┆      ┆   ┆       ┆           ┆ …               ┆     │
        │ …   ┆ …         ┆ …       ┆ …    ┆ … ┆ …     ┆ …         ┆ …               ┆ …   │
        │ 748 ┆ -1        ┆ -1      ┆ -1   ┆ … ┆ -1    ┆ -1        ┆ [[[ 30  21  14] ┆ -1  │
        │     ┆           ┆         ┆      ┆   ┆       ┆           ┆   [ 32  23  16] ┆     │
        │     ┆           ┆         ┆      ┆   ┆       ┆           ┆ …               ┆     │
        │ 749 ┆ -1        ┆ -1      ┆ -1   ┆ … ┆ -1    ┆ -1        ┆ [[[184 181 190] ┆ -1  │
        │     ┆           ┆         ┆      ┆   ┆       ┆           ┆   [182 179 188] ┆     │
        │     ┆           ┆         ┆      ┆   ┆       ┆           ┆ …               ┆     │
        └─────┴───────────┴─────────┴──────┴───┴───────┴───────────┴─────────────────┴─────┘
        
    
    """

    @classmethod
    def load(cls) -> Self:
        train = pl.read_csv(TRAIN_DIR / 'train_set.csv')

        img_train = [
            np.load(TRAIN_DIR / 'img' / f'train_{i}.npy')
            for i, _ in enumerate(train.iter_rows())
        ]

        img_seg = [
            np.load(TRAIN_DIR / 'seg' / f'train_{i}.npy')
            for i, _ in enumerate(train.iter_rows())
        ]

        train = train.with_columns(
            img=pl.Series(values=img_train, dtype=pl.Object),
            seg=pl.Series(values=img_seg, dtype=pl.Object)
        )

        #
        test = pl.read_csv(TEST_DIR / 'test_set.csv')

        img_train = [
            np.load(TEST_DIR / 'img' / f'test_{i}.npy')
            for i, _ in enumerate(test.iter_rows())
        ]

        test = test.with_columns(
            img=pl.Series(values=img_train, dtype=pl.Object),
            seg=pl.lit(-1)
        )

        return ImageClassification(train, test)

    def get_train_label(self) -> pl.DataFrame:
        """train dataset with only label"""
        return self.train_data.drop(['Id', 'img', 'seg'])

    @property
    def n_labels(self) -> int:
        return self.get_train_label().shape[1]

    @property
    def labels_list(self) -> list[str]:
        return self.get_train_label().columns

    @property
    def image_labels_literal(self) -> list[np.ndarray]:
        ret = []
        labels = np.array(self.labels_list)
        for row in self.get_train_label().iter_rows():
            label = labels[np.nonzero([it == 1 for it in row])[0]].tolist()
            ret.append(label)
        return ret

    @property
    def image_labels_int(self) -> list[np.ndarray]:
        ret = []
        for row in self.get_train_label().iter_rows():
            label = np.nonzero([it == 1 for it in row])[0].tolist()
            ret.append(label)
        return ret

    # @property
    # def train_data_array(self):
    #     return self.train_data['img'].to_numpy()

    def as_mean(self) -> pl.DataFrame:
        return self.get_train_label().mean()

    def train_image_stack(self) -> tf.Tensor:
        train_img = self.train_data['img']

        ret = []
        for img in train_img:
            img = tf.image.resize(img, [448, 448])
            ret.append(img / 255.)

        return tf.stack(ret)

    def train_label_stack(self):
        return self.get_train_label().to_numpy()

    def test_image_stack(self) -> tf.Tensor:
        train_img = self.test_data['img']

        ret = []
        for img in train_img:
            img = tf.image.resize(img, [448, 448])
            ret.append(img / 255.)

        return tf.stack(ret)
