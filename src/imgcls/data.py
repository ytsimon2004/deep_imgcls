from typing import NamedTuple, Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from typing_extensions import Self

from src.imgcls.io import TRAIN_DIR, TEST_DIR


class ImageClassification(NamedTuple):
    data: pl.DataFrame
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

    dtype: Literal['test', 'train']

    @classmethod
    def load_train(cls) -> Self:
        df = pl.read_csv(TRAIN_DIR / 'train_set.csv')

        img_train = [
            np.load(TRAIN_DIR / 'img' / f'train_{i}.npy')
            for i, _ in enumerate(df.iter_rows())
        ]

        img_seg = [
            np.load(TRAIN_DIR / 'seg' / f'train_{i}.npy')
            for i, _ in enumerate(df.iter_rows())
        ]

        df = df.with_columns(
            img=pl.Series(values=img_train, dtype=pl.Object),
            seg=pl.Series(values=img_seg, dtype=pl.Object)
        )

        return ImageClassification(df, dtype='train')


    @classmethod
    def load_test(cls) -> Self:
        df = pl.read_csv(TRAIN_DIR / 'train_set.csv')

        img_train = [
            np.load(TEST_DIR / 'img' / f'test_{i}.npy')
            for i, _ in enumerate(df.iter_rows())
        ]

        df = df.with_columns(
            img=pl.Series(values=img_train, dtype=pl.Object),
            seg=pl.lit(-1)
        )

        return ImageClassification(df, dtype='train')

    def only_labels(self) -> Self:
        return self._replace(data=(self.data.drop(['Id', 'img', 'seg'])))

    @property
    def n_labels(self) -> int:
        return self.only_labels().data.shape[1]
    
    @property
    def labels(self) -> list[str]:
        return self.only_labels().data.columns

    @property
    def image_labels(self) -> list[list[str]]:
        ret = []
        labels = np.array(self.labels)
        for row in self.only_labels().data.iter_rows():
            label = labels[np.nonzero([it == 1 for it in row])[0]].tolist()
            ret.append(label)
        return ret



if __name__ == '__main__':
    clz = ImageClassification.load_test()
    print(clz.data)