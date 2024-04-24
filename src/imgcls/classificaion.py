from typing import Final

import numpy as np
from typing_extensions import TypeAlias, Self

import polars as pl

from imgcls.data import ImageClassification

ImageArrayList: TypeAlias = list[np.ndarray]  # n x (height x width x 3)
ClassArrayList: TypeAlias = list[np.ndarray]  # n x (nb_classes)


class RandomClassificationModel:
    """Random classification model:

    - generates random labels for the inputs based on the class distribution observed during training
    - assumes an input can have multiple labels
    """

    def __init__(self, labels: list[str]):
        self.distribution: np.ndarray | None = None
        self.labels: Final[list[str]] = labels

    def __call__(self, X: ImageArrayList):
        return self.predict(X)

    def fit(self, X: ImageArrayList, y: ClassArrayList | pl.DataFrame) -> Self:
        """
        Adjusts the class ratio variable to the one observed in y.

        :param X:
        :param y:  TODO inconsistent type in the kaggle notebook
        :return:
        """
        if isinstance(y, pl.DataFrame):
            self.distribution = y.mean().row(0)
            print(self.distribution)
        else:
            self.distribution = np.mean(y, axis=0)

        print("Setting class distribution to:\n{}".format(
            "\n".join(f"{label}: {p}" for label, p in zip(self.labels, self.distribution))))

        return self

    def predict(self, X: ImageArrayList) -> list[np.ndarray]:
        """
        Predicts for each input a label

        :param X:
        :return:
        """
        np.random.seed(0)
        return [np.array([int(np.random.rand() < p) for p in self.distribution]) for _ in X]


def main():
    cls = ImageClassification.load()
    model = RandomClassificationModel(cls.labels_list)
    model.fit(cls.train_data['img'].to_list(), cls.get_train_label())

if __name__ == '__main__':
    main()