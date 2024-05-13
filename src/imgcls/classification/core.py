import abc
from typing import Final, TypeAlias, Any

import numpy as np
from typing_extensions import Self

from imgcls.data import ImageClassification

__all__ = [
    'ImageArrayList',
    'ClassArrayList',
    'AbstractClassificationModel'
]


ImageArrayList: TypeAlias = list[np.ndarray]  # n x (height x width x 3)
ClassArrayList: TypeAlias = list[np.ndarray]  # n x (nb_classes)


class AbstractClassificationModel(metaclass=abc.ABCMeta):
    """ABC class for train/predict the image classification"""

    def __init__(self, data_source: ImageClassification):
        self.data_source: Final[ImageClassification] = data_source

        self.model = self.init_model()

    def __call__(self, X: ImageArrayList):
        self.predict(X)

    @classmethod
    def load_dataset(cls) -> Self:
        return cls(ImageClassification.load())

    @abc.abstractmethod
    def init_model(self) -> Any:
        pass

    @abc.abstractmethod
    def fit(self, X: ImageArrayList, y: ClassArrayList) -> Self:
        """
        Adjusts the class ratio variable to the one observed in y.

        :param X:
        :param y:
        :return:
        """
        pass

    @abc.abstractmethod
    def predict(self, X: ImageArrayList) -> list[np.ndarray]:
        """
        Predicts for each input a label

        :param X:
        :return:
        """
        pass
