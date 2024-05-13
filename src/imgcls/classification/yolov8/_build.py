from typing_extensions import Self

import numpy as np

from imgcls.classification.core import AbstractClassificationModel, ImageArrayList, ClassArrayList
import tensorflow as tf

from imgcls.util import printdf

# TODO not yet done
class YOLOClassificationModel(AbstractClassificationModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._epochs = 10
        self._batch_size = 32

    def init_model(self):
        return _init_yolo_from_scratch(self.data_source.n_labels)

    def fit(self, X: ImageArrayList, y: ClassArrayList) -> Self:
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(X, y, epochs=self._epochs, batch_size=self._batch_size)

        return self

    def predict(self, X: ImageArrayList) -> list[np.ndarray]:
        return self.model.predict(X)


def _init_yolo_from_scratch(n_classes: int):
    """train model from scratch"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(500, 500, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])

    return model



def main():
    yolo = YOLOClassificationModel.load_dataset()
    data = yolo.data_source
    printdf(data.train_data)

    # X_train = data.train_image_stack()
    # y_train = tf.convert_to_tensor(data.train_label_stack())
    #
    # X_test = data.test_image_stack()
    #
    # yolo.fit(X_train, y_train)
    # ret = yolo.predict(X_test)
    #
    # np.save('test.npy', ret)


if __name__ == '__main__':
    main()
