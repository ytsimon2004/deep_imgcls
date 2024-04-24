from typing_extensions import Self

import numpy as np

from imgcls.classification.core import AbstractClassificationModel, ImageArrayList, ClassArrayList
import tensorflow as tf


class YOLOClassificationModel(AbstractClassificationModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._epochs = 10
        self._batch_size = 32

    def init_model(self):
        return _init_yolo_from_pretrain()

    def fit(self, X: ImageArrayList, y: ClassArrayList) -> Self:
        X = tf.image.resize(X, [448, 448])
        X = tf.stack(X)

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(X, y, epochs=self._epochs, batch_size=self._batch_size)

        return self

    def predict(self, X: ImageArrayList) -> list[np.ndarray]:
        return self.model.predict(X)


def _init_yolo_from_scratch():
    """train model from scratch"""
    pass


def _init_yolo_from_pretrain():
    """fine tune the model from other's pretrained model"""
    model = tf.keras.applications.ResNet50(include_top=False, input_shape=(448, 448, 3), pooling='avg')
    model = tf.keras.Sequential([
        model,
        # tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(20, activation='softmax')
    ])

    return model


def main():
    yolo = YOLOClassificationModel.load_dataset()
    yolo.init_model()
    data = yolo.data_source
    # X_train = data.train_data['img'].to_list()
    # y_train = data.image_labels_int
    X_train = data.train_image_stack()
    y_train = tf.convert_to_tensor(data.train_label_stack())


    X_test = data.test_image_stack()

    yolo.fit(X_train, y_train)
    ret = yolo.predict(X_test)

    np.save('test.npy', ret)


if __name__ == '__main__':
    main()
