import shutil
from typing import NamedTuple

import numpy as np
import torch
from ultralytics.engine.results import Results

__all__ = ['YoloBoxes',
           'extract_yolo_predict_box']

from imgcls.io import ImageClsDir, CACHE_DIRECTORY


class YoloBoxes(NamedTuple):
    cls: torch.Tensor
    conf: torch.Tensor
    data: torch.Tensor
    id: str
    is_track: bool
    orig_shape: tuple[int, int]
    shape: torch.Size
    xywh: torch.Tensor
    xywhn: torch.Tensor
    xyxy: torch.Tensor
    xyxyn: torch.Tensor


def extract_yolo_predict_box(predict_result: list[Results]) -> list[YoloBoxes]:
    """get predicted result boxes object"""
    if isinstance(predict_result, list):
        ret = []
        for result in predict_result:
            if hasattr(result, 'boxes'):
                ret.append(result.boxes)
            else:
                raise AttributeError('')

        return ret

    else:
        raise TypeError(f'{type(predict_result)}')


def create_attack_folder_structure(img_dir: ImageClsDir):
    """

    :param img_dir:
    :return:
    """
    img_png = img_dir.train_image_png
    dat = img_dir.train_dataframe.drop('Id').to_numpy()

    for img_id, cls in enumerate(dat):
        cls_ids = np.nonzero(cls)[0]
        filename = f'train_{img_id}.png'
        for cls_id in cls_ids:
            src = img_png / filename
            dst = img_dir.attack_train_dir / f'class{cls_id}' / filename
            dst.parent.mkdir(exist_ok=True)
            shutil.copy2(src, dst)


if __name__ == '__main__':
    d = ImageClsDir(CACHE_DIRECTORY)
    create_attack_folder_structure(d)
