from typing import NamedTuple

import torch
from ultralytics.engine.results import Results

__all__ = ['YoloBoxes',
           'extract_yolo_predict_box']

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


def extract_yolo_predict_box(predict_result: list[Results]) -> YoloBoxes:
    """get predicted result boxes object"""
    if isinstance(predict_result, list):
        if len(predict_result) == 1:
            return predict_result[0].boxes
        else:
            raise RuntimeError('invalid list length')
    else:
        raise TypeError('')
