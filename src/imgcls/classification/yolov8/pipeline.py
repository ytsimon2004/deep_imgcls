from pathlib import Path
from pprint import pprint
from typing import final, TypeAlias, Final, Literal

import cv2
import numpy as np
import polars as pl
import torch
import yaml
from PIL import Image
from matplotlib import pyplot as plt, patches
from skimage.measure import regionprops, label
from tqdm import tqdm
from ultralytics import YOLO

from imgcls.io import ImageClsDir, CACHE_DIRECTORY
from imgcls.util import fprint

__all__ = ['YoloUltralyticsPipeline']

ClassInt: TypeAlias = int
ClassName: TypeAlias = str
DetectClassSquare: TypeAlias = dict[ClassInt, list[tuple[float, float, float, float]]]  # cls: [xc, yc, w, h]

YOLO8_MODEL_TYPE = Literal['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']


@final
class YoloUltralyticsPipeline:
    """Custom pipeline for implementing the YOLOv8 from ultralytics

    .. seealso :: `<https://docs.ultralytics.com/modes/train/>`_

    """

    def __init__(self,
                 root_dir: str | Path, *,
                 model_type: YOLO8_MODEL_TYPE = 'yolov8n',
                 model_path: str | Path | None = None,
                 resize_dim: tuple[int, int] | None = None,
                 use_gpu: bool = False,
                 epochs: int = 10,
                 batch_size: int = 32):
        """

        :param root_dir:
        :param model_type: {'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'}
        :param model_path: model path (If already trained)
        :param resize_dim: (w,h) resize
        :param use_gpu: whether use gpu for fine-tuned the model
        :param epochs: number of the training epoch
        :param batch_size: training bathc size
        """
        self.image_dir = ImageClsDir(root_dir)
        self.resize_dim = resize_dim

        # label_dict, order sensitive
        df = self.image_dir.train_dataframe.drop('Id')
        self.label_dict: Final[dict[ClassInt, ClassName]] = {
            i: df.columns[i]
            for i in range(df.shape[1])
        }

        # training/prediction parameters
        self.model_type = model_type
        self.model = model_path  # if already fine-tuned. If None, then auto-inferred
        self._epochs = epochs
        self._lr0 = 0.01
        self._batch = batch_size
        self._train_filename: str | None = None  # incremental folder name, assign foreach train
        self._predict_filename: str | None = None  # incremental folder name, assign foreach predict

        # resources
        if use_gpu:
            if torch.cuda.is_available():
                fprint('Process using cuda GPU')
                self._device = torch.device('cuda')
            elif check_mps_available():  # use cpu mode or increase the batch size if NMS time issue
                fprint('Process using mps GPU')
                self._device = torch.device('mps')
                self._lr0 = 0.00025
        else:
            self._device = None  # torch.device('cpu')?

    def run(self):
        # if fine-tuned model already specified
        if self.model is None:
            self.clone_png_dir()
            self.gen_yaml()
            self.gen_label_txt(debug_mode=False)
            self.yolo_train(model_type=self.model_type)

        self.yolo_predict(self.model)
        self.create_predicted_csv()

    @property
    def n_test(self) -> int:
        return len(list(self.image_dir.test_img_png.glob('*.png')))

    @property
    def train_filename(self) -> str:
        if self._train_filename is None:
            raise RuntimeWarning('run model train first')
        return self._train_filename

    @property
    def predict_filename(self) -> str:
        if self._predict_filename is None:
            raise RuntimeWarning('run model predict first')
        return self._predict_filename

    def clone_png_dir(self) -> None:
        """Clone batch raw .npy files to png in a separated folder, image resize if needed"""
        fprint('<STATE 1> -> clone png dir')

        _clone_png_dir(self.image_dir.train_img, self.resize_dim)
        _clone_png_dir(self.image_dir.train_seg, self.resize_dim)
        _clone_png_dir(self.image_dir.test_img)  # no need resize for prediction

    def gen_yaml(self, output: Path | str | None = None, verbose: bool = False) -> None:
        """
        Generate the yaml for yolov8 config

        :param output: output filepath of the yaml file
        :param verbose: show output verbose
        :return:
        """
        fprint('<STATE 2> -> generate yaml file')

        if output is None:
            output = self.image_dir.root_dir / 'dataset.yml'

        dy = {
            'path': str(self.image_dir.root_dir),
            'train': str(self.image_dir.train_img_png),
            'val': str(self.image_dir.train_img_png),
            'test': str(self.image_dir.test_img_png),
            'names': self.label_dict
        }

        with open(output, 'w') as file:
            yaml.safe_dump(dy, file, sort_keys=False)

        if verbose:
            with open(output, 'rb') as file:
                config = yaml.safe_load(file)
                pprint(config)

    def gen_label_txt(self, debug_mode: bool = True, min_area: int = 500) -> None:
        """
        Detect the object edge from seg files and generate the yolov8 required label file for train dataset ::

        <class_id> <xc> <y_center> <width> <height>

        :param debug_mode: debug mode to see the train dataset segmentation result
        :param min_area: minimum area threshold to consider an object. smaller areas are ignored
        :return:
        """
        fprint('<STATE 3> -> auto annotate segmentation file and generate label txt')

        files = sorted(list(self.image_dir.train_seg.glob('*.npy')),
                       key=lambda it: int(it.stem.split('_')[1]))

        iter_seg = tqdm(files,
                        total=len(files),
                        unit='file',
                        ncols=80,
                        desc='detect edge')

        for seg in iter_seg:
            im = np.load(seg)

            if self.resize_dim is not None:
                im = cv2.resize(im, dsize=self.resize_dim, interpolation=cv2.INTER_NEAREST)

            detected = detect_segmented_objects(im, min_area=min_area)

            if debug_mode:
                fprint(f'IMAGE -> {seg.stem}')
                fprint(f'DETECTED RESULT -> {detected}')
                fig, ax = plt.subplots(1, 2)

                # query raw
                _, _, idx = seg.stem.partition('_')
                file = list(self.image_dir.train_img_png.glob(f'*_{idx}.png'))[0]
                raw = Image.open(str(file))
                ax[0].imshow(raw)

                ax[1].imshow(im)

                # draw
                colors = plt.cm.rainbow(np.linspace(0, 1, 20))
                legend_handles = []
                for cls, info in detected.items():
                    color = colors[cls % len(colors)]  # cycle through colors
                    for (xc, yc, width, height) in info:
                        rect = patches.Rectangle((xc - width / 2, yc - height / 2),
                                                 width, height,
                                                 linewidth=1,
                                                 edgecolor=color,
                                                 facecolor='none')
                        ax[0].add_patch(rect)

                    legend_patch = patches.Patch(color=color, label=self.label_dict[cls])
                    legend_handles.append(legend_patch)

                ax[0].legend(handles=legend_handles, loc='best')

                plt.show()

            #
            write_yolo_label_txt(seg, detected,
                                 self.resize_dim if self.resize_dim is not None else im.shape,
                                 output_dir=self.image_dir.train_img_png)

    def yolo_train(self, model_type: YOLO8_MODEL_TYPE = 'yolov8n',
                   save: bool = True) -> None:
        """Load a pretrained model for training"""
        fprint(f'<STATE 4> -> Train the dataset using {model_type}')
        model_path = (self.image_dir.run_dir / model_type).with_suffix('.pt')
        model = YOLO(model_path)

        model.train(data=self.image_dir.root_dir / 'dataset.yml',
                    device=self._device,
                    batch=self._batch,
                    lr0=self._lr0,
                    project=self.image_dir.run_dir,
                    epochs=self._epochs,
                    cache=True)

        self._train_filename = model.overrides['name']

        if save:
            model.export(format='onnx')

    def yolo_predict(self, model_path: Path | str | None = None,
                     save_plot: bool = True,
                     save_txt: bool = True):
        fprint('<STATE 5> -> Predicted result using test dataset')

        if model_path is None:
            model_path = self.image_dir.get_model_weights(self._train_filename) / 'best.pt'

        model = YOLO(model_path)
        model.predict(source=self.image_dir.test_img_png,
                      save=save_plot,
                      save_txt=save_txt,
                      project=self.image_dir.run_dir)

        self._predict_filename = model.predictor.save_dir.name

    def create_predicted_csv(self) -> None:
        fprint('<STATE 6> -> Write predicted result to csv')

        ret = {}
        for txt in self.image_dir.get_predict_label_dir(self.predict_filename).glob('test*.txt'):
            classes = set()
            with open(txt, 'r') as file:
                for line in file:
                    cls = line.split(' ')[0]
                    classes.add(cls)

            _, _, idx = txt.stem.partition('_')
            ret[idx] = list(classes)

        ret = dict(sorted(ret.items()))

        #
        dy = dict(Id=np.arange(self.n_test))
        for i, field in enumerate(self.label_dict.values()):
            dy[field] = np.full(self.n_test, 0)

        df = pl.DataFrame(dy)

        for i, classes in ret.items():
            for cls in classes:
                df[int(i), self.label_dict[int(cls)]] = 1

        dst = self.image_dir.get_predict_dir(self.predict_filename) / 'test_set.csv'
        df.write_csv(dst)
        fprint(f'Successful create result in {dst}', vtype='io')


def _clone_png_dir(directory: Path | str,
                   resize_dim: tuple[int, int] | None = None) -> None:
    """Clone batch raw .npy files to png in a separated folder, image resize if needed

    :param directory: directory contains .npy files
    :param resize_dim: resize dim in (w, h) if not None
    """
    dst = Path(directory).parent / f'{directory.stem}_png'
    if not dst.exists():
        dst.mkdir()

    files = list(Path(directory).glob('*.npy'))
    iter_file = tqdm(files,
                     total=len(files),
                     unit='file',
                     desc=f'npy clone and resize as png to {dst}')

    for file in iter_file:
        img = np.load(file)
        if resize_dim is not None:
            img = cv2.resize(img, dsize=resize_dim, interpolation=cv2.INTER_NEAREST)

        out = (dst / file.name).with_suffix('.png')
        plt.imsave(out, img)


def detect_segmented_objects(seg_arr: np.ndarray, min_area: int = 500) -> DetectClassSquare:
    """
    Detects objects in a segmented image array and returns their center, width, and height.

    TODO some could be merged for optimization if needed

    :param seg_arr: array of the segmented image where different values represent different objects
    :param min_area: minimum area threshold to consider an object. smaller areas are ignored
    :return: A dictionary where keys are object classes and values are lists of tuples containing
    (x_center, y_center, width, height) for each object of that class.
    """

    objects_info = {}

    # the unique values in the image array represent different classes
    object_classes = np.unique(seg_arr)
    # exclude the background class (value equal to 0)
    object_classes = object_classes[object_classes != 0]

    for object_class in object_classes:
        class_mask = (seg_arr == object_class)

        # label connected regions of the mask
        label_img = label(class_mask)
        regions = regionprops(label_img)

        class_objects_info = []
        for region in regions:

            if region.area < min_area:
                continue

            y0, x0, y1, x1 = region.bbox
            x_center = (x0 + x1) / 2
            y_center = (y0 + y1) / 2
            width = x1 - x0
            height = y1 - y0
            class_objects_info.append((x_center, y_center, width, height))

        objects_info[object_class - 1] = class_objects_info  # 1-base to 0-base

    return objects_info


def write_yolo_label_txt(img_filepath: str | Path,
                         detected: DetectClassSquare,
                         img_dim: tuple[int, int],
                         output_dir: str | Path) -> None:
    filename = Path(img_filepath).stem
    out = (Path(output_dir) / filename).with_suffix('.txt')

    width, height = img_dim
    with open(out, 'w') as file:
        for cls, info in detected.items():
            for (xc, yc, w, h) in info:
                x_center_normalized = xc / width
                y_center_normalized = yc / height
                width_normalized = w / width
                height_normalized = h / height
                content = f'{cls} {x_center_normalized} {y_center_normalized} {width_normalized} {height_normalized}\n'
                file.write(content)


def check_mps_available() -> bool:
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            fprint('MPS not available because pytorch install not built with MPS enable', vtype='warning')
        else:
            fprint('MPS not available because current MacOs version is not 12.3+,'
                   ' or do not have MPS-enabled device on this machine', vtype='warning')
        return False
    else:
        fprint('MPS is available', vtype='pass')
        return True


def main():
    root = CACHE_DIRECTORY
    path = ...
    yolo = YoloUltralyticsPipeline(root, resize_dim=(500, 500), model_path=path)
    yolo.run()


if __name__ == '__main__':
    main()
