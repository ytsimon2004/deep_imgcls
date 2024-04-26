from pathlib import Path
from typing import final, TypeAlias

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from matplotlib import pyplot as plt, patches
from skimage.measure import regionprops, label
from tqdm import tqdm
from ultralytics import YOLO

from imgcls.io import ImageClsDir, CACHE_DIRECTORY
from imgcls.util import fprint

ClassInt: TypeAlias = int
ClassName: TypeAlias = str
DetectClassSquare: TypeAlias = dict[ClassInt, list[tuple[float, float, float, float]]]  # cls: [xc, yc, w, h]


@final
class YoloUltralyticsPipeline:
    """TODO"""

    def __init__(self,
                 root_dir: str | Path, *,
                 resize_dim: tuple[int, int] | None = None,
                 use_gpu: bool = True,
                 epochs: int = 3):
        """

        :param root_dir:
        :param resize_dim: (w,h) resize
        """
        self.image_dir = ImageClsDir(root_dir)
        self.resize_dim = resize_dim

        if use_gpu:
            pass  # TODO

        self._epochs = epochs

    def run(self):
        # self.clone_png_dir()
        # self.gen_yaml(output=self.image_dir.root_dir / 'dataset.yml')
        # self.gen_label_txt(debug_mode=False)
        model = self.yolo_train()
        self.yolo_predict(model)

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            fprint('Process using cuda GPU')
            return torch.device('cuda')
        elif check_mps_available():
            fprint('Process using mps GPU')
            return torch.device('mps')
        else:
            fprint(f'GPU not available, using CPU instead')
            return torch.device('cpu')

    def clone_png_dir(self):
        fprint('<STATE 1> -> clone dir')

        _clone_png_dir(self.image_dir.train_img, self.resize_dim)
        _clone_png_dir(self.image_dir.train_seg, self.resize_dim)
        _clone_png_dir(self.image_dir.test_img)  # no need resize for prediction

    def gen_yaml(self, output: Path | str) -> None:
        """
        Generate the yaml for yolov8 config

        :param output: output filepath of the yaml file
        :return:
        """
        fprint('<STATE 2> -> generate yaml file')

        dy = {
            'path': str(self.image_dir.root_dir),
            'train': str(self.image_dir.train_img_png),
            'val': str(self.image_dir.train_img_png),
            'test': str(self.image_dir.test_img_png),
            'names': self._create_label_dict()
        }

        with open(output, 'w') as file:
            yaml.dump(dy, file, sort_keys=False)

    def _create_label_dict(self) -> dict[ClassInt, ClassName]:
        df = self.image_dir.train_dataframe.drop('Id')
        return {
            i: df.columns[i]
            for i in range(df.shape[1])
        }

    def gen_label_txt(self,
                      debug_mode: bool = False,
                      debug_show_raw: bool = True) -> None:
        """
        Detect the object edge from seg files and generate the yolo4 required label file for train dataset ::

        <class_id> <xc> <y_center> <width> <height>

        :return:
        """
        fprint('<STATE 3> -> auto annotate segmentation file and generate label txt')
        iter_seg = tqdm(self.image_dir.train_seg.glob('*.npy'),
                        unit='file',
                        ncols=80,
                        desc='detect edge')

        for seg in iter_seg:
            im = np.load(seg)

            if self.resize_dim is not None:
                im = cv2.resize(im, dsize=self.resize_dim, interpolation=cv2.INTER_NEAREST)

            detected = detect_segmented_objects(im)

            if debug_mode:
                fprint(f'IMAGE -> {seg.stem}')
                fprint(f'DETECTED RESULT -> {detected}')
                fig, ax = plt.subplots()

                if debug_show_raw:
                    _, _, idx = seg.stem.partition('_')
                    file = list(self.image_dir.train_img_png.glob(f'*{idx}.png'))[0]
                    raw = Image.open(str(file))
                    ax.imshow(raw)
                else:
                    ax.imshow(raw)

                colors = plt.cm.rainbow(np.linspace(0, 1, len(detected)))
                for cls, info in detected.items():
                    color = colors[cls % len(colors)]  # cycle through colors
                    for (xc, yc, width, height) in info:
                        rect = patches.Rectangle((xc - width / 2, yc - height / 2),
                                                 width, height, linewidth=1, edgecolor=color, facecolor='none')
                        ax.add_patch(rect)

                plt.show()

            else:
                write_yolo_label_txt(seg, detected,
                                     self.resize_dim if self.resize_dim is not None else im.shape,
                                     output_dir=self.image_dir.train_img_png)

    def yolo_train(self, save: bool = True) -> YOLO:
        fprint('<STATE 4> -> Train the dataset using yolov8')
        model = YOLO('yolov8n.pt')
        # model = model.to(device=self.device)
        ret = model.train(data=self.image_dir.root_dir / 'dataset.yml', epochs=self._epochs)  # TODO check return
        ret = model.val()  # TODO check return

        if save:
            model.export(format='onnx')

        return model

    def yolo_predict(self, model, save_plot: bool = True, save_txt: bool = True):
        model.predict(source=self.image_dir.test_img_png, save=save_plot, save_txt=save_txt)

    def create_predicted_csv(self):
        pass


def _clone_png_dir(directory: Path | str,
                   resize_dim: tuple[int, int] | None = None):
    """Clone batch raw .npy files to png in a separated folder, image resize if needed

    :param directory: directory contains .npy files
    :param resize_dim: resize dim in (w, h)
    """
    dst = Path(directory).parent / f'{directory.stem}_png'
    if not dst.exists():
        dst.mkdir()

    iter_file = tqdm(Path(directory).glob('*.npy'),
                     unit='file',
                     ncols=80,
                     desc=f'clone {directory} to png')

    for file in iter_file:
        img = np.load(file)
        if resize_dim is not None:
            img = cv2.resize(img, dsize=resize_dim, interpolation=cv2.INTER_NEAREST)

        out = (dst / file.name).with_suffix('.png')
        plt.imsave(out, img)


def detect_segmented_objects(seg_arr: np.ndarray) -> DetectClassSquare:
    """
    Detects objects in a segmented image array and returns their center, width, and height.

    TODO some could be merged for optimization

    :param seg_arr: Numpy array of the segmented image where different values represent different objects.
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
    yolo = YoloUltralyticsPipeline(root, resize_dim=(500, 500))
    yolo.run()


if __name__ == '__main__':
    main()
