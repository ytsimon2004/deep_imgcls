from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np
from matplotlib import pyplot as plt, patches
from skimage.measure import regionprops, label
from tqdm import tqdm

from imgcls.io import TRAIN_DIR


def gen_yolo4_label_file(resize_dim: tuple[int, int] | None = (500, 500),
                         debug_mode: bool = False):
    """
    Generate the yolo4 required label file for train dataset ::

        <class_id> <xc> <y_center> <width> <height>

    from image segmentation image file

    :return:
    """
    iter_seg = (TRAIN_DIR / 'seg').glob('*.npy')

    for seg in tqdm(iter_seg, unit='n_images', desc='process yolo trained label'):
        _, _, idx = seg.stem.partition('_')
        im = np.load(seg)

        # resize
        if resize_dim is not None:
            im = cv2.resize(im, dsize=resize_dim, interpolation=cv2.INTER_NEAREST)

        #
        detected = detect_segmented_objects(im)

        if debug_mode:
            print(f'IMAGE -> {seg.stem}')
            print(f'DETECTED RESULT -> {detected}')
            fig, ax = plt.subplots()
            ax.imshow(im)

            colors = plt.cm.rainbow(np.linspace(0, 1, len(detected)))
            for cls, info in detected.items():
                color = colors[cls % len(colors)]  # Cycle through colors
                for (xc, yc, width, height) in info:
                    rect = patches.Rectangle((xc - width / 2, yc - height / 2),
                                             width, height, linewidth=1, edgecolor=color, facecolor='none')
                    ax.add_patch(rect)

            plt.show()

        else:
            write_yolo_label_txt(seg, detected, resize_dim if resize_dim is not None else im.shape)


DETECT_LABEL_TYPE: TypeAlias = dict[int, list[tuple[float, float, float, float]]]  # cls: [xc, yc, w, h]


def detect_segmented_objects(seg_arr: np.ndarray) -> DETECT_LABEL_TYPE:
    """
    Detects objects in a segmented image array and returns their center, width, and height.

    TODO some could be merged for optimization

    :param seg_arr: Numpy array of the segmented image where different values represent different objects.
    :return: A dictionary where keys are object classes and values are lists of tuples containing
    (x_center, y_center, width, height) for each object of that class.
    """

    objects_info = {}

    # The unique values in the image array represent different classes
    object_classes = np.unique(seg_arr)
    # Exclude the background class, usually represented by 0
    object_classes = object_classes[object_classes != 0]

    # object_classes -= 1
    print(f'{object_classes=}')

    for object_class in object_classes:
        # Create a mask for the current class
        class_mask = (seg_arr == object_class)

        # Label connected regions of the mask
        label_img = label(class_mask)
        regions = regionprops(label_img)

        # Extract the bounding box info for each region
        class_objects_info = []
        for region in regions:
            y0, x0, y1, x1 = region.bbox
            x_center = (x0 + x1) / 2
            y_center = (y0 + y1) / 2
            width = x1 - x0
            height = y1 - y0
            class_objects_info.append((x_center, y_center, width, height))

        objects_info[object_class - 1] = class_objects_info

    return objects_info


def write_yolo_label_txt(img_filepath: str, detected: DETECT_LABEL_TYPE, img_dim: tuple[int, int]) -> None:
    out = Path(img_filepath).with_suffix('.txt')

    height, width = img_dim
    with open(out, 'w') as file:
        for cls, info in detected.items():
            for (xc, yc, w, h) in info:
                x_center_normalized = xc / height
                y_center_normalized = yc / height
                width_normalized = w / width
                height_normalized = h / width
                content = f'{cls} {x_center_normalized} {y_center_normalized} {width_normalized} {height_normalized}\n'
                file.write(content)


if __name__ == '__main__':
    gen_yolo4_label_file()
