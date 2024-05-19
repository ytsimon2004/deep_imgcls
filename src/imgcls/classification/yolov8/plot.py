from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from imgcls.io import ImageClsDir
from imgcls.util import uglob

__all__ = ['plot_image_seg',
           'dir_ipy_imshow']


def plot_image_seg(img_dir: ImageClsDir, index_range: tuple[int, int]):
    """
    Visualize image and its segmentation

    :param img_dir: :class:`ImageClsDir`
    :param index_range: index range for images.
    :return:
    """

    n_images = index_range[1] - index_range[0]
    fig, ax = plt.subplots(2, n_images)

    for i, idx in enumerate(np.arange(*index_range)):
        pattern = f'train_{idx}.png'
        img = uglob(img_dir.train_image_png, pattern)
        seg = uglob(img_dir.train_seg_png, pattern)

        ax[0, i].imshow(Image.open(str(img)))
        ax[1, i].imshow(Image.open(str(seg)))

        ax[0, i].set_title(f'{pattern.split(".")[0]}')
        ax[0, i].axis("off")
        ax[1, i].axis("off")

    plt.tight_layout()
    plt.show()


def dir_ipy_imshow(directory: Path | str,
                   pattern: str = '*.png') -> None:
    """
    Display images from a directory with a button to load the next image

    :param directory: directory contain image sequences
    :param pattern: glob pattern in the directory
    :return:
    """
    from IPython.display import display
    from IPython.core.display import clear_output
    import ipywidgets as widgets

    files = sorted(list(Path(directory).glob(pattern)), key=lambda it: int(it.stem.split('_')[1]))
    iter_files = iter(files)

    image_display = widgets.Image()
    button = widgets.Button(description="Next Image")

    def on_button_clicked(b):
        try:
            file = next(iter_files)
        except StopIteration:
            clear_output(wait=True)
        else:
            with open(file, 'rb') as f:
                img = f.read()
            image_display.value = img

    button.on_click(on_button_clicked)
    display(button)
    display(image_display)
    on_button_clicked(None)
