import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from imgcls.io import ImageClsDir
from imgcls.util import uglob

__all__ = ['plot_image_seg']


def plot_image_seg(img_dir: ImageClsDir, index_range: tuple[int, int]):
    """
    Visual image and its segmentation

    :param img_dir: :class:`ImageClsDir`
    :param index_range: index range for images.
    :return:
    """

    n_images = index_range[1] - index_range[0]
    fig, ax = plt.subplots(2, n_images)

    for i, idx in enumerate(np.arange(*index_range)):
        pattern = f'train_{idx}.png'
        img = uglob(img_dir.train_img_png, pattern)
        seg = uglob(img_dir.train_seg_png, pattern)

        ax[0, i].imshow(Image.open(str(img)))
        ax[1, i].imshow(Image.open(str(seg)))

        ax[0, i].set_title(f'{pattern.split(".")[0]}')
        ax[0, i].axis("off")
        ax[1, i].axis("off")

    plt.tight_layout()
    plt.show()
