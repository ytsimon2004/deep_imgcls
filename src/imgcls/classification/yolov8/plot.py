import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from imgcls.io import ImageClsDir
from imgcls.util import uglob, fprint

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
        img = uglob(img_dir.train_img_png, pattern)
        seg = uglob(img_dir.train_seg_png, pattern)

        ax[0, i].imshow(Image.open(str(img)))
        ax[1, i].imshow(Image.open(str(seg)))

        ax[0, i].set_title(f'{pattern.split(".")[0]}')
        ax[0, i].axis("off")
        ax[1, i].axis("off")

    plt.tight_layout()
    plt.show()


def dir_ipy_imshow(img_dir: ImageClsDir):
    """

    :param img_dir:
    :return:
    """
    from IPython.display import display
    from IPython.core.display_functions import clear_output
    import ipywidgets as widgets

    files = list(img_dir.predict_dir.glob('*.png'))
    files = iter(sorted(files, key=lambda it: int(it.stem.split('_')[1])))

    # for i, file in enumerate(files):
    #     fprint(f'show -> {file.stem}')
    #     img = Image.open(file)
    #     display(img)

    def on_button_clicked(b):
        # Move to next image on button click
        try:
            file = next(files)
            img = Image.open(file)
            fprint(f'show -> {file.stem}')
            clear_output(wait=True)
            display(img)
            display(button)
        except StopIteration:
            # If no more images, clear the output and remove the button
            clear_output(wait=True)

    button = widgets.Button(description="Next Image")
    button.on_click(on_button_clicked)
    on_button_clicked(None)
