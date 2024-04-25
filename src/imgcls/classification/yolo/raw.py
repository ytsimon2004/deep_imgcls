from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def clone_raw_resize(directory: Path | str,
                     resize_dim: tuple[int, int] | None = (500, 500)):
    """clone batch raw .npy files and resize to image png"""
    dst = Path(directory).parent / 'img_png'
    if not dst.exists():
        dst.mkdir()

    for file in Path(directory).glob('*.npy'):
        img = np.load(file)
        if resize_dim is not None:
            img = cv2.resize(img, dsize=resize_dim, interpolation=cv2.INTER_NEAREST)

        out = (dst/file.name).with_suffix('.png')
        plt.imsave(out, img)


if __name__ == '__main__':
    d = '/Users/yuting/.cache/comvis/imgcls/train/img'
    clone_raw_resize(d)