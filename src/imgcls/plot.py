from matplotlib import pyplot as plt
import polars as pl

from imgcls.data import ImageClassification

__all__ = ['plot_example']

def plot_example(dat: ImageClassification, num: int = 1):
    n = dat.n_labels
    fig, ax = plt.subplots(2, n, figsize=(3 * n, 4), sharex=True, sharey=True)

    label_index = dat.image_labels_literal

    for i, label in enumerate(dat.labels_list):
        df = dat.train_data.filter(pl.col(label) == 1)

        ax[0, i].imshow(df['img'][num], vmin=0, vmax=255)
        ax[0, i].set_title("\n".join(label_index[df['Id'][num]]), fontsize=5)
        ax[1, i].imshow(df["seg"][num], vmin=0, vmax=20)

        ax[0, i].axis("off")
        ax[1, i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    td = ImageClassification.load()
    plot_example(td)
