from imgcls.io import CACHE_DIRECTORY, TRAIN_DIR, TEST_DIR
import polars as pl
import yaml

def generate_yolov5_yml(root_dir: str | None = None,
                        train_dir: str | None = None,
                        val_dir: str | None = None,
                        test_dir: str | None = None,
                        output_filename: str = 'dataset.yml'):

    if root_dir is None:
        root_dir = str(CACHE_DIRECTORY)
    if train_dir is None:
        train_dir = str(TRAIN_DIR / 'img_png')
    if val_dir is None:
        val_dir = str(TRAIN_DIR / 'img_png')
    if test_dir is None:
        test_dir = str(TEST_DIR / 'img_png')

    dy = {
        'path': root_dir,
        'train': train_dir,
        'val': val_dir,
        'test': test_dir,
        'names': create_label_dict()

    }

    with open(output_filename, 'w') as file:
        yaml.dump(dy, file, sort_keys=False)


def create_label_dict() -> dict[int, str]:
    """"""
    file = TRAIN_DIR / 'train_set.csv'
    df = pl.read_csv(file).drop('Id')
    return {
        i: df.columns[i]
        for i in range(df.shape[1])
    }


if __name__ == '__main__':
    generate_yolov5_yml()