from typing import Literal, get_args

import torchattacks
from ultralytics import YOLO

from imgcls.io import ImageClsDir, CACHE_DIRECTORY

ATTACK_TYPE = Literal[
    "PGD",
    "FGSM",
    "VANILA",
    "GN",
    "FGSM",
    "BIM",
    "RFGSM",
    "PGD",
    "EOTPGD",
    "FFGSM",
    "TPGD",
    "MIFGSM",
    "UPGD",
    "APGD",
    "APGDT",
    "DIFGSM",
    "TIFGSM",
    "Jitter",
    "NIFGSM",
    "PGDRS",
    "SINIFGSM",
    "VMIFGSM",
    "VNIFGSM",
    "SPSA",
    "JSMA",
    "EADL1",
    "EADEN",
    "PIFGSM",
    "PIFGSMPP",
    "CW",
    "PGDL2",
    "DeepFool",
    "PGDRSL2",
    "SparseFool",
    "OnePixel",
    "Pixle",
    "FAB",
    "AutoAttack",
    "Square",
    "MultiAttack",
    "LGV"
]


def do_adversarial_attack(img_dir: ImageClsDir,
                          name: str,
                          attack_type: ATTACK_TYPE,
                          epsilon: float =6/255):
    """
    https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demo/White-box%20Attack%20on%20ImageNet.ipynb

    :param img_dir:
    :param name:
    :param attack_type:
    :param epsilon
    :return:
    """
    model_path = img_dir.get_model_weights(name) / 'best.pt'
    model = YOLO(model_path)

    try:
        atk_func = getattr(torchattacks, attack_type)
    except AttributeError:
        raise RuntimeError(f'unknown attack type: {attack_type}, choose from {get_args(ATTACK_TYPE)}')

    atk = atk_func(model,  eps=8/255, alpha=2/225, steps=10, random_start=True)
    print(atk)
    atk(img_dir.test_image_png / 'test_0.png')
    atk(img_dir.test_image_png / 'test_1.png')




if __name__ == '__main__':
    cls_dir = ImageClsDir(CACHE_DIRECTORY)
    do_adversarial_attack(cls_dir, 'train3', 'PGD')
