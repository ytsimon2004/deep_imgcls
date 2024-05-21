from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, get_args

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchattacks
import torchvision
import torchvision.datasets as dsets
from PIL import Image
from matplotlib.axes import Axes
from torch.utils.data import DataLoader
from torchvision import transforms
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


def get_pred(model, images, device):
    logits = model(images.to(device))
    _, pres = logits.max(dim=1)
    return pres.cpu()


def tensor_imshow(img: torch.Tensor, ax: Axes):
    img = torchvision.utils.make_grid(img.cpu().data, normalize=True)
    npimg = img.numpy()

    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def image_folder_custom_label(root, transform, idx2label):
    # custom_label
    # type : List
    # index -> label
    # ex) ['tench', 'goldfish', 'great_white_shark', 'tiger_shark']

    old_data = dsets.ImageFolder(root=root, transform=transform)
    old_classes = old_data.classes

    label2idx = {}

    for i, item in enumerate(idx2label):
        label2idx[item] = i

    new_data = dsets.ImageFolder(root=root, transform=transform,
                                 target_transform=lambda x: idx2label.index(old_classes[x]))
    new_data.classes = idx2label
    new_data.class_to_idx = label2idx

    return new_data


def l2_distance(model, images, adv_images, labels, device="cuda"):
    outputs = model(adv_images)
    _, pre = torch.max(outputs.data, 1)
    corrects = (labels.to(device) == pre)
    delta = (adv_images - images.to(device)).view(len(images), -1)
    l2 = torch.norm(delta[~corrects], p=2, dim=1).mean()
    return l2


@torch.no_grad()
def get_accuracy(model, data_loader, atk=None, n_limit=1e10, device=None):
    model = model.eval()

    if device is None:
        device = next(model.parameters()).device

    correct = 0
    total = 0

    for images, labels in data_loader:

        X = images.to(device)
        Y = labels.to(device)

        if atk:
            X = atk(X, Y)

        pre = model(X)

        _, pre = torch.max(pre.data, 1)
        total += pre.size(0)
        correct += (pre == Y).sum()

        if total > n_limit:
            break

    return 100 * float(correct) / total

# =================== #

def create_image_tensor(image_dir: Path, image_size=(224, 224)):
    """Create a tensor from images in a specified directory"""
    image_dir = Path(image_dir)

    # List all files in the directory
    image_files = sorted(list(image_dir.glob('*.png')), key=lambda it: int(it.stem.split('_')[1]))

    # Transformation to apply to each image (resize and convert to tensor)
    transform = transforms.Compose([
        transforms.Resize(image_size),  # Resize to the specified size
        transforms.ToTensor()  # Convert to a tensor
    ])

    # List to hold all image tensors
    image_tensors = []

    # Loop through each file, open, transform, and add to list
    for image_file in image_files:
        image = Image.open(image_file).convert('RGB')  # Ensure 3 channels
        image_tensor = transform(image)
        image_tensors.append(image_tensor)

    # Stack all image tensors into a single tensor
    images_tensor = torch.stack(image_tensors)

    return images_tensor


def do_adversarial_attack(img_dir: ImageClsDir,
                          name: str,
                          attack_type: ATTACK_TYPE,
                          epsilon: float = 6 / 255):
    """
    https://github.com/Harry24k/adversarial-attacks-pytorch/tree/master/demo

    :param img_dir: :class:`ImageClsDir`
    :param name: folder name of the yolo model weights
    :param attack_type: `ATTACK_TYPE`
    :param epsilon: attack epsilon
    :return:
    """
    model_path = img_dir.get_model_weights(name) / 'best.pt'
    model = YOLO(model_path)

    try:
        atk_func = getattr(torchattacks, attack_type)
    except AttributeError:
        raise RuntimeError(f'unknown attack type: {attack_type}, choose from {get_args(ATTACK_TYPE)}')

    atk = atk_func(model, eps=epsilon, alpha=2 / 225, steps=10, random_start=True)
    images = create_image_tensor(img_dir.train_image_png)
    labels = torch.tensor(img_dir.train_dataframe.drop('Id').to_numpy())

    adv_images = atk(images, labels)
    # TODO TypeError: cross_entropy_loss(): argument 'input' (position 1) must be Tensor, not tuple
    # FIXME likely due to ultralytics.YOLO

    fig, ax = plt.subplots()
    tensor_imshow(adv_images[0], ax=ax)


if __name__ == '__main__':
    img_dir = ImageClsDir(CACHE_DIRECTORY)
    do_adversarial_attack(img_dir, 'train', 'PGD')
