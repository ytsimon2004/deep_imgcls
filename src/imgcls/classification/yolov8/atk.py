from __future__ import annotations

from pathlib import Path
from typing import Literal, get_args, Callable

import torch
import torchattacks
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from ultralytics import YOLO

from imgcls.classification.yolov8.util import extract_yolo_predict_box
from imgcls.io import ImageClsDir

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
                          epsilon: float = 6 / 255):
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

    atk = atk_func(model, eps=8 / 255, alpha=2 / 225, steps=10, random_start=True)
    print(atk)
    atk(img_dir.test_image_png / 'test_0.png')
    atk(img_dir.test_image_png / 'test_1.png')


# ========== #


class EncoderDecoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        #
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        #
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2, padding=0),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> None:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AdversarialAttack:

    def __init__(self, model_path: Path | str,
                 dataset_path: Path | str,
                 attack_type: ATTACK_TYPE, *,
                 batch_size: int = 32,
                 image_size: tuple[int, int] = (224, 224),
                 lr: float = 0.01,
                 epochs: int = 10,
                 eps: float = 0.007):
        """

        :param model_path:
        :param dataset_path:
        :param batch_size:
        :param image_size:
        :param lr:
        :param epochs:
        :param eps:
        """
        self.model = YOLO(model_path)

        # CNN
        self._batch_size = batch_size
        self._image_size = image_size
        self._lr = lr
        self._eps = eps
        self._epochs = epochs

        #
        self.transform = transforms.Compose([
            transforms.Resize(self._image_size),
            transforms.ToTensor()
        ])

        #
        self.perturbation_model = EncoderDecoderCNN()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.perturbation_model.parameters(), lr=self._lr)

        # dataset
        self.dataset = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=self.transform)
        self.data_loader = DataLoader(self.dataset, batch_size=self._batch_size, shuffle=True)

        # attack
        atk_list = get_args(ATTACK_TYPE)
        if attack_type not in atk_list:
            raise ValueError(f'unknown {attack_type}, selected from {atk_list}')
        self.atk: Callable = getattr(torchattacks, attack_type)

    def train(self) -> None:
        for epoch in range(self._epochs):
            for images, labels in self.data_loader:
                # Generate perturbations
                perturbations = self.perturbation_model(images)

                # Create adversarial examples
                adv_images = images + perturbations
                adv_images = torch.clamp(adv_images, 0, 1)  # Ensure pixel values are in range

                # Apply adversarial attack using torchattacks
                attack = self.atk(self.model, eps=self._eps)
                adv_images = attack(adv_images, labels)

                # Get predictions from the classifier
                outputs = self.model(adv_images)
                cls = extract_yolo_predict_box(outputs).cls

                # Compute the adversarial loss
                loss = self.criterion(cls, labels)

                # Backpropagation and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f'Epoch [{epoch + 1}/{self._epochs}], Loss: {loss.item():.4f}')

    def evaluate(self, test_image_path: str) -> None:
        # Switch to evaluation mode
        self.perturbation_model.eval()

        # Load and preprocess the test image
        test_image = Image.open(test_image_path)
        test_image = self.transform(test_image).unsqueeze(0)  # Add batch dimension

        # Generate perturbation
        perturbation = self.perturbation_model(test_image)
        adv_test_image = test_image + perturbation
        adv_test_image = torch.clamp(adv_test_image, 0, 1)

        # Apply adversarial attack
        attack = self.atk(self.model, eps=self._eps)
        adv_test_image = attack(adv_test_image)

        # Classify the adversarial image
        output = self.model(adv_test_image)
        _, predicted = torch.max(output, 1)

        print(f'Predicted Label: {predicted.item()}')


def main():
    import argparse

    ap = argparse.ArgumentParser(description='adversarial attack for a YOLO model')

    # train arg
    ap.add_argument('--model', metavar='PATH', type=str, required=True, help='model path')
    ap.add_argument('--data', metavar='PATH', type=str, required=True, help='dataset path for train adversarial model')
    ap.add_argument('--atk-type', type=str, choices=get_args(ATTACK_TYPE), default='PGD', help='attack type',
                    dest='atk_type')

    # TODO kwarg if need

    # test arg
    ap.add_argument('--predict', metavar='PATH', type=str, required=True, help='test image path')

    opt = ap.parse_args()

    atk = AdversarialAttack(opt.model, opt.data, opt.atk_type)
    atk.train()
    atk.evaluate(opt.predict)


if __name__ == '__main__':
    main()
