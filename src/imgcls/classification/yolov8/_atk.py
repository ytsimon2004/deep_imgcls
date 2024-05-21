import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import transforms
from ultralytics import YOLO

from imgcls.classification.yolov8.util import extract_yolo_predict_box
from imgcls.io import ImageClsDir, CACHE_DIRECTORY
from imgcls.util import fprint


class AdversarialAutoencoder(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int]) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, input_shape[0], kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AdversarialAttackModel:
    def __init__(self,
                 yolo_model: nn.Module,
                 input_shape: tuple[int, int, int] = (3, 416, 416),
                 reg_weight: float = 0.1,
                 lr: float = 0.001):
        """

        :param yolo_model:
        :param input_shape:
        :param reg_weight:
        :param lr:
        """
        self.yolo_model = yolo_model
        self.reg_weight = reg_weight
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo_model.to(self.device)
        self.freeze_yolo_weights()

        self.adversary = AdversarialAutoencoder(input_shape).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.adversary.parameters(), lr=lr)

    def freeze_yolo_weights(self) -> None:
        for param in self.yolo_model.parameters():
            param.requires_grad = False  # Freeze YOLO model parameters

    def adversarial_loss(self,
                         y_pred: torch.Tensor,
                         y_true: torch.Tensor,
                         perturbation: torch.Tensor) -> torch.Tensor:
        fprint(f'{y_pred=}, {y_pred.shape=}')
        fprint(f'{y_true=}, {y_true.shape=}', vtype='io')
        classification_loss = self.criterion(y_pred, y_true)
        reg_loss = torch.norm(perturbation, p=2)  # L2 norm
        return classification_loss + self.reg_weight * reg_loss

    def train(self,
              dataloader: DataLoader,
              num_epochs: int) -> None:
        self.adversary.train()

        for epoch in range(num_epochs):
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                perturbations = self.adversary(images)
                perturbed_images = images + perturbations
                perturbed_images = torch.clamp(perturbed_images, 0, 1)

                total_loss = torch.tensor(0.0)
                for i in range(32):
                    perturbed_img = perturbed_images[i].unsqueeze(0)
                    out = extract_yolo_predict_box(self.yolo_model(perturbed_img))
                    loss = self.adversarial_loss(out[0].cls, labels[i], perturbations[i])
                    fprint(f'{loss=}')
                    total_loss += loss

                # out = extract_yolo_predict_box(self.yolo_model(perturbed_images))
                # for i, yolo_predictions in enumerate(out):
                #     yolo_predictions_data = yolo_predictions.data
                #     loss = self.adversarial_loss(yolo_predictions_data, labels[i], perturbations[i])
                #     loss.backward()
                total_loss.backward()
                self.optimizer.step()

    def evaluate(self, test_dataloader: DataLoader) -> None:
        self.adversary.eval()
        with torch.no_grad():
            for test_images, test_labels in test_dataloader:
                test_images, test_labels = test_images.to(self.device), test_labels.to(self.device)
                perturbations = self.adversary(test_images)
                perturbed_images = test_images + perturbations
                predictions = self.yolo_model(perturbed_images)
                # Calculate and print evaluation metrics

    def visualize_examples(self, images: torch.Tensor) -> None:
        self.adversary.eval()
        with torch.no_grad():
            images = images.to(self.device)
            perturbations = self.adversary(images)
            perturbed_images = images + perturbations

            for i in range(min(3, len(images))):
                self.show_images(images[i].cpu(), perturbations[i].cpu(), perturbed_images[i].cpu())

    @staticmethod
    def show_images(original: torch.Tensor,
                    perturbation: torch.Tensor,
                    perturbed: torch.Tensor) -> None:
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(original.permute(1, 2, 0).numpy())
        ax[0].set_title('Original')
        ax[1].imshow(perturbation.permute(1, 2, 0).numpy())
        ax[1].set_title('Perturbation')
        ax[2].imshow(perturbed.permute(1, 2, 0).numpy())
        ax[2].set_title('Perturbed')
        plt.show()


class TestDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.root_dir = root
        self.transform = transform
        self.image_paths = [os.path.join(root, img) for img in os.listdir(root) if img.endswith(('jpg', 'jpeg', 'png'))]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def main():
    file = '/Users/yuting/.cache/comvis/imgcls/epochs_100/weights/best.pt'
    yolo_model = YOLO(file)

    dataset_src = ImageClsDir(CACHE_DIRECTORY)

    trans = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5,))
    ])
    train_dataset = datasets.ImageFolder(root=dataset_src.attack_train_dir, transform=trans)
    test_dataset = TestDataset(root=dataset_src.attack_test_dir, transform=trans)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # train_dataloader = DataLoader()

    model = AdversarialAttackModel(yolo_model)
    model.train(train_dataloader, num_epochs=10)
    model.evaluate(test_dataloader)

    for test_images, _ in test_dataloader:
        model.visualize_examples(test_images)


if __name__ == '__main__':
    main()
