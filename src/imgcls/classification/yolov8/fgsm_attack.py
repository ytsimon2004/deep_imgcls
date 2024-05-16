import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms
from ultralytics import YOLO

from imgcls.util import fprint

# Load the image
img = Image.open("/Users/yuting/.cache/comvis/imgcls/test/img_png/test_12.png").convert("RGB")

# Preprocess the image
preprocess = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

input_image = preprocess(img).unsqueeze(0)  # Add batch dimension


def fgsm_attack(image, epsilon, gradient):
    # Get the sign of the gradients
    sign_gradient = gradient.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_gradient
    # Clip the image to ensure it's within the valid range [0,1]
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


# Set requires_grad attribute of the input image for gradient computation
input_image.requires_grad = True

model = YOLO('/Users/yuting/.cache/comvis/imgcls/run/train3/weights/best.pt')
# Forward pass the image through the model
output = model(img, save=True)


loss = torch.nn.functional.cross_entropy(output, torch.tensor([11]))  # Assuming you have a target class

# Zero all existing gradients
model.zero_grad()

# Backward pass to calculate the gradients
loss.backward()

# Collect the gradients of the input image
gradient = input_image.grad.data

# Set epsilon value for the attack
epsilon = 0.01

# Generate the adversarial image
perturbed_image = fgsm_attack(input_image, epsilon, gradient)