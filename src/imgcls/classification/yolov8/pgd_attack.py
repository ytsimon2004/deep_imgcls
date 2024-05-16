import torchattacks
from ultralytics import YOLO

f = '/Users/yuting/.cache/comvis/imgcls/run/train3/weights/best.pt'
img = '/Users/yuting/.cache/comvis/imgcls/train/img_png/train_549.png'

model = YOLO(f)

epsilon = 0.1  # Define the epsilon value
attack = torchattacks.PGD(model, eps=epsilon, alpha=2/225, steps=10, random_start=True)
attack.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#
perturbed_img = attack(img, None)
#
# print(perturbed_img.shape)