import torchattacks
from ultralytics import YOLO

f = '/Users/yuting/.cache/comvis/imgcls/run/train3/weights/best.pt'
img = '/Users/yuting/.cache/comvis/imgcls/train/img_png/train_549.png'

model = YOLO(f)

epsilon = 0.1  # Define the epsilon value
attack = torchattacks.FGSM(model, eps=epsilon)
#
perturbed_img = attack(img, None)
