from pathlib import Path

from ultralytics import YOLO

f = '/opt/homebrew/runs/detect/train/weights/best.pt'
model = YOLO(f)

test = '/Users/yuting/.cache/comvis/imgcls/test/img_png'

# for file in Path(test).glob('*.png'):
#     model(file)

results = model.predict(source=test, save=True)

