from ultralytics import YOLO

f = '/opt/homebrew/runs/detect/train2/weights/best.pt'

test = '/Users/yuting/.cache/comvis/imgcls/test/img_png'

model = YOLO(f)
model.predict(test, save=True)