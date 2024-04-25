from ultralytics import YOLO

model = YOLO('yolov8n.pt')
result = model.train(data='/Users/yuting/code/yolov5/imgcls.yml', epochs=3)
results = model.val()

model('/Users/yuting/.cache/comvis/imgcls/test/img_png/img_test_0.png')
success = model.export(format='onnx')