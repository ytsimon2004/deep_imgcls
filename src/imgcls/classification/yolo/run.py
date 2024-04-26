from ultralytics import YOLO

model = YOLO('yolov8n.pt')
result = model.train(data='/Users/yuting/code/yolov5/imgcls.yml', epochs=3)
results = model.val()
success = model.export(format='onnx')