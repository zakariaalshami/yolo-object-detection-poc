
# YOLO11 Training on COCO8
# mAP50: 0.844 (84.4%)
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
results = model.train(
    data='coco8.yaml',
    epochs=3,
    imgsz=640,
    device=0
)
print(f"Best mAP50: {results.box.map50}")
