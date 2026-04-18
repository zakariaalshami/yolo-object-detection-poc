
# YOLO11 Object Detection PoC

**Practical Proof-of-Concept for Object Detection using YOLOv11 on COCO8 Dataset**

## Overview
Trained YOLOv8 Nano model on COCO8 dataset (40 images, 8 classes) for rapid prototyping.

### Results
- **mAP50:** 0.844 (84.4% accuracy) ✅
- **Precision:** 0.616
- **Recall:** 0.85 (detects 85% of objects)
- **Training Time:** ~1 minute (GPU T4)
- **Model Size:** 5.5 MB

### Detected Classes
person, dog, horse, elephant, umbrella, potted plant, car, bicycle

## Quick Start

### Requirements
```bash
pip install ultralytics torch torchvision
```

### Training
```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
results = model.train(
    data='coco8.yaml',
    epochs=3,
    imgsz=640,
    device=0
)
```

### Inference
```python
model = YOLO('runs/detect/train4/weights/best.pt')
results = model('image.jpg', conf=0.5)
results.show()
```

## Files
- `yolo_train.py` - Training script
- `training_results.png` - Metrics visualization
- Notebook - Full training pipeline (Colab)

## Use Cases
- Industrial defect detection (edge)
- Real-time object recognition
- Autonomous systems perception
- Basis for custom dataset fine-tuning

## Next Steps
1. Fine-tune on custom industrial dataset
2. Export to ONNX/TensorRT for edge deployment
3. Integrate with robotics/drone systems

---
**Author:** Zakaria Al-Shami  
**Framework:** Ultralytics YOLOv11  
**Trained on:** COCO8 (public dataset)  
**Status:** PoC Complete ✅
