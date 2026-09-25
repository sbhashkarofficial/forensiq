from ultralytics import YOLO

model = YOLO("passport_dataset/weights/yolo11n_run1.pt")

results = model.predict(
    source="passport_dataset/images/val",
    conf=0.25,  # Confidence threshold
    save=True   # Save images with predicted bounding boxes
)

print(f"Inference completed! Results saved to: runs/detect/predict")