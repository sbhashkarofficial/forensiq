import cv2
from ultralytics import YOLO

model = YOLO("passport_dataset/weights/best.pt")

def apply_yolo_segmentation(img_path):
    results = model(source=img_path, save=False)
    return results

    #display output
    #results[0].show()

