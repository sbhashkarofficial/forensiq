import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoModelForImageSegmentation
from torchvision import transforms

# CONSTANTS

# 1. Enable CPU Multi-threading
torch.set_num_threads(torch.multiprocessing.cpu_count())

print("Loading local model...")
birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet",
    trust_remote_code=True
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
birefnet = birefnet.to(device).float()
birefnet.eval()


def extract_foreground(image_path):
    """Processes the image and returns both the original PIL Image and the prediction mask in memory."""
    image = Image.open(image_path).convert("RGB")

    transform_image = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_tensor = transform_image(image).unsqueeze(0).to(device, dtype=torch.float32)

    with torch.no_grad():
        preds = birefnet(input_tensor)[-1].sigmoid().cpu()

    pred_mask = preds[0]
    
    return image, pred_mask


