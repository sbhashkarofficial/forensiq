import cv2
import os
import matplotlib.pyplot as plt

# Map your class IDs to their actual names for the labels
CLASS_NAMES = {
    0: "main_photo",
    1: "mrz",
    2: "signature",
}

def verify_yolo_labels(image_path, label_path):
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return
        
    img_h, img_w, _ = img.shape
    
    # OpenCV loads images in BGR format, convert to RGB for accurate color display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Open and read the YOLO .txt file
    if not os.path.exists(label_path):
        print(f"Error: Label file not found at {label_path}")
        return
        
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # 3. Process each line in the text file
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
            
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        norm_width = float(parts[3])
        norm_height = float(parts[4])

        # 4. Convert YOLO normalized coordinates back to absolute pixel coordinates
        # xmin = (x_center - width/2) * image_width
        xmin = int((x_center - (norm_width / 2)) * img_w)
        ymin = int((y_center - (norm_height / 2)) * img_h)
        xmax = int((x_center + (norm_width / 2)) * img_w)
        ymax = int((y_center + (norm_height / 2)) * img_h)

        # 5. Draw the bounding box (Red color, thickness of 3)
        cv2.rectangle(img_rgb, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)
        
        # 6. Add the class text label right above the box
        label_text = CLASS_NAMES.get(class_id, f"Class {class_id}")
        cv2.putText(img_rgb, label_text, (xmin, ymin - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # 7. Display the final image with boxes
    plt.figure(figsize=(12, 12))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

# --- Execution Setup ---
# Define your base dataset directory
dataset_path = "passport_dataset"
split = "val"  # Change to "val" to check validation images

# The filename you want to check (do NOT include the .jpg or .txt extension here)
file_name = "srb_passport_80"

# Construct the full paths based on the strict YOLO directory structure
img_file = os.path.join(dataset_path, "images", split, f"{file_name}.jpg")  # Change .jpg to .png if necessary
txt_file = os.path.join(dataset_path, "labels", split, f"{file_name}.txt")

# Run the visualization
verify_yolo_labels(img_file, txt_file)