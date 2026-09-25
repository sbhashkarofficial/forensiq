import os
import json
import numpy as np
import cv2 as cv

print(os.getcwd())

Class_mapping = {"main_photo": 0, "mrz_zone": 1, "signature": 2}

def normalize_annotations(shat, img_width, img_height):
    [x, y, w, h] = [shat["x"], shat["y"], shat["width"], shat["height"]]
    return [(x + w / 2) / img_width, (y + h / 2) / img_height, w / img_width, h / img_height]

def get_mrz_annotations(mrz1, mrz2, img_width, img_height):
    [x1, y1, w1, h1] = [mrz1["x"], mrz1["y"], mrz1["width"], mrz1["height"]]
    [x2, y2, w2, h2] = [mrz2["x"], mrz2["y"], mrz2["width"], mrz2["height"]]
    [x, y] = [min(x1, x2), min(y1, y2)]
    [w, h] = [max(x1 + w1, x2 + w2) - x, max(y1 + h1, y2 + h2) - y]
    return [(x + w / 2) / img_width, (y + h / 2) / img_height, w / img_width, h / img_height]

def stringify_annotations(class_id, annotation):
    [x, y, w, h] = annotation
    return str(class_id) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n"

def write_annotations(filename, photo_ann, mrz_ann, sig_ann):
    with open(filename, "w") as f:
        if photo_ann:
            f.write(stringify_annotations(Class_mapping["main_photo"], photo_ann))
        if mrz_ann:
            f.write(stringify_annotations(Class_mapping["mrz_zone"], mrz_ann))
        if sig_ann:
            f.write(stringify_annotations(Class_mapping["signature"], sig_ann))

target_img_dir = "../dataset/images/grc_passport"
annotation_file = "../dataset/annotations/grc_passport.json"

output_img_dir_train = "./passport_dataset/images/train"
output_img_dir_validate = "./passport_dataset/images/val"

output_label_dir_train = "./passport_dataset/labels/train"
output_label_dir_validate = "./passport_dataset/labels/val"

with open(annotation_file, "r") as f:
    annotation_data = json.load(f)

images = os.listdir(target_img_dir)
print(f"{len(images)} images loaded")

counter = 0
for image in images:
    image_path = target_img_dir + "/" + image
    
    # Safely handle missing files or incorrect paths
    if not os.path.exists(image_path):
        continue
        
    image_size = os.path.getsize(image_path)
    img = cv.imread(image_path)
    
    if img is None:
        continue
        
    [img_height, img_width] = img.shape[:2]
    metadata_key = image + str(image_size)
    
    # Skip if image metadata is missing in the JSON
    if metadata_key not in annotation_data["_via_img_metadata"]:
        continue
        
    regions = annotation_data["_via_img_metadata"][metadata_key]["regions"]

    photo_annotations = None
    signature_annotations = None
    mrz_annotations_line1 = None
    mrz_annotations_line2 = None

    # Dynamically locate regions based on field_name
    for region in regions:
        field_name = region.get("region_attributes", {}).get("field_name", "")
        
        if field_name == "photo":
            photo_annotations = normalize_annotations(region["shape_attributes"], img_width, img_height)
        elif field_name == "signature":
            signature_annotations = normalize_annotations(region["shape_attributes"], img_width, img_height)
        elif field_name == "mrz_line0":
            mrz_annotations_line1 = region["shape_attributes"]
        elif field_name == "mrz_line1":
            mrz_annotations_line2 = region["shape_attributes"]

    mrz_annotations = None
    if mrz_annotations_line1 and mrz_annotations_line2:
        mrz_annotations = get_mrz_annotations(mrz_annotations_line1, mrz_annotations_line2, img_width, img_height) 

    prefix = "/grc_passport_"
    
    if counter % 5 == 0:
        cv.imwrite(output_img_dir_validate + prefix + image, img)
        write_annotations(output_label_dir_validate + prefix + image[:-4] + ".txt", photo_annotations, mrz_annotations, signature_annotations)
    else:
        cv.imwrite(output_img_dir_train + prefix + image, img)
        write_annotations(output_label_dir_train + prefix + image[:-4] + ".txt", photo_annotations, mrz_annotations, signature_annotations)

    counter += 1