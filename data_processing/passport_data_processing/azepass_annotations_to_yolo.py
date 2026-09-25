import os
import json
import numpy as np
import cv2 as cv

print(os.getcwd())

Class_mapping = {"main_photo":0, "mrz_zone":1 , "signature":2}

def normalize_annotations(shat):
    [x,y, w, h] = [shat["x"], shat["y"], shat["width"], shat["height"]]
    return [(x+w/2)/img_width, (y+h/2)/img_height, w/img_width, h/img_height]

def get_mrz_annotations(mrz1, mrz2):
    [x1,y1, w1, h1] = [mrz1["x"], mrz1["y"], mrz1["width"], mrz1["height"]]
    [x2,y2, w2, h2] = [mrz2["x"], mrz2["y"], mrz2["width"], mrz2["height"]]
    [x, y] = [min(x1, x2), min(y1, y2)]
    [w, h] = [max(x1 + w1, x2 + w2)-x, max(y1 + h1, y2 + h2) - y]
    return [(x+w/2)/img_width, (y+h/2)/img_height, w/img_width, h/img_height]

def stringify_annotations(class_id, annotation):
    [x, y, w, h] = annotation
    return str(class_id) + " " + str(x)+" "+str(y)+" "+str(w)+" "+str(h)+"\n"

def write_annotations(filename):
    with open(filename, "w") as f:
        f.write(stringify_annotations(Class_mapping["main_photo"],photo_annotations))
        f.write(stringify_annotations(Class_mapping["mrz_zone"],mrz_annotations))
        f.write(stringify_annotations(Class_mapping["signature"],signature_annotations))

target_img_dir = "../dataset/images/aze_passport"
annotation_file = "../dataset/annotations/aze_passport.json"

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
    image_size = os.path.getsize(image_path)

    img = cv.imread(image_path)
    [img_height, img_width] = img.shape[:2]

    annotation = annotation_data["_via_img_metadata"][image + str(image_size)]["regions"]

    photo_annotations = normalize_annotations(annotation[-3]["shape_attributes"])
    signature_annotations = normalize_annotations(annotation[-4]["shape_attributes"])

    mrz_annotations_line1 = annotation[-6]["shape_attributes"]
    mrz_annotations_line2 = annotation[-5]["shape_attributes"]
    mrz_annotations = get_mrz_annotations(mrz_annotations_line1, mrz_annotations_line2) 

    if counter%5==0:
        cv.imwrite(output_img_dir_validate + "/aze_passport" + image, img)
        write_annotations(output_label_dir_validate + "/aze_passport" + image[:-4] + ".txt")
    else:
        cv.imwrite(output_img_dir_train + "/aze_passport" + image, img)
        write_annotations(output_label_dir_train + "/aze_passport" + image[:-4] + ".txt")

    counter+=1
