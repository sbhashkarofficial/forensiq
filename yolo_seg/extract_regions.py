def extract_regions(class_id, results):
   #results is the output produced by yolo_predict
   img = results.orig_img
   boxes = results[0].boxes.data.cpu().numpy()

   for box in boxes:
    x1, y1, x2, y2, conf, cls_id = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    cropped = img[y1:y2, x1:x2]

    if int(cls_id) == class_id:
      return cropped
