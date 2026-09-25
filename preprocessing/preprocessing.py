import cv2
import os
from fg_segmentation import extract_foreground
from outliner import draw_mask_outline
from dewarping import dewarp_image

print(os.getcwd())

img_path = "../images/sp1.jpg"
 

fg_segment, pred_mask = extract_foreground(img_path)
fg_segment.save('fg.jpg')
contour = draw_mask_outline(fg_segment, pred_mask, thickness=4)

unwarped_image = dewarp_image(img_path, contour, output_path = "flattened.jpg")