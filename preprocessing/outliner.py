import cv2
import numpy as np

def draw_mask_outline(original_pil_image, pred_mask, output_path="outlined.jpg", thickness=4, color=(0, 255, 0)):
    """Receives the PIL Image and prediction mask directly in memory to draw the outline.

    Returns None (instead of crashing) if no contour is found, so callers must check.
    """
    image_np = np.array(original_pil_image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    height, width = image_bgr.shape[:2]

    if hasattr(pred_mask, 'cpu'):
        pred_mask = pred_mask.cpu().numpy()

    pred_mask = np.squeeze(pred_mask)
    mask_resized = cv2.resize(pred_mask, (width, height))
    _, binary_mask = cv2.threshold((mask_resized * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found.")
        return None  # <-- FIX: no more UnboundLocalError

    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(image_bgr, [largest_contour], -1, color, thickness)
    cv2.imwrite(output_path, image_bgr)
    print(f"Outlined image saved to {output_path}")

    return largest_contour



