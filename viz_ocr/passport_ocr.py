import math
import cv2
import numpy as np
import paddle
from paddleocr import PaddleOCR

paddle.set_flags({"FLAGS_enable_pir_api": False})
ocr = PaddleOCR(use_textline_orientation=True, lang="en", enable_mkldnn=False)

# Extended label keywords to match multi-language passport fields
FIELD_LABELS = {
    "surname": [
        "nom", "surname",
        "soyadı"
    ],
    "given_names": [
        "prénoms", "prenoms", "given names",
        "adı", "atasının adı", "given name", "patronymic"
    ],
    "nationality": [
        "nationalité", "nationality",
        "vətəndaşlığı"
    ],
    "date_of_birth": [
        "date de naissance", "date of birth",
        "doğulduğu tarix"
    ],
    "sex": [
        "sexe", "sex",
        "cinsi"
    ],
    "height": [
        "taille", "height"
    ],
    "place_of_birth": [
        "lieu de naissance", "place of birth",
        "doğulduğu yer"
    ],
    "date_of_issue": [
        "date de délivrance", "date of issue",
        "verilmə tarixi"
    ],
    "issuing_authority": [
        "autorité de délivrance", "issuing authority",
        "pasportu verən orqan"
    ],
    "date_of_expiry": [
        "date d'expiration", "date of expiry",
        "etibarlılıq müddəti"
    ],
    "passport_number": [
        "passeport n", "passport nr", "passport no",
        "pasportun nömrəsi"
    ],
    # New recommended key based on visual data
    "personal_number": [
        "fərdi identifikasiya nömrəsi", "personal no"
    ]
}


def get_box_center(bbox):
    x_coords = [p[0] for p in bbox]
    y_coords = [p[1] for p in bbox]
    return sum(x_coords) / 4.0, sum(y_coords) / 4.0


def extract_passport_fields_robust(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not open image at {image_path}")

    results = ocr.predict(img)
    if not results:
        return {}

    detections = []
    for res in results:
        rec_polys = (
            res.get("rec_polys", [])
            if isinstance(res, dict)
            else getattr(res, "rec_polys", [])
        )
        rec_texts = (
            res.get("rec_texts", [])
            if isinstance(res, dict)
            else getattr(res, "rec_texts", [])
        )
        rec_scores = (
            res.get("rec_scores", [])
            if isinstance(res, dict)
            else getattr(res, "rec_scores", [])
        )

        for bbox, text, score in zip(rec_polys, rec_texts, rec_scores):
            if score > 0.4:
                cx, cy = get_box_center(bbox)
                detections.append(
                    {
                        "text": str(text).strip(),
                        "bbox": bbox,
                        "center": (cx, cy),
                        "matched": False,
                    }
                )

    extracted_data = {}

    # Identify label anchor positions
    for field_key, keywords in FIELD_LABELS.items():
        label_det = None

        for det in detections:
            text_lower = det["text"].lower()
            if any(kw in text_lower for kw in keywords):
                label_det = det
                det["is_label"] = True
                break

        if label_det is None:
            continue

        lx, ly = label_det["center"]
        best_candidate = None
        min_distance = float("inf")

        # Find nearest non-label text block directly under or next to the label
        for det in detections:
            if det == label_det or det.get("matched", False):
                continue

            cx, cy = det["center"]

            # Field values are usually printed directly below the label in passports
            is_below = (
                (cy > ly) and (abs(cx - lx) < 80) and (10 < (cy - ly) < 90)
            )
            is_right = (cx > lx) and (abs(cy - ly) < 25) and ((cx - lx) < 250)

            if is_below or is_right:
                dist = math.hypot(cx - lx, cy - ly)
                if dist < min_distance:
                    min_distance = dist
                    best_candidate = det

        if best_candidate:
            extracted_data[field_key] = best_candidate["text"]
            best_candidate["matched"] = True

    return extracted_data


# Example:
# fields = extract_passport_fields_robust("../images/sp3.jpg")
# print(fields)