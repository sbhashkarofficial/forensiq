import cv2

def preprocess_mrz(cropped_mrz):
    #cropped_mrz is a numpy array(output of extract_regions)
    mrz_preprocessed = cv2.cvtColor(cropped_mrz, cv2.COLOR_BGR2GRAY)
    mrz_preprocessed = cv2.medianBlur(mrz_preprocessed, 2)
    # _, mrz_preprocessed = cv2.threshold(mrz_preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mrz_preprocessed