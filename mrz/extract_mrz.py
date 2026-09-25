import pytesseract
# Load image via OpenCV (returns a NumPy array)

# Run OCR on the processed NumPy array
def extract_mrz(mrz_preprocessed):
    #mrz_preprocessed is a numpy array (output of preprocess_mrz)
    custom_config = r'--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
    mrz_text = pytesseract.image_to_string(mrz_preprocessed, config=custom_config)
    print(f"mrz text extracted :: length of which is :: {len(mrz_text)}")
    return mrz_text.strip()
