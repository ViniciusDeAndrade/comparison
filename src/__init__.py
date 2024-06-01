from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from skimage.metrics import structural_similarity as compare_ssim

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = FastAPI()

def calculate_image_similarity(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1_gray = cv2.resize(img1_gray, (256, 256))
    img2_gray = cv2.resize(img2_gray, (256, 256))
    score, _ = compare_ssim(img1_gray, img2_gray, full=True)
    return score

def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

@app.post("/compare-images/")
async def compare_and_extract(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    img1 = Image.open(BytesIO(await file1.read()))
    img2 = Image.open(BytesIO(await file2.read()))

    img1_array = np.array(img1)
    img2_array = np.array(img2)

    similarity_score = calculate_image_similarity(img1_array, img2_array)
    text1 = extract_text_from_image(img1)
    text2 = extract_text_from_image(img2)

    return JSONResponse(content={
        "similarity_score": similarity_score,
        "text1": text1,
        "text2": text2
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
