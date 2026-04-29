import cv2
import numpy as np
import os
from PIL import Image, ExifTags

# =====================================
# KONFIGURASI GLOBAL
# =====================================
SCALE = 0.3

TOP_CROP = 0.03
BOTTOM_CROP = 0.97
LEFT_CROP = 0.05
RIGHT_CROP = 0.95

RASIO_CM_PER_PIXEL = 0.29

LOWER_GREEN = np.array([35, 60, 60])
UPPER_GREEN = np.array([85, 255, 255])


# =====================================
# FIX ORIENTASI FOTO (WA, DRIVE, DLL)
# =====================================
def fix_exif_rotation(img_path):
    pil_img = Image.open(img_path)
    try:
        exif = pil_img._getexif()
        if exif:
            for tag, value in exif.items():
                if ExifTags.TAGS.get(tag) == 'Orientation':
                    if value == 3:
                        pil_img = pil_img.rotate(180, expand=True)
                    elif value == 6:
                        pil_img = pil_img.rotate(270, expand=True)
                    elif value == 8:
                        pil_img = pil_img.rotate(90, expand=True)
    except:
        pass
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# =====================================
# FOTO DEPAN
# =====================================
def process_image(input_path, save_output=True, output_folder="output_hsv"):
    if save_output:
        os.makedirs(output_folder, exist_ok=True)

    img = fix_exif_rotation(input_path)
    if img is None:
        return {"success": False, "message": "Gambar tidak terbaca"}

    # STEP 1: RESIZE (PAKAI SCALE)
    height, width = img.shape[:2]
    img = cv2.resize(img, (int(width * SCALE), int(height * SCALE)))

    # STEP 2: CROP
    h, w = img.shape[:2]
    crop_img = img[
        int(h * TOP_CROP):int(h * BOTTOM_CROP),
        int(w * LEFT_CROP):int(w * RIGHT_CROP)
    ]

    # STEP 3: BLUR
    blur = cv2.GaussianBlur(crop_img, (5, 5), 0)

    # STEP 4: HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # STEP 5: MASK HIJAU
    green_mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    # STEP 6: BODY MASK
    body_mask = cv2.bitwise_not(green_mask)

    # STEP 7: MORPHOLOGY
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN, kernel_open)
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel_close)

    # STEP 8: CONTOUR
    contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"success": False, "message": "Kontur tidak ditemukan"}

    largest = max(contours, key=cv2.contourArea)

    clean_mask = np.zeros_like(body_mask)
    cv2.drawContours(clean_mask, [largest], -1, 255, -1)

    # RAPATKAN
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)
    clean_mask = cv2.medianBlur(clean_mask, 5)

    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.erode(clean_mask, kernel_erode, 1)

    # AUTO CROP
    x, y, w_box, h_box = cv2.boundingRect(largest)

    x1 = max(0, x - 20)
    y1 = max(0, y - 10)
    x2 = min(crop_img.shape[1], x + w_box + 20)
    y2 = min(crop_img.shape[0], y + h_box + 15)

    clean_mask = clean_mask[y1:y2, x1:x2]

    # STEP 9: UKUR
    ys, xs = np.where(clean_mask == 255)

    tinggi_pixel = np.max(ys) - np.min(ys)
    lebar_pixel = np.max(xs) - np.min(xs)

    tinggi_cm = tinggi_pixel * RASIO_CM_PER_PIXEL
    lebar_cm = lebar_pixel * RASIO_CM_PER_PIXEL

    return {
        "success": True,
        "tinggi_cm": round(tinggi_cm, 2),
        "lebar_cm": round(lebar_cm, 2)
    }


# =====================================
# FOTO SAMPING
# =====================================
def process_image_side(input_path, save_output=True, output_folder="output_side"):
    if save_output:
        os.makedirs(output_folder, exist_ok=True)

    img = fix_exif_rotation(input_path)
    if img is None:
        return {"success": False, "message": "Gambar tidak terbaca"}

    # RESIZE
    height, width = img.shape[:2]
    img = cv2.resize(img, (int(width * SCALE), int(height * SCALE)))

    # CROP
    h, w = img.shape[:2]
    crop_img = img[
        int(h * 0.08):int(h * 0.97),
        int(w * 0.15):int(w * 0.90)
    ]

    # BLUR + HSV
    blur = cv2.GaussianBlur(crop_img, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    body_mask = cv2.bitwise_not(green_mask)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"success": False, "message": "Kontur tidak ditemukan"}

    largest = max(contours, key=cv2.contourArea)

    clean_mask = np.zeros_like(body_mask)
    cv2.drawContours(clean_mask, [largest], -1, 255, -1)

    ys, xs = np.where(clean_mask == 255)

    tinggi_pixel = np.max(ys) - np.min(ys)
    tebal_pixel = np.max(xs) - np.min(xs)

    tinggi_cm = tinggi_pixel * RASIO_CM_PER_PIXEL
    tebal_cm = tebal_pixel * RASIO_CM_PER_PIXEL

    return {
        "success": True,
        "tinggi_cm": round(tinggi_cm, 2),
        "tebal_cm": round(tebal_cm, 2)
    }


# =====================================
# ESTIMASI BERAT
# =====================================
def estimate_weight(tinggi_cm, lebar_cm, tebal_cm):
    return 0.23 * tinggi_cm - 8.5
