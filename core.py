import cv2
import numpy as np
import os

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


def process_image(input_path, save_output=True, output_folder="output_hsv"):
    """
    Proses foto depan:
    output utama -> tinggi dan lebar
    """
    if save_output:
        os.makedirs(output_folder, exist_ok=True)

    # =====================================
    # BACA GAMBAR
    # =====================================
    img = cv2.imread(input_path)
    if img is None:
        return {
            "success": False,
            "message": f"Gambar tidak ditemukan: {input_path}"
        }

    # =====================================
    # STEP 1: RESIZE
    # =====================================
    TARGET_WIDTH = 480
    height, width = img.shape[:2]
    new_height = int(height * TARGET_WIDTH / width)
    img = cv2.resize(img, (TARGET_WIDTH, new_height))

    # =====================================
    # STEP 2: CROP
    # =====================================
    h, w = img.shape[:2]
    top_crop = int(h * TOP_CROP)
    bottom_crop = int(h * BOTTOM_CROP)
    left_crop = int(w * LEFT_CROP)
    right_crop = int(w * RIGHT_CROP)

    crop_img = img[top_crop:bottom_crop, left_crop:right_crop]

    # =====================================
    # STEP 3: BLUR
    # =====================================
    blur = cv2.GaussianBlur(crop_img, (5, 5), 0)

    # =====================================
    # STEP 4: HSV
    # =====================================
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # =====================================
    # STEP 5: MASK BACKGROUND HIJAU
    # =====================================
    green_mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    # =====================================
    # STEP 6: BALIK JADI BODY MASK
    # =====================================
    body_mask = cv2.bitwise_not(green_mask)

    # =====================================
    # STEP 7: MORPHOLOGY
    # =====================================
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN, kernel_open)
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel_close)

    # =====================================
    # STEP 8: AMBIL KONTUR TERBESAR
    # =====================================
    contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = crop_img.copy()
    clean_mask = np.zeros_like(body_mask)

    if not contours:
        return {
            "success": False,
            "message": "Kontur tubuh tidak ditemukan."
        }

    largest_contour = max(contours, key=cv2.contourArea)

    cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # rapikan lagi
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_smooth)
    clean_mask = cv2.medianBlur(clean_mask, 5)
    # SHRINK MASK (biar tidak terlalu gemuk)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.erode(clean_mask, kernel_erode, iterations=1)

    contours_final, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours_final:
        return {
            "success": False,
            "message": "Kontur final tidak ditemukan."
        }

    largest_final = max(contours_final, key=cv2.contourArea)

    # =====================================
    # AUTO-CROP FOKUS KE TUBUH DEPAN
    # =====================================
    x, y, w_box, h_box = cv2.boundingRect(largest_final)

    margin_top = 10
    margin_bottom = 15
    margin_x = 20

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_top)
    x2 = min(crop_img.shape[1], x + w_box + margin_x)
    y2 = min(crop_img.shape[0], y + h_box + margin_bottom)

    # crop ulang gambar dan mask
    crop_img = crop_img[y1:y2, x1:x2]
    clean_mask = clean_mask[y1:y2, x1:x2]

    # result juga harus diambil dari crop yang baru
    result = crop_img.copy()

    # cari ulang kontur setelah auto-crop
    contours_final_crop, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours_final_crop:
        return {
            "success": False,
            "message": "Kontur final setelah auto-crop tidak ditemukan."
        }

    largest_final = max(contours_final_crop, key=cv2.contourArea)

    # gambar kontur asli
    cv2.drawContours(result, [largest_final], -1, (0, 255, 0), 2)

    # =====================================
    # STEP 9: UKUR TITIK EKSTREM
    # =====================================
    ys, xs = np.where(clean_mask == 255)

    top_y = np.min(ys)
    bottom_y = np.max(ys)
    left_x = np.min(xs)
    right_x = np.max(xs)

    tinggi_pixel = bottom_y - top_y
    lebar_pixel = right_x - left_x

    tinggi_cm = tinggi_pixel * RASIO_CM_PER_PIXEL
    lebar_cm = lebar_pixel * RASIO_CM_PER_PIXEL

    # visualisasi
    cv2.rectangle(result, (left_x, top_y), (right_x, bottom_y), (255, 0, 0), 2)

    center_x = (left_x + right_x) // 2
    cv2.line(result, (center_x, top_y), (center_x, bottom_y), (0, 0, 255), 2)

    cv2.putText(result, f"Tinggi: {round(tinggi_cm, 2)} cm", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(result, f"Lebar: {round(lebar_cm, 2)} cm", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # =====================================
    # STEP 10: SIMPAN OUTPUT
    # =====================================
    if save_output:
        cv2.imwrite(os.path.join(output_folder, "1_resize_crop.jpg"), crop_img)
        cv2.imwrite(os.path.join(output_folder, "2_blur.jpg"), blur)
        cv2.imwrite(os.path.join(output_folder, "3_green_mask.jpg"), green_mask)
        cv2.imwrite(os.path.join(output_folder, "4_body_mask.jpg"), body_mask)
        cv2.imwrite(os.path.join(output_folder, "5_clean_mask.jpg"), clean_mask)
        cv2.imwrite(os.path.join(output_folder, "6_result.jpg"), result)

    return {
        "success": True,
        "message": "Proses foto depan berhasil",
        "tinggi_pixel": int(tinggi_pixel),
        "tinggi_cm": round(tinggi_cm, 2),
        "lebar_pixel": int(lebar_pixel),
        "lebar_cm": round(lebar_cm, 2),
        "crop_img": crop_img,
        "green_mask": green_mask,
        "body_mask": body_mask,
        "clean_mask": clean_mask,
        "result_img": result
    }


def process_image_side(input_path, save_output=True, output_folder="output_side"):
    """
    Proses foto samping:
    output utama -> tinggi dan tebal
    """
    if save_output:
        os.makedirs(output_folder, exist_ok=True)

    # =====================================
    # BACA GAMBAR
    # =====================================
    img = cv2.imread(input_path)
    if img is None:
        return {
            "success": False,
            "message": f"Gambar tidak ditemukan: {input_path}"
        }

    # =====================================
    # STEP 1: RESIZE
    # =====================================
    TARGET_WIDTH = 480
    height, width = img.shape[:2]
    new_height = int(height * TARGET_WIDTH / width)
    img = cv2.resize(img, (TARGET_WIDTH, new_height))

    # =====================================
    # STEP 2: CROP
    # =====================================
    h, w = img.shape[:2]
    top_crop = int(h * 0.08)
    bottom_crop = int(h * 0.97)
    left_crop = int(w * 0.15)
    right_crop = int(w * 0.90)

    crop_img = img[top_crop:bottom_crop, left_crop:right_crop]

    # =====================================
    # STEP 3: BLUR
    # =====================================
    blur = cv2.GaussianBlur(crop_img, (5, 5), 0)

    # =====================================
    # STEP 4: HSV
    # =====================================
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # =====================================
    # STEP 5: MASK BACKGROUND HIJAU
    # =====================================
    green_mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    # =====================================
    # STEP 6: BALIK JADI BODY MASK
    # =====================================
    body_mask = cv2.bitwise_not(green_mask)

    # =====================================
    # STEP 7: MORPHOLOGY
    # =====================================
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN, kernel_open)
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel_close)

    # =====================================
    # STEP 8: AMBIL KONTUR TERBESAR
    # =====================================
    contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = crop_img.copy()
    clean_mask = np.zeros_like(body_mask)

    if not contours:
        return {
            "success": False,
            "message": "Kontur tubuh samping tidak ditemukan."
        }

    largest_contour = max(contours, key=cv2.contourArea)

    cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_smooth)
    clean_mask = cv2.medianBlur(clean_mask, 5)

    contours_final, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours_final:
        return {
            "success": False,
            "message": "Kontur final samping tidak ditemukan."
        }

    largest_final = max(contours_final, key=cv2.contourArea)

    # gambar kontur asli
    cv2.drawContours(result, [largest_final], -1, (0, 255, 0), 2)

    # =====================================
    # AUTO-CROP FOKUS KE TUBUH SAMPING
    # =====================================
    x, y, w_box, h_box = cv2.boundingRect(largest_final)

    # tambahkan margin biar badan tidak mepet
    margin_top = 10
    margin_bottom = 20
    margin_x = 10

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_top)
    x2 = min(clean_mask.shape[1], x + w_box + margin_x)
    y2 = min(clean_mask.shape[0], y + h_box + margin_bottom)

    # crop ulang mask dan result
    focused_mask = clean_mask[y1:y2, x1:x2]
    focused_result = result[y1:y2, x1:x2]

    #   ukur ulang dari mask yang sudah fokus
    ys, xs = np.where(focused_mask == 255)

    top_y = np.min(ys)
    bottom_y = np.max(ys)
    left_x = np.min(xs)
    right_x = np.max(xs)

    tinggi_pixel = bottom_y - top_y
    tebal_pixel = right_x - left_x

    tinggi_cm = tinggi_pixel * RASIO_CM_PER_PIXEL
    tebal_cm = tebal_pixel * RASIO_CM_PER_PIXEL

    # gambar kotak pengukuran di hasil fokus
    cv2.rectangle(focused_result, (left_x, top_y), (right_x, bottom_y), (255, 0, 0), 2)

    center_x = (left_x + right_x) // 2
    cv2.line(focused_result, (center_x, top_y), (center_x, bottom_y), (0, 0, 255), 2)

    cv2.putText(focused_result, f"Tinggi: {round(tinggi_cm, 2)} cm", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(focused_result, f"Tebal: {round(tebal_cm, 2)} cm", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # timpa result_img supaya yang ditampilkan adalah hasil fokus
    result = focused_result
    clean_mask = focused_mask

    # =====================================
    # STEP 9: UKUR TITIK EKSTREM
    # =====================================
    ys, xs = np.where(clean_mask == 255)

    top_y = np.min(ys)
    bottom_y = np.max(ys)
    left_x = np.min(xs)
    right_x = np.max(xs)

    tinggi_pixel = bottom_y - top_y
    tebal_pixel = right_x - left_x

    tinggi_cm = tinggi_pixel * RASIO_CM_PER_PIXEL
    tebal_cm = tebal_pixel * RASIO_CM_PER_PIXEL

    # visualisasi
    cv2.rectangle(result, (left_x, top_y), (right_x, bottom_y), (255, 0, 0), 2)

    center_x = (left_x + right_x) // 2
    cv2.line(result, (center_x, top_y), (center_x, bottom_y), (0, 0, 255), 2)

    cv2.putText(result, f"Tinggi: {round(tinggi_cm, 2)} cm", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(result, f"Tebal: {round(tebal_cm, 2)} cm", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # =====================================
    # STEP 10: SIMPAN OUTPUT
    # =====================================
    if save_output:
        cv2.imwrite(os.path.join(output_folder, "1_resize_crop.jpg"), crop_img)
        cv2.imwrite(os.path.join(output_folder, "2_blur.jpg"), blur)
        cv2.imwrite(os.path.join(output_folder, "3_green_mask.jpg"), green_mask)
        cv2.imwrite(os.path.join(output_folder, "4_body_mask.jpg"), body_mask)
        cv2.imwrite(os.path.join(output_folder, "5_clean_mask.jpg"), clean_mask)
        cv2.imwrite(os.path.join(output_folder, "6_result.jpg"), result)

    return {
        "success": True,
        "message": "Proses foto samping berhasil",
        "tinggi_pixel": int(tinggi_pixel),
        "tinggi_cm": round(tinggi_cm, 2),
        "tebal_pixel": int(tebal_pixel),
        "tebal_cm": round(tebal_cm, 2),
        "crop_img": crop_img,
        "green_mask": green_mask,
        "body_mask": body_mask,
        "clean_mask": clean_mask,
        "result_img": result
    }


def estimate_weight(tinggi_cm, lebar_cm, tebal_cm):
    # model regresi sederhana
    berat = 0.23 * tinggi_cm - 8.5
    return berat
