from fastapi import FastAPI, Form, UploadFile, File
from antropometri import AntropometriSK
from datetime import datetime
from core import process_image, process_image_side, estimate_weight
import shutil
import os

app = FastAPI()

sk = AntropometriSK("sk_antropometri.xlsx")


def hitung_umur_bulan(tanggal_lahir_str):
    try:
        lahir = datetime.strptime(tanggal_lahir_str, "%d-%m-%Y")
        sekarang = datetime.now()

        umur_bulan = (sekarang.year - lahir.year) * 12 + (sekarang.month - lahir.month)

        if sekarang.day < lahir.day:
            umur_bulan -= 1

        return umur_bulan
    except ValueError:
        return None


def gabung_status(status_bbtb, status_tbu):
    if status_bbtb == "Normal" and status_tbu == "Normal":
        return "Kondisi Normal"
    elif status_bbtb in ["Kurus", "Sangat Kurus"] and status_tbu == "Normal":
        return "Perlu Perhatian Gizi"
    elif status_bbtb == "Normal" and status_tbu in ["Pendek", "Sangat Pendek"]:
        return "Perlu Perhatian Pertumbuhan"
    elif status_bbtb in ["Kurus", "Sangat Kurus"] and status_tbu in ["Pendek", "Sangat Pendek"]:
        return "Perlu Perhatian Gizi dan Pertumbuhan"
    elif status_bbtb == "Gemuk" and status_tbu == "Normal":
        return "Berat Badan Berlebih"
    elif status_bbtb == "Gemuk" and status_tbu in ["Pendek", "Sangat Pendek"]:
        return "Berat Badan Berlebih dan Pertumbuhan Tidak Optimal"
    else:
        return "Perlu Pemantauan Lanjutan"


@app.get("/")
def root():
    return {"message": "API status gizi aktif"}


@app.post("/predict-manual")
def predict_manual(
    jenis_kelamin: str = Form(...),
    tanggal_lahir: str = Form(...),
    tinggi: float = Form(...),
    berat: float = Form(...)
):
    umur_bulan = hitung_umur_bulan(tanggal_lahir)
    if umur_bulan is None:
        return {"error": "Format tanggal harus DD-MM-YYYY"}

    status_bbtb = sk.classify_bbtb(jenis_kelamin, tinggi, berat)
    status_tbu = sk.classify_tbu(jenis_kelamin, umur_bulan, tinggi)
    kesimpulan = gabung_status(status_bbtb, status_tbu)

    return {
        "jenis_kelamin": jenis_kelamin,
        "tanggal_lahir": tanggal_lahir,
        "umur_bulan": umur_bulan,
        "tinggi": tinggi,
        "berat": berat,
        "status_bbtb": status_bbtb,
        "status_tbu": status_tbu,
        "kesimpulan": kesimpulan
    }

@app.post("/predict-photo")
async def predict_photo(
    jenis_kelamin: str = Form(...),
    tanggal_lahir: str = Form(...),
    foto_depan: UploadFile = File(...),
    foto_samping: UploadFile = File(...)
):
    os.makedirs("temp", exist_ok=True)

    path_depan = f"temp/{foto_depan.filename}"
    path_samping = f"temp/{foto_samping.filename}"

    with open(path_depan, "wb") as buffer:
        shutil.copyfileobj(foto_depan.file, buffer)

    with open(path_samping, "wb") as buffer:
        shutil.copyfileobj(foto_samping.file, buffer)

    umur_bulan = hitung_umur_bulan(tanggal_lahir)
    if umur_bulan is None:
        return {"error": "Format tanggal harus DD-MM-YYYY"}

    depan = process_image(path_depan)
    samping = process_image_side(path_samping)

    if not depan["success"] or not samping["success"]:
        return {
            "error": "Gagal memproses gambar",
            "depan": depan.get("message", ""),
            "samping": samping.get("message", "")
        }

    tinggi = depan["tinggi_cm"]
    lebar = depan["lebar_cm"]
    tebal = samping["tebal_cm"]

    if tinggi < 40 or tinggi > 150:
        return {
            "error": "Deteksi foto tidak valid",
            "message": f"Tinggi terdeteksi tidak masuk akal: {tinggi} cm"
        }

    berat = estimate_weight(tinggi, lebar, tebal)

    # validasi berat agar tidak negatif / tidak masuk akal
    if berat <= 0:
        return {
            "error": "Estimasi berat tidak valid",
            "message": f"Berat hasil estimasi tidak masuk akal: {berat} kg"
        }

    status_bbtb = sk.classify_bbtb(jenis_kelamin, tinggi, berat)
    status_tbu = sk.classify_tbu(jenis_kelamin, umur_bulan, tinggi)
    kesimpulan = gabung_status(status_bbtb, status_tbu)

    return {
        "jenis_kelamin": jenis_kelamin,
        "tanggal_lahir": tanggal_lahir,
        "umur_bulan": umur_bulan,
        "tinggi": tinggi,
        "lebar": lebar,
        "tebal": tebal,
        "berat": berat,
        "status_bbtb": status_bbtb,
        "status_tbu": status_tbu,
        "kesimpulan": kesimpulan
    }
