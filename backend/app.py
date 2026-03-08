"""
SkyWatcher - Flask Web API (app.py)
HTML dashboard için REST API sunucusu.

Kullanım:
    python backend/app.py
    veya
    cd backend && python app.py

Endpoints:
    GET  /                    → Dashboard HTML
    POST /api/analyze         → Görüntü analizi (multipart/form-data: file)
    GET  /outputs/<filename>  → ML grafikleri
"""

import os
import sys
import json
import tempfile
import logging
from pathlib import Path

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

# Backend modüllerini path'e ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sky_classifier import SkyAnalysisPipeline
from moon_masker import MoonMasker
from day_night_sorter import is_night, extract_datetime_from_filename
import cv2
import numpy as np

# =============================================================================
# UYGULAMA AYARLARI
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=FRONTEND_DIR)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB
CORS(app)


# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def analyze_day_image(img: np.ndarray) -> dict:
    """Gündüz görüntüsü analizi."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    std_dev = float(np.std(gray))

    b, g, r = cv2.split(img)
    blue_ratio = float(np.mean(b) / (np.mean(r) + np.mean(g) + np.mean(b) + 1e-6))

    if blue_ratio > 0.36 and brightness < 220:
        condition = "CLEAR"
        desc = "Açık mavi gökyüzü"
    elif brightness > 220 or blue_ratio < 0.32:
        condition = "CLOUDY"
        desc = "Bulutlu gökyüzü"
    else:
        condition = "NEUTRAL"
        desc = "Parçalı bulutlu"

    return {
        "time_of_day": "DAY",
        "sky_condition": condition,
        "description": desc,
        "mean_brightness": round(brightness, 2),
        "std_deviation": round(std_dev, 2),
        "blue_ratio": round(blue_ratio, 4),
        "cloud_coverage_percent": None,
        "moon_detected": False,
        "moon_center": None,
        "moon_radius": None,
        "moon_confidence": None,
        "status": "OBSERVABLE" if condition == "CLEAR" else ("RISKY" if condition == "NEUTRAL" else "NOT_OBSERVABLE"),
    }


def analyze_night_image(image_path: str, img: np.ndarray) -> dict:
    """Gece görüntüsü analizi (ay tespiti + sınıflandırma)."""
    # Ay tespiti
    moon_masker = MoonMasker()
    gray = moon_masker.to_grayscale(img)
    moon_result = moon_masker.detect_moon(gray)

    # Gökyüzü sınıflandırması
    pipeline = SkyAnalysisPipeline()
    result = pipeline.analyze_image(image_path)

    moon_info = {
        "moon_detected": moon_result.detected,
        "moon_center": (moon_result.center_x, moon_result.center_y) if moon_result.detected else None,
        "moon_radius": moon_result.radius,
        "moon_confidence": round(moon_result.confidence, 2) if moon_result.detected else None,
    }

    if result["success"]:
        r = result["result"]
        condition = r["sky_condition"]
        desc_map = {
            "CLEAR": "Açık gökyüzü - Gözlem yapılabilir",
            "NEUTRAL": "Parçalı bulutlu - Riskli",
            "CLOUDY": "Kapalı - Gözlem yapılamaz",
        }
        return {
            "time_of_day": "NIGHT",
            "sky_condition": condition,
            "description": desc_map.get(condition, condition),
            "mean_brightness": r["mean_brightness"],
            "std_deviation": None,
            "blue_ratio": None,
            "cloud_coverage_percent": r["cloud_coverage_percent"],
            "status": r["status"],
            **moon_info,
        }
    else:
        return {"error": result.get("error", "Analiz hatası"), **moon_info}


# =============================================================================
# ROTALAR
# =============================================================================

@app.route("/")
def index():
    """Dashboard HTML sayfasını serve et."""
    html_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(html_path):
        return send_file(html_path)
    return "index.html bulunamadı. Lütfen frontend/index.html dosyasını oluşturun.", 404


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    """ML grafik PNG dosyalarını serve et."""
    return send_from_directory(OUTPUTS_DIR, filename)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    Görüntü analizi endpoint'i.

    Request: multipart/form-data
        file: Görüntü dosyası (.jpg, .jpeg, .png, ...)

    Response: JSON
        { time_of_day, sky_condition, description, mean_brightness,
          cloud_coverage_percent, moon_detected, moon_center, moon_radius,
          moon_confidence, status, blue_ratio, std_deviation }
    """
    if "file" not in request.files:
        return jsonify({"error": "Dosya yüklenmedi. 'file' alanı gerekli."}), 400

    uploaded = request.files["file"]
    if not uploaded.filename:
        return jsonify({"error": "Dosya adı boş."}), 400

    ext = Path(uploaded.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Desteklenmeyen format: {ext}. İzin verilenler: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    # Geçici dosyaya kaydet
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp_path = tmp.name
        uploaded.save(tmp_path)

    try:
        img = cv2.imread(tmp_path)
        if img is None:
            return jsonify({"error": "Görüntü okunamadı. Lütfen geçerli bir görüntü dosyası yükleyin."}), 400

        filename = uploaded.filename
        dt = extract_datetime_from_filename(filename)

        if dt:
            night = is_night(dt)
        else:
            # Dosya yolundan tahmin et
            night = "night" in filename.lower()

        if night:
            result = analyze_night_image(tmp_path, img)
        else:
            result = analyze_day_image(img)

        result["filename"] = filename
        if dt:
            result["datetime"] = dt.strftime("%Y-%m-%d %H:%M:%S")

        logger.info(f"Analiz: {filename} -> {result.get('sky_condition')} ({result.get('time_of_day')})")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Analiz hatası ({uploaded.filename}): {e}", exc_info=True)
        return jsonify({"error": f"Sunucu hatası: {str(e)}"}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.route("/api/stats")
def api_stats():
    """Proje istatistikleri endpoint'i."""
    stats = {
        "dataset": {
            "night": {
                "clear": 796,
                "neutral": 9760,
                "cloudy": 3098,
                "total": 13654,
            }
        },
        "ml_results": {
            "svm": {"accuracy": 0.9711, "precision": 0.9821, "recall": 0.8641, "f1": 0.9067},
            "knn": {"accuracy": 0.9894, "precision": 0.9745, "recall": 0.9806, "f1": 0.9775},
            "baseline": {"accuracy": 0.9930},
        },
        "train_test_split": {"train": 10923, "test": 2731},
        "outputs": [f for f in os.listdir(OUTPUTS_DIR) if f.endswith(".png")] if os.path.exists(OUTPUTS_DIR) else [],
    }
    return jsonify(stats)


# =============================================================================
# SUNUCU BAŞLATMA
# =============================================================================

if __name__ == "__main__":
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(FRONTEND_DIR, exist_ok=True)

    print("=" * 60)
    print("  SkyWatcher - Web Dashboard")
    print("=" * 60)
    print(f"  Adres : http://localhost:5000")
    print(f"  Outputs: {OUTPUTS_DIR}")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5000, debug=False)
