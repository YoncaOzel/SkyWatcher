# SkyWatcher

**Automated Sky Condition Classification System for Astronomical Observatories**

SkyWatcher is a computer-vision pipeline that automatically classifies sky-camera images captured at the TÜBİTAK National Observatory (TUG) in Bakırtepe, Turkey. It determines whether astronomical observation conditions are **CLEAR**, **NEUTRAL**, or **CLOUDY** by combining astronomical ephemeris calculations, moon masking, brightness-based classification, and machine learning validation — all exposed through a Flask REST API and a dark-themed single-page dashboard.

---

## Table of Contents

- [Features](#features)
- [Pipeline Overview](#pipeline-overview)
- [ML Performance](#ml-performance)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [CLI](#cli)
  - [REST API](#rest-api)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)

---

## Features

- **Astronomical Day/Night Sorting** — Uses PyEphem to compute real sun altitude for each image's timestamp at TUG coordinates (36.8245°N, 30.3353°E, 2500 m). Images with sun altitude < −12° (astronomical twilight) are classified as nighttime.
- **Moon Detection & Masking** — Detects the Moon in grayscale night images using OpenCV's `HoughCircles` algorithm with brightness validation. Applies a padded black-circle mask to prevent moonlight from corrupting cloud classification.
- **Sky Condition Classification** — Classifies masked night images by mean pixel brightness. Separate blue-channel ratio logic handles daytime images. Returns one of three conditions: `CLEAR`, `NEUTRAL`, or `CLOUDY`.
- **ML Evaluation** — Trains and evaluates **SVM (RBF kernel)** and **KNN (k=5)** classifiers on a 13,654-image labeled dataset, achieving up to **98.94% accuracy**. Generates confusion matrices and model comparison charts.
- **REST API** — Flask server with CORS support. Accepts image uploads and returns full analysis results as JSON.
- **Interactive Dashboard** — Single-file SPA with a dark space theme, drag-and-drop image upload, live classification results, animated ML metric bars, and interactive confusion matrix visualization.
- **Batch Processing** — Auto-label entire folders of raw images; sort raw telescope output into structured dataset directories in one command.

---

## Pipeline Overview

```
raw_dataset/                    Phase 1 — Day/Night Sorting
└── YYYY-MM-DD/                 day_night_sorter.py
    └── YYYY_MM_DD__HH_MM_SS.jpg  ↓  ephem sun altitude < −12° → night
                                ↓
dataset/night/  ←──────────────┘    dataset/day/
                                ↓
                            Phase 2 — Moon Masking
                            moon_masker.py
                            HoughCircles + brightness validation
                                ↓
                            masked image (Moon blacked out)
                                ↓
                            Phase 3 — Sky Classification
                            sky_classifier.py
                            mean brightness thresholds on masked image
                                ↓
                        CLEAR / NEUTRAL / CLOUDY
                        OBSERVABLE / RISKY / NOT_OBSERVABLE
                                ↓
                            Phase 4 — ML Evaluation
                            evaluate.py
                            SVM (RBF) · KNN (k=5) · Baseline threshold
                                ↓
                        outputs/confusion_matrix_*.png
                        outputs/comparison.png
```

### Classification Thresholds (Night Images, 0–255 scale)

| Mean Brightness | Sky Condition | Observation Status |
|-----------------|---------------|--------------------|
| 0 – 19          | CLEAR         | OBSERVABLE         |
| 20 – 44         | NEUTRAL       | RISKY              |
| ≥ 45            | CLOUDY        | NOT_OBSERVABLE     |

### Daytime Classification (Blue-Channel Ratio)

| Condition | Logic |
|-----------|-------|
| CLEAR     | `blue_ratio > 0.36` AND `brightness < 220` |
| CLOUDY    | `brightness > 220` OR `blue_ratio < 0.32` |
| NEUTRAL   | Otherwise |

---

## ML Performance

Trained on 13,654 labeled night images (80/20 stratified split). Features: mean brightness, cloud coverage percentage, standard deviation.

| Model              | Accuracy  | Precision | Recall    | F1        |
|--------------------|-----------|-----------|-----------|-----------|
| SVM (RBF kernel)   | 97.11%    | 98.21%    | 86.41%    | 90.67%    |
| KNN (k=5)          | **98.94%**| 97.45%    | **98.06%**| **97.75%**|
| Baseline Threshold | 99.30%    | —         | —         | —         |

Training set: 10,923 samples · Test set: 2,731 samples

---

## Project Structure

```
SkyWatcher/
├── backend/
│   ├── app.py               # Flask REST API server
│   ├── auto_labeler.py      # Batch night image auto-labeler
│   ├── config.py            # Observatory constants & thresholds
│   ├── day_auto_labeler.py  # Batch day image auto-labeler
│   ├── day_night_sorter.py  # Phase 1 — astronomical day/night sort
│   ├── evaluate.py          # Phase 4 — SVM/KNN ML evaluation
│   ├── main.py              # Unified CLI entry point
│   ├── moon_masker.py       # Phase 2 — moon detection & masking
│   ├── sky_classifier.py    # Phase 3 — cloud classification
│   ├── test_image.py        # Single-image debug utility
│   └── requirements.txt
├── frontend/
│   └── index.html           # Single-page dashboard
├── dataset/
│   ├── day/
│   │   ├── clear/
│   │   ├── cloudy/
│   │   └── neutral/
│   ├── night/
│   │   ├── clear/           # 796 images
│   │   ├── cloudy/          # 3,098 images
│   │   └── neutral/         # 9,760 images
│   └── night_masked/
├── raw_dataset/
│   └── YYYY-MM-DD/          # Date-organized raw telescope images
└── outputs/
    ├── confusion_matrix_svm.png
    ├── confusion_matrix_knn.png
    └── comparison.png
```

---

## Installation

**Requirements:** Python 3.9+

```bash
# Clone the repository
git clone https://github.com/your-username/SkyWatcher.git
cd SkyWatcher

# Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

---

## Usage

### CLI

The `main.py` CLI provides three sub-commands:

#### Analyze — classify a single image or an entire folder

```bash
# Analyze a single image
python backend/main.py analyze --input path/to/image.jpg

# Analyze all images in a folder and export results to JSON
python backend/main.py analyze --input dataset/night/ --output results.json
```

#### Sort — sort raw telescope images into day/night folders

```bash
# Move images (default)
python backend/main.py sort --input raw_dataset/

# Copy images instead of moving
python backend/main.py sort --input raw_dataset/ --copy
```

#### Label — auto-label sorted images into clear/neutral/cloudy subfolders

```bash
# Copy and label night images
python backend/main.py label --input dataset/night/

# Dry run (no files are moved)
python backend/main.py label --input dataset/night/ --dry-run

# Move instead of copy
python backend/main.py label --input dataset/night/ --move
```

#### Run ML Evaluation

```bash
python backend/evaluate.py
# Outputs: outputs/confusion_matrix_svm.png
#          outputs/confusion_matrix_knn.png
#          outputs/comparison.png
```

#### Test a Single Image (Debug)

```bash
# Auto-detect day/night from filename
python backend/test_image.py path/to/image.jpg

# Force day or night mode
python backend/test_image.py path/to/image.jpg --day
python backend/test_image.py path/to/image.jpg --night
```

### REST API

Start the Flask development server:

```bash
python backend/app.py
# Server runs on http://localhost:5000
```

Open your browser at `http://localhost:5000` to access the interactive dashboard.

---

## API Reference

### `POST /api/analyze`

Analyzes an uploaded sky image and returns full classification results.

**Request:** `multipart/form-data` with a `file` field containing the image (`.jpg`, `.jpeg`, or `.png`). Maximum size: 32 MB.

```bash
curl -X POST http://localhost:5000/api/analyze \
     -F "file=@2021_10_18__16_30_30.jpg"
```

**Response:**

```json
{
  "filename": "2021_10_18__16_30_30.jpg",
  "datetime": "2021-10-18 16:30:30",
  "time_of_day": "NIGHT",
  "sky_condition": "CLEAR",
  "description": "Clear sky — observation recommended.",
  "mean_brightness": 12.34,
  "std_deviation": 5.67,
  "cloud_coverage_percent": 8,
  "cloud_coverage_raw": 7.9,
  "moon_detected": true,
  "moon_center": [320, 240],
  "moon_radius": 35,
  "moon_confidence": 0.92,
  "status": "OBSERVABLE"
}
```

> For daytime images, the response additionally includes a `blue_ratio` field and omits moon-related fields.

**Sky condition values:** `CLEAR` · `NEUTRAL` · `CLOUDY`  
**Status values:** `OBSERVABLE` · `RISKY` · `NOT_OBSERVABLE`  
**Time of day values:** `DAY` · `NIGHT`

---

### `GET /api/stats`

Returns dataset statistics and ML evaluation results.

```bash
curl http://localhost:5000/api/stats
```

**Response:**

```json
{
  "dataset": {
    "night": { "clear": 796, "neutral": 9760, "cloudy": 3098, "total": 13654 }
  },
  "ml_results": {
    "svm":      { "accuracy": 0.9711, "precision": 0.9821, "recall": 0.8641, "f1": 0.9067 },
    "knn":      { "accuracy": 0.9894, "precision": 0.9745, "recall": 0.9806, "f1": 0.9775 },
    "baseline": { "accuracy": 0.9930 }
  },
  "train_test_split": { "train": 10923, "test": 2731 },
  "outputs": ["comparison.png", "confusion_matrix_knn.png", "confusion_matrix_svm.png"]
}
```

---

### `GET /outputs/<filename>`

Serves ML chart images (confusion matrices, model comparison chart).

```
GET /outputs/confusion_matrix_svm.png
GET /outputs/confusion_matrix_knn.png
GET /outputs/comparison.png
```

---

## Configuration

All observatory constants and classification thresholds are defined in [`backend/config.py`](backend/config.py):

| Constant | Default | Description |
|----------|---------|-------------|
| `OBSERVER_LATITUDE` | `"36.8245"` | TUG Bakırtepe latitude (°N) |
| `OBSERVER_LONGITUDE` | `"30.3353"` | TUG Bakırtepe longitude (°E) |
| `OBSERVER_ELEVATION` | `2500` | Observatory elevation (metres) |
| `SUN_ALTITUDE_THRESHOLD` | `-12.0` | Astronomical twilight boundary (degrees) |
| `CLEAR_THRESHOLD` | `20` | Brightness ≤ this → CLEAR |
| `NEUTRAL_THRESHOLD` | `45` | Brightness ≤ this → NEUTRAL, otherwise CLOUDY |
| `RAW_DATA_FOLDER` | `raw_dataset/` | Input folder for raw images |
| `DAY_FOLDER` | `dataset/day/` | Output folder for daytime images |
| `NIGHT_FOLDER` | `dataset/night/` | Output folder for nighttime images |
| `SUPPORTED_EXTENSIONS` | `.jpg .jpeg .png` | Accepted image formats |

---

## Dataset

Raw images were captured at TUG Bakırtepe Observatory over **20 nights** (2021-09-27 to 2021-10-19) at approximately 44–60 second intervals.

**Filename format:** `YYYY_MM_DD__HH_MM_SS[_1_1].jpg`  
(e.g. `2021_10_18__16_30_30_1_1.jpg`)

| Split | Label   | Count  |
|-------|---------|--------|
| Night | CLEAR   | 796    |
| Night | NEUTRAL | 9,760  |
| Night | CLOUDY  | 3,098  |
| **Total** |     | **13,654** |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.9+ |
| Astronomical calculations | [PyEphem](https://rhodesmill.org/pyephem/) |
| Image processing | [OpenCV](https://opencv.org/) |
| Numerical computing | [NumPy](https://numpy.org/) |
| Machine learning | [scikit-learn](https://scikit-learn.org/) (SVM, KNN, StandardScaler) |
| Visualization | [Matplotlib](https://matplotlib.org/) · [Seaborn](https://seaborn.pydata.org/) |
| REST API | [Flask](https://flask.palletsprojects.com/) · [Flask-CORS](https://flask-cors.readthedocs.io/) |
| Frontend | Vanilla HTML/CSS/JS · [Chart.js](https://www.chartjs.org/) |

---

## Observatory

**TÜBİTAK National Observatory (TUG)**  
Bakırtepe, Antalya, Turkey  
Coordinates: 36.8245°N, 30.3353°E · Elevation: 2,500 m
