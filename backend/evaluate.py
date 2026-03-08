"""
SkyWatcher - ML Değerlendirme Modülü (evaluate.py)
Faz 4: SVM ve KNN modelleri ile gökyüzü sınıflandırması değerlendirmesi.

Kullanım:
    python backend/evaluate.py

Çıktılar:
    - Terminal: Accuracy, Precision, Recall, F1, Classification Report
    - outputs/confusion_matrix_svm.png
    - outputs/confusion_matrix_knn.png
    - outputs/comparison.png
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path

# Backend klasörünü path'e ekle (config'e ulaşmak için)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from config import NIGHT_FOLDER, CLEAR_THRESHOLD, NEUTRAL_THRESHOLD

# =============================================================================
# SABITLER
# =============================================================================
CLASS_NAMES = ["CLEAR", "NEUTRAL", "CLOUDY"]
LABEL_MAP = {"clear": 0, "neutral": 1, "cloudy": 2}

# outputs/ klasörü: proje kökünde
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")


# =============================================================================
# ADIM 1: ÖZELLİK ÇIKARIMI
# =============================================================================

def extract_features(image_path: str):
    """
    Tek bir görüntüden 3 özellik çıkarır.

    Özellikler:
        1. mean_brightness  — Ortalama piksel parlaklığı (0–255)
        2. cloud_coverage   — Parlak piksel (>30) oranı (0–100)
        3. std_deviation    — Piksel standart sapması

    Returns:
        numpy array [3] veya None (okuma hatası)
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mean_brightness = float(np.mean(gray))
    cloud_coverage = float(np.sum(gray > 30) / gray.size * 100)
    std_deviation = float(np.std(gray))

    return np.array([mean_brightness, cloud_coverage, std_deviation])


def load_dataset(night_folder: str):
    """
    dataset/night/{clear,neutral,cloudy}/ klasörlerini tarayarak
    X (özellik matrisi) ve y (etiket vektörü) oluşturur.

    Labels:
        clear   → 0
        neutral → 1
        cloudy  → 2

    Returns:
        X: np.ndarray [N, 3]
        y: np.ndarray [N]
    """
    night_path = Path(night_folder)
    X, y = [], []
    skipped = 0

    print("Veri seti yükleniyor...")
    for label_name, label_id in LABEL_MAP.items():
        class_dir = night_path / label_name
        if not class_dir.exists():
            print(f"  [UYARI] Klasör bulunamadı: {class_dir}")
            continue

        # .jpg ve .png uzantılarını topla
        image_files = list(class_dir.glob("*.jpg")) + \
                      list(class_dir.glob("*.jpeg")) + \
                      list(class_dir.glob("*.png")) + \
                      list(class_dir.glob("*.JPG")) + \
                      list(class_dir.glob("*.PNG"))

        for img_path in image_files:
            features = extract_features(str(img_path))
            if features is None:
                skipped += 1
                continue
            X.append(features)
            y.append(label_id)

        print(f"  {label_name.upper():8s}: {len(image_files)} görüntü yüklendi")

    if skipped > 0:
        print(f"  [UYARI] {skipped} görüntü okunamadı ve atlandı.")

    return np.array(X), np.array(y)


# =============================================================================
# ADIM 2: KALİBRASYON TABANLI SINIFLANDIRICI (Baseline)
# =============================================================================

def threshold_predict(mean_brightness_values: np.ndarray) -> np.ndarray:
    """
    Mevcut kural tabanlı sınıflandırıcıyı (config eşik değerleri) simüle eder.

    CLEAR   < CLEAR_THRESHOLD  → label 0
    NEUTRAL < NEUTRAL_THRESHOLD → label 1
    CLOUDY  >= NEUTRAL_THRESHOLD → label 2

    Args:
        mean_brightness_values: X_test[:, 0] (mean_brightness sütunu)

    Returns:
        Tahmin edilen etiketler [N]
    """
    preds = []
    for b in mean_brightness_values:
        if b < CLEAR_THRESHOLD:
            preds.append(0)   # CLEAR
        elif b < NEUTRAL_THRESHOLD:
            preds.append(1)   # NEUTRAL
        else:
            preds.append(2)   # CLOUDY
    return np.array(preds)


# =============================================================================
# ADIM 3: METRİKLER
# =============================================================================

def print_model_results(model_name: str, y_true: np.ndarray, y_pred: np.ndarray):
    """Modelin performans metriklerini terminale yazdırır."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n{'='*45}")
    print(f"  {model_name} Sonuçları")
    print(f"{'='*45}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (macro)")
    print(f"  Recall    : {rec:.4f}  (macro)")
    print(f"  F1 Score  : {f1:.4f}  (macro)")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                zero_division=0))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# =============================================================================
# ADIM 4: GÖRSELLEŞTİRME
# =============================================================================

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          model_name: str, save_path: str):
    """
    Seaborn ile confusion matrix çizer ve kaydeder.

    Args:
        y_true:     Gerçek etiketler
        y_pred:     Tahmin edilen etiketler
        model_name: Başlıkta kullanılacak model adı
        save_path:  Kaydedilecek dosya yolu (.png)
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.5,
        ax=ax
    )

    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, pad=12)
    ax.set_xlabel("Tahmin Edilen (Predicted)", fontsize=11)
    ax.set_ylabel("Gerçek (True)", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Kaydedildi: {save_path}")


def plot_comparison(svm_metrics: dict, knn_metrics: dict, baseline_acc: float,
                    save_path: str):
    """
    SVM vs KNN vs Threshold karşılaştırma bar chart'ı çizer ve kaydeder.

    Her metrik grubu için 3 yan yana çubuk:
      - Accuracy, Precision, Recall, F1
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    svm_vals = [svm_metrics["accuracy"], svm_metrics["precision"],
                svm_metrics["recall"], svm_metrics["f1"]]
    knn_vals = [knn_metrics["accuracy"], knn_metrics["precision"],
                knn_metrics["recall"], knn_metrics["f1"]]
    # Threshold için sadece accuracy mevcut; diğerlerini NaN bırakıyoruz
    threshold_vals = [baseline_acc, None, None, None]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_svm = ax.bar(x - width, svm_vals, width, label="SVM (RBF)",
                      color="#2196F3", alpha=0.85)
    bars_knn = ax.bar(x, knn_vals, width, label="KNN (k=5)",
                      color="#4CAF50", alpha=0.85)

    # Threshold barını sadece Accuracy için çiz
    ax.bar(x[0] + width, baseline_acc, width, label="Threshold (Baseline)",
           color="#FF9800", alpha=0.85)

    # Değerleri çubukların üstüne yaz
    for bar in bars_svm:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_knn:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    # Threshold için ayrıca yaz
    ax.text(x[0] + width + width / 2, baseline_acc + 0.01,
            f"{baseline_acc:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Metrik", fontsize=12)
    ax.set_ylabel("Skor (0–1)", fontsize=12)
    ax.set_title("SVM vs KNN vs Threshold — Karşılaştırma", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Kaydedildi: {save_path}")


# =============================================================================
# ANA FONKSİYON
# =============================================================================

def main():
    # --- outputs/ klasörünü oluştur ---
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # --- Veri setini yükle ---
    X, y = load_dataset(NIGHT_FOLDER)

    if len(X) == 0:
        print("\n[HATA] Hiç görüntü yüklenemedi. "
              "dataset/night/clear|neutral|cloudy/ klasörlerini kontrol edin.")
        return

    print(f"\nToplam örnek   : {len(X)}")
    for label_name, label_id in LABEL_MAP.items():
        count = np.sum(y == label_id)
        print(f"  {label_name.upper():8s}: {count} örnek")

    # --- Train/Test Split (%80 / %20, stratified) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain seti     : {len(X_train)} örnek")
    print(f"Test seti      : {len(X_test)} örnek")

    # --- Feature Scaling ---
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # --- Model A: SVM ---
    print("\nSVM modeli eğitiliyor...")
    svm_model = SVC(kernel="rbf", random_state=42)
    svm_model.fit(X_train_sc, y_train)
    y_pred_svm = svm_model.predict(X_test_sc)
    svm_metrics = print_model_results("SVM (RBF Kernel)", y_test, y_pred_svm)

    # --- Model B: KNN ---
    print("\nKNN modeli eğitiliyor...")
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_sc, y_train)
    y_pred_knn = knn_model.predict(X_test_sc)
    knn_metrics = print_model_results("KNN (k=5)", y_test, y_pred_knn)

    # --- Baseline: Threshold ---
    y_pred_threshold = threshold_predict(X_test[:, 0])
    baseline_acc = accuracy_score(y_test, y_pred_threshold)
    print(f"\n{'='*45}")
    print(f"  Baseline (Threshold) Sonuçları")
    print(f"{'='*45}")
    print(f"  Accuracy  : {baseline_acc:.4f}  (sadece mean_brightness eşiği)")

    # --- Görselleştirmeler ---
    print(f"\nGrafikler kaydediliyor -> {OUTPUTS_DIR}")
    plot_confusion_matrix(
        y_test, y_pred_svm, "SVM (RBF Kernel)",
        os.path.join(OUTPUTS_DIR, "confusion_matrix_svm.png")
    )
    plot_confusion_matrix(
        y_test, y_pred_knn, "KNN (k=5)",
        os.path.join(OUTPUTS_DIR, "confusion_matrix_knn.png")
    )
    plot_comparison(
        svm_metrics, knn_metrics, baseline_acc,
        os.path.join(OUTPUTS_DIR, "comparison.png")
    )

    print("\n✅ Değerlendirme tamamlandı!")
    print(f"   Tüm grafikler: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
