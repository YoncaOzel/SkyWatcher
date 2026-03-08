# PRD — SkyWatcher ML Evaluation Module
**Proje:** SkyWatcher — Faz 4: Makine Öğrenmesi Değerlendirme Modülü  
**Versiyon:** 1.0  
**Tarih:** 08.03.2026  
**Amaç:** Üniversite ödevi gereksinimlerini karşılamak (Train/Test Split, ML Model, Metrikler, Görselleştirme)

---

## 1. Neden Bu Modül Gerekli?

Mevcut SkyWatcher sistemi kural tabanlı (threshold-based) çalışmaktadır. Üniversite ödevi aşağıdakileri zorunlu kılmaktadır:

| Gereksinim | Mevcut Durum | Bu Modül Sonrası |
|---|---|---|
| ML modeli (SVM/KNN/CNN vb.) | ❌ Yok | ✅ SVM + KNN |
| Train/Test split | ❌ Yok | ✅ 80/20 ayrımı |
| Accuracy / Precision / Recall / F1 | ❌ Yok | ✅ Hesaplanıyor |
| Confusion Matrix | ❌ Yok | ✅ Görselleştiriliyor |
| Karşılaştırmalı sonuçlar | ❌ Yok | ✅ SVM vs KNN vs Threshold |

---

## 2. Dosya Planı

Projeye yalnızca **1 yeni dosya** eklenecek:

```
backend/
└── evaluate.py   ← YENİ (bu PRD'nin konusu)
```

---

## 3. evaluate.py — Adım Adım İş Akışı

### Adım 1: Ground Truth Verisi Oluştur
- `auto_labeler.py` tarafından oluşturulan klasör yapısını oku:
  ```
  dataset/night/clear/     → label: 0
  dataset/night/neutral/   → label: 1
  dataset/night/cloudy/    → label: 2
  ```
- Her görüntü için **3 özellik (feature)** hesapla:
  1. `mean_brightness` — Ortalama piksel parlaklığı (zaten mevcut)
  2. `cloud_coverage_percent` — Bulut yüzdesi (zaten mevcut)
  3. `std_deviation` — Piksel standart sapması (yeni, 1 satır: `np.std()`)

### Adım 2: Train/Test Split
- Scikit-learn ile **%80 train / %20 test** ayırımı yap:
  ```python
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  ```

### Adım 3: İki Model Eğit ve Test Et
- **Model A:** SVM (Support Vector Machine)
  ```python
  from sklearn.svm import SVC
  ```
- **Model B:** KNN (K-Nearest Neighbors, k=5)
  ```python
  from sklearn.neighbors import KNeighborsClassifier
  ```

### Adım 4: Metrikleri Hesapla
Her iki model için şunları hesapla ve ekrana yazdır:
- Accuracy
- Precision (macro)
- Recall (macro)
- F1 Score (macro)
- Classification Report

### Adım 5: Görselleştirme
3 grafik kaydet (`/outputs/` klasörüne):
1. **Confusion Matrix** — SVM için (`confusion_matrix_svm.png`)
2. **Confusion Matrix** — KNN için (`confusion_matrix_knn.png`)
3. **Karşılaştırma Tablosu** — SVM vs KNN vs Threshold bar chart (`comparison.png`)

---

## 4. Kullanılacak Kütüphaneler

```
scikit-learn   (SVM, KNN, metrikler, confusion matrix)
matplotlib     (grafikler)
seaborn        (güzel confusion matrix görseli)
numpy          (zaten mevcut)
opencv-python  (zaten mevcut)
```

`requirements.txt`'e eklenecekler:
```
scikit-learn
matplotlib
seaborn
```

---

## 5. Beklenen Çıktılar

### Terminal Çıktısı (örnek):
```
=== SVM Results ===
Accuracy:  0.87
Precision: 0.85
Recall:    0.86
F1 Score:  0.85

=== KNN Results ===
Accuracy:  0.81
Precision: 0.79
Recall:    0.80
F1 Score:  0.79

=== Baseline (Threshold) ===
Accuracy:  0.74
```

### Kaydedilen Dosyalar:
```
outputs/
├── confusion_matrix_svm.png
├── confusion_matrix_knn.png
└── comparison.png
```

---

## 6. Ödev Raporu İçin Notlar

Bu modül tamamlandığında raporda şunları yazabilirsin:

- **Dataset:** TUG gözlemevi All-Sky görüntüleri, 3 sınıf (CLEAR/NEUTRAL/CLOUDY), `auto_labeler.py` ile etiketlenmiş
- **Method:** 3 özellik (mean brightness, cloud coverage, std deviation) kullanılarak SVM ve KNN modelleri eğitilmiş, kural tabanlı sistem ile karşılaştırılmıştır
- **Experiment:** %80 train / %20 test split; accuracy, precision, recall, F1 metrikleri hesaplanmıştır
- **Result:** SVM, KNN ve threshold yöntemlerinin confusion matrix karşılaştırması yapılmıştır

---

## 7. Uygulama Sırası (Copilot'a verilecek sıra)

1. `evaluate.py` dosyasını oluştur
2. `requirements.txt`'i güncelle
3. Çalıştır: `python backend/evaluate.py`
4. `outputs/` klasöründeki grafikleri raporda kullan
