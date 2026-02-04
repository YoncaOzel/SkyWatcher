# Product Requirements Document (PRD)

**Proje Adı:** SkyWatcher - Teleskop Görüntü Analiz Sistemi  
**Versiyon:** 1.1 (Updated: Day/Night Separation Logic Added)  
**Tarih:** 04.02.2026  
**Hazırlayan:** Zeynep Yonca Özel (Backend Dev & Lead)  
**Durum:** Final Taslak

---

## 1. Proje Özeti (Executive Summary)
**SkyWatcher**, All-Sky (Tüm Gökyüzü) kameralarından alınan görüntüleri analiz ederek gökyüzünün astronomik gözlem yapmaya uygun olup olmadığını belirleyen bir görüntü işleme sistemidir. 

Sistem, önce dosya ismindeki tarih/saat verisine bakarak gece-gündüz ayrımı yapar, ardından gece görüntülerini işleyerek **Ay ışığını maskeler** ve bulut yoğunluğunu hesaplar.

## 2. Problem Tanımı
* **Sorun:** Mevcut sistemler parlak nesneleri (Ay, Güneş) ayırt edemediği için, açık ama Ay'ın olduğu geceleri "Bulutlu" olarak işaretlemektedir (False Positive).
* **Çözüm:** Fiziksel (Tarih/Saat) ve Görüntüsel (Maskeleme) filtreler kullanarak hibrit bir doğrulama mekanizması kurmak.

## 3. Hedef Kitle ve Kullanım Senaryosu
* **Kullanıcı:** Astronomlar, TUG (Tübitak Ulusal Gözlemevi) Operatörleri.
* **Senaryo:** Sistem klasöre düşen her yeni fotoğrafı analiz eder ve şu çıktıyı üretir:
    ```json
    {
      "filename": "img_20260204_2300.jpg",
      "status": "OBSERVABLE",
      "sky_condition": "CLEAR",
      "moon_present": true,
      "cloud_coverage_percent": 12
    }
    ```

## 4. Teknik Gereksinimler (Technical Requirements)
* **Dil:** Python 3.9+
* **Kütüphaneler:** * `ephem` (Güneş/Ay konumu ve gece-gündüz hesabı için)
    * `opencv-python` (Görüntü işleme ve maskeleme için)
    * `numpy` (Matris işlemleri için)
* **Girdi:** `.jpg` formatında, dosya isminde tarih barındıran (Örn: `YYYYMMDD_HHMMSS`) görseller.

## 5. Fonksiyonel Gereksinimler (İş Akışı)

Proje 3 aşamalı bir "Pipeline" (Boru Hattı) mimarisinde çalışacaktır.

### 5.1. Adım 0: Zamansal Ayrıştırma (Temporal Filtering)
* **Amaç:** Gündüz çekilen masmavi gökyüzü fotoğraflarını, "karanlık arayan" gece algoritmamıza sokmamak.
* **Yöntem:**
    1.  Dosya isminden tarih ve saat regex ile çekilir.
    2.  `ephem` kütüphanesi ile Gözlemevi koordinatlarında Güneş'in yüksekliği hesaplanır.
* **Kural:**
    * `Güneş Açısı > -12°` (Nautical Twilight ve üzeri) → **GÜNDÜZ/ALACAKARANLIK** (Klasör: `/dataset/day`) - **İşlem Durdurulur.**
    * `Güneş Açısı < -12°` → **GECE** (Klasör: `/dataset/night`) - **Bir sonraki adıma geçilir.**

### 5.2. Adım 1: Ay Maskeleme (Moon Masking)
* **Amaç:** Gece fotoğrafında Ay varsa, parlaklığının bulut sanılmasını engellemek.
* **Algoritma:**
    1.  Görüntü gri tonlamaya (Grayscale) çevrilir.
    2.  `cv2.HoughCircles` ile görüntüdeki en parlak ve keskin daire (Ay) aranır.
    3.  **Bulunursa:** Dairenin koordinatlarına **siyah (0,0,0)** bir daire çizilerek Ay yok edilir.
    4.  **Bulunamazsa:** İşleme devam edilir (Ay yok veya bulut arkasında).

### 5.3. Adım 2: Sınıflandırma ve Karar (Decision Logic)
* **Amaç:** Ay silindikten sonra kalan görüntüdeki "gerçek" parlaklığı (bulutları) ölçmek.
* **Algoritma:**
    1.  Maskelenmiş görüntünün **Ortalama Piksel Parlaklığı (Mean Intensity)** hesaplanır.
    2.  Önceden belirlenen eşik değerlerine (Thresholds) göre karar verilir:

| Parlaklık Değeri (0-255) | Etiket | Karar |
| :--- | :--- | :--- |
| **< 20** | **CLEAR** (Açık) | GÖZLEM YAPILABİLİR |
| **20 - 45** | **NEUTRAL** (Parçalı) | RİSKLİ / GÖZLEM YAPILABİLİR |
| **> 45** | **CLOUDY** (Kapalı) | GÖZLEMLENEMEZ |

*(Not: Eşik değerleri, gerçek verilerle yapılacak denemelerden sonra kalibre edilecektir.)*

## 6. Veri Seti Stratejisi
1.  **Toplama:** Ham veriler `ham_veriler` klasörüne atılır.
2.  **Script 1 (Splitter):** `day_night_sorter.py` çalıştırılır -> Gece ve Gündüz klasörlerine ayrılır.
3.  **Script 2 (Sorter):** `auto_labeler.py` çalıştırılır -> Gece klasöründeki resimler Clear/Cloudy/Neutral olarak alt klasörlere ayrılır.
4.  **Manuel Kontrol:** Geliştirici, klasörleri hızlıca gözden geçirip hataları düzeltir.

## 7. Başarı Kriterleri
* **False Positive (Yanlış Alarm):** Açık havanın "Bulutlu" sanılması %5'in altında olmalı.
* **Hız:** 1000 adet fotoğrafın gece/gündüz ayrımı 10 saniyenin altında yapılmalı.
* **Güvenilirlik:** Dosya isminde tarih formatı bozuk olsa bile script çökmemeli (Error Handling).

## 8. Yol Haritası (Implementation Roadmap)
* [ ] **Faz 1:** Dosya isminden tarih okuyan ve gece/gündüz ayıran scriptin yazılması. (`day_night_sorter.py`)
* [ ] **Faz 2:** Ay'ı tespit edip maskeleyen (siyah daire çizen) OpenCV fonksiyonunun yazılması.
* [ ] **Faz 3:** Parlaklık ölçerek JSON çıktısı veren ana uygulamanın birleştirilmesi.