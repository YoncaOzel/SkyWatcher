# SkyWatcher - Görev Listesi (Task List)

**Proje:** SkyWatcher - Teleskop Görüntü Analiz Sistemi  
**Oluşturulma Tarihi:** 04.02.2026

---

## Faz 1: Gece/Gündüz Ayrıştırma Modülü (`day_night_sorter.py`)

### 1.1 Temel Altyapı
- [ ] Proje klasör yapısını oluştur (`ham_veriler/`, `dataset/day/`, `dataset/night/`)
- [ ] `requirements.txt` dosyası oluştur (`ephem`, `opencv-python`, `numpy`)
- [ ] Sabit değerleri tanımla (Gözlemevi koordinatları, Güneş açısı eşiği: -12°)

### 1.2 Tarih/Saat Çıkarma Fonksiyonu
- [ ] Dosya isminden tarih çıkaran regex fonksiyonu yaz (`YYYYMMDD_HHMMSS` formatı)
- [ ] Farklı tarih formatlarını destekle (opsiyonel)
- [ ] Hatalı/eksik tarih formatı için error handling ekle

### 1.3 Güneş Pozisyonu Hesaplama
- [ ] `ephem` kütüphanesi ile Güneş açısı hesaplama fonksiyonu yaz
- [ ] TUG koordinatlarını varsayılan olarak ayarla (36.8245° N, 30.3353° E)
- [ ] `is_night()` fonksiyonu oluştur (Güneş açısı < -12° kontrolü)

### 1.4 Dosya Sıralama Mekanizması
- [ ] Klasördeki tüm `.jpg` dosyalarını tarayan fonksiyon yaz
- [ ] Gece/Gündüz klasörlerine dosya taşıma/kopyalama işlevi
- [ ] İşlem özeti (log) yazdır (kaç dosya gece, kaç dosya gündüz)

### 1.5 Performans ve Test
- [ ] 1000 fotoğraf için < 10 saniye hedefini test et
- [ ] Birim testleri yaz (farklı tarih formatları, edge case'ler)

---

## Faz 2: Ay Maskeleme Modülü

### 2.1 Görüntü Ön İşleme
- [ ] Görüntüyü grayscale'e çeviren fonksiyon yaz
- [ ] Gürültü azaltma için Gaussian blur uygula (opsiyonel)

### 2.2 Ay Tespiti
- [ ] `cv2.HoughCircles` parametrelerini ayarla
- [ ] En parlak ve keskin daireyi tespit eden fonksiyon yaz
- [ ] Ay tespit edilip edilmediğini döndür (boolean + koordinatlar)

### 2.3 Maskeleme İşlemi
- [ ] Tespit edilen Ay'ı siyah daire ile maskeleyen fonksiyon yaz
- [ ] Maske çapını Ay çapından biraz büyük tut (halo etkisi için)
- [ ] Maskelenmiş görüntüyü döndür

### 2.4 Test ve Kalibrasyon
- [ ] Farklı Ay fazlarında (dolunay, hilal) test et
- [ ] Parametreleri gerçek verilerle kalibre et

---

## Faz 3: Sınıflandırma ve Ana Uygulama

### 3.1 Parlaklık Analizi
- [ ] Ortalama piksel parlaklığı (Mean Intensity) hesaplayan fonksiyon yaz
- [ ] Eşik değerlerini tanımla (CLEAR: <20, NEUTRAL: 20-45, CLOUDY: >45)
- [ ] Bulut yüzdesini hesaplayan algoritma geliştir

### 3.2 JSON Çıktı Üretimi
- [ ] Çıktı şablonunu oluştur (filename, status, sky_condition, moon_present, cloud_coverage_percent)
- [ ] JSON dosyasına yazma fonksiyonu
- [ ] Toplu işlem için JSON array desteği

### 3.3 Pipeline Entegrasyonu
- [ ] Tüm modülleri birleştiren ana `main.py` dosyası oluştur
- [ ] Komut satırı argümanları ekle (input klasörü, output dosyası)
- [ ] Hata yönetimi ve logging mekanizması

### 3.4 `auto_labeler.py` Script'i
- [ ] Gece klasöründeki resimleri analiz eden script yaz
- [ ] Clear/Cloudy/Neutral alt klasörlerine otomatik taşıma
- [ ] Manuel kontrol için özet rapor oluştur

---

## Bonus: Dokümantasyon ve DevOps

- [ ] README.md dosyası oluştur (kurulum, kullanım)
- [ ] Kod içi docstring'leri yaz
- [ ] Örnek veri seti ile demo hazırla
- [ ] GitHub Actions ile CI/CD kur (opsiyonel)

---

## Başarı Kriterleri Kontrol Listesi

| Kriter | Hedef | Durum |
|--------|-------|-------|
| False Positive Oranı | < %5 | ⏳ |
| 1000 Fotoğraf İşleme Süresi | < 10 saniye | ⏳ |
| Error Handling | Bozuk tarih formatında çökmeme | ⏳ |

---

**Öncelik Sırası:** Faz 1 → Faz 2 → Faz 3 → Bonus