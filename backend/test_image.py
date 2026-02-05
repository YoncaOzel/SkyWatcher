"""
SkyWatcher - Tek Görüntü Test Script'i
Bir görüntüyü analiz ederek sonuçları gösterir.

Kullanım:
    python test_image.py <görüntü_yolu>
    python test_image.py ../dataset/night/clear/2021_10_07__00_53_10.jpg
    python test_image.py ../dataset/day/2021_10_07__12_00_43.jpg --day
"""

import sys
import os
import argparse
import cv2
import numpy as np

# Backend modüllerini import et
from sky_classifier import SkyClassifier, SkyAnalysisPipeline
from moon_masker import MoonMasker
from day_night_sorter import is_night, extract_datetime_from_filename
from config import OBSERVER_LATITUDE, OBSERVER_LONGITUDE, OBSERVER_ELEVATION


def test_night_image(image_path: str, show_details: bool = True):
    """Gece görüntüsünü test eder."""
    print("\n" + "=" * 60)
    print("🌙 GECE GÖRÜNTÜSÜ ANALİZİ")
    print("=" * 60)
    print(f"📁 Dosya: {image_path}")
    print("-" * 60)
    
    # Görüntüyü oku
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Görüntü okunamadı: {image_path}")
        return None
    
    # 1. Ay Tespiti
    print("\n🔍 Ay Tespiti...")
    moon_masker = MoonMasker()
    gray = moon_masker.to_grayscale(img)
    moon_result = moon_masker.detect_moon(gray)
    
    if moon_result.detected:
        print(f"   ✅ Ay tespit edildi!")
        print(f"   📍 Konum: ({moon_result.center_x}, {moon_result.center_y})")
        print(f"   📏 Yarıçap: {moon_result.radius} px")
        print(f"   🎯 Güven: {moon_result.confidence:.1f}%")
    else:
        print("   ❌ Ay tespit edilmedi")
    
    # 2. Gökyüzü Sınıflandırması
    print("\n📊 Gökyüzü Sınıflandırması...")
    pipeline = SkyAnalysisPipeline()
    result = pipeline.analyze_image(image_path)
    
    if result["success"]:
        r = result["result"]
        condition = r["sky_condition"]
        status = r["status"]
        
        # Emoji seç
        if condition == "CLEAR":
            emoji = "✨"
            color_desc = "Açık gökyüzü - Gözlem yapılabilir"
        elif condition == "NEUTRAL":
            emoji = "⛅"
            color_desc = "Parçalı bulutlu - Riskli"
        else:
            emoji = "☁️"
            color_desc = "Kapalı - Gözlem yapılamaz"
        
        print(f"   {emoji} Durum: {condition}")
        print(f"   📝 Açıklama: {color_desc}")
        print(f"   💡 Parlaklık: {r['mean_brightness']:.2f}")
        print(f"   ☁️ Bulut Oranı: %{r['cloud_coverage_percent']}")
        print(f"   🔭 Gözlem: {status}")
    else:
        print(f"   ❌ Hata: {result.get('error', 'Bilinmeyen hata')}")
    
    print("\n" + "=" * 60)
    return result


def test_day_image(image_path: str, show_details: bool = True):
    """Gündüz görüntüsünü test eder."""
    
    print("\n" + "=" * 60)
    print("☀️ GÜNDÜZ GÖRÜNTÜSÜ ANALİZİ")
    print("=" * 60)
    print(f"📁 Dosya: {image_path}")
    print("-" * 60)
    
    # Görüntüyü oku
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Görüntü okunamadı!")
        return None
    
    # Analiz
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    std_dev = np.std(gray)
    
    # BGR kanalları
    b, g, r = cv2.split(img)
    blue_ratio = np.mean(b) / (np.mean(r) + np.mean(g) + np.mean(b) + 1e-6)
    
    # Sınıflandırma (gündüz için)
    if blue_ratio > 0.36 and brightness < 220:
        condition = "CLEAR"
        emoji = "🔵"
        desc = "Açık mavi gökyüzü"
    elif brightness > 220 or blue_ratio < 0.32:
        condition = "CLOUDY"
        emoji = "☁️"
        desc = "Bulutlu gökyüzü"
    else:
        condition = "NEUTRAL"
        emoji = "⛅"
        desc = "Parçalı bulutlu"
    
    print(f"\n📊 Analiz Sonuçları:")
    print(f"   {emoji} Durum: {condition}")
    print(f"   📝 Açıklama: {desc}")
    print(f"   💡 Parlaklık: {brightness:.2f}")
    print(f"   🔵 Mavi Oranı: {blue_ratio:.4f}")
    print(f"   📈 Std Sapma: {std_dev:.2f}")
    
    print("\n" + "=" * 60)
    
    return {
        "condition": condition,
        "brightness": brightness,
        "blue_ratio": blue_ratio,
        "std_dev": std_dev
    }


def auto_detect_and_test(image_path: str):
    """Görüntü tipini otomatik tespit edip analiz eder."""
    filename = os.path.basename(image_path)
    
    # Dosya adından tarih çıkar
    dt = extract_datetime_from_filename(filename)
    
    if dt:
        # Gece mi gündüz mü kontrol et
        if is_night(dt):
            print(f"🕐 Zaman: {dt.strftime('%Y-%m-%d %H:%M:%S')} - GECE")
            return test_night_image(image_path)
        else:
            print(f"🕐 Zaman: {dt.strftime('%Y-%m-%d %H:%M:%S')} - GÜNDÜZ")
            return test_day_image(image_path)
    else:
        # Yol içinden tespit et
        if "night" in image_path.lower():
            return test_night_image(image_path)
        elif "day" in image_path.lower():
            return test_day_image(image_path)
        else:
            print("⚠️ Görüntü tipi tespit edilemedi. Gece olarak analiz ediliyor...")
            return test_night_image(image_path)


def main():
    parser = argparse.ArgumentParser(
        description="Tek bir görüntüyü analiz et"
    )
    parser.add_argument(
        "image_path",
        help="Analiz edilecek görüntü yolu"
    )
    parser.add_argument(
        "--day", "-d",
        action="store_true",
        help="Gündüz görüntüsü olarak analiz et"
    )
    parser.add_argument(
        "--night", "-n",
        action="store_true",
        help="Gece görüntüsü olarak analiz et"
    )
    parser.add_argument(
        "--auto", "-a",
        action="store_true",
        default=True,
        help="Otomatik tespit (varsayılan)"
    )
    
    args = parser.parse_args()
    
    # Dosya kontrolü
    if not os.path.exists(args.image_path):
        print(f"❌ Dosya bulunamadı: {args.image_path}")
        sys.exit(1)
    
    # Analiz
    if args.day:
        test_day_image(args.image_path)
    elif args.night:
        test_night_image(args.image_path)
    else:
        auto_detect_and_test(args.image_path)


if __name__ == "__main__":
    main()