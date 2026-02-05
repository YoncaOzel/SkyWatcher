"""
SkyWatcher - Gündüz Görüntüleri Otomatik Etiketleme Script'i (day_auto_labeler.py)
Day klasöründeki görüntüleri CLEAR/NEUTRAL/CLOUDY alt klasörlerine ayırır.

Gündüz görüntüleri için farklı eşik değerleri kullanılır çünkü
gündüz gökyüzü parlaklığı gece gökyüzünden çok farklıdır.

Kullanım:
    python day_auto_labeler.py
    python day_auto_labeler.py --input ../dataset/day --dry-run
    python day_auto_labeler.py --clear-threshold 180 --neutral-threshold 220
"""

import os
import shutil
import argparse
import logging
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from config import DAY_FOLDER

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Gündüz görüntüleri için eşik değerleri
# Gündüz gökyüzü çok daha parlak olduğundan farklı değerler kullanılır
# CLEAR: Açık mavi gökyüzü (yüksek parlaklık, düşük varyans)
# CLOUDY: Bulutlu gökyüzü (beyaz/gri tonlar, yüksek parlaklık ama farklı desen)
# NEUTRAL: Ara durumlar

DAY_CLEAR_THRESHOLD = 180      # Bu değerin üstü = CLEAR (açık mavi gökyüzü)
DAY_NEUTRAL_THRESHOLD = 220    # Bu değerin üstü = CLOUDY (beyaz/bulutlu)


class DaySkyClassifier:
    """Gündüz gökyüzü görüntülerini sınıflandıran sınıf."""
    
    def __init__(self, clear_threshold: float = DAY_CLEAR_THRESHOLD, 
                 neutral_threshold: float = DAY_NEUTRAL_THRESHOLD):
        """
        Args:
            clear_threshold: CLEAR/NEUTRAL ayırım eşiği
            neutral_threshold: NEUTRAL/CLOUDY ayırım eşiği
        """
        self.clear_threshold = clear_threshold
        self.neutral_threshold = neutral_threshold
    
    def analyze_image(self, image_path: str) -> Dict:
        """
        Gündüz görüntüsünü analiz eder.
        
        Gündüz sınıflandırması için kullanılan metrikler:
        - Ortalama parlaklık (brightness)
        - Mavi kanal yoğunluğu (blue_intensity)
        - Standart sapma (variance) - bulut tespiti için
        
        Returns:
            Analiz sonuçları
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {"success": False, "error": f"Görüntü okunamadı: {image_path}"}
            
            # Grayscale için parlaklık
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            std_dev = np.std(gray)
            
            # BGR kanallarını ayır
            b, g, r = cv2.split(img)
            blue_ratio = np.mean(b) / (np.mean(r) + np.mean(g) + np.mean(b) + 1e-6)
            
            # Sınıflandırma mantığı
            # Gündüz için: Mavi gökyüzü yüksek mavi oranı ve orta parlaklık
            # Bulutlu: Yüksek parlaklık, düşük mavi oranı (beyazımsı)
            
            # Mavi oranı düşükse ve parlaklık yüksekse -> CLOUDY (beyaz bulutlar)
            # Mavi oranı yüksekse -> CLEAR (mavi gökyüzü)
            # Ara değerler -> NEUTRAL
            
            if blue_ratio > 0.36 and brightness < self.neutral_threshold:
                # Mavi baskın ve çok parlak değil -> CLEAR
                condition = "CLEAR"
            elif brightness > self.neutral_threshold or blue_ratio < 0.32:
                # Çok parlak veya mavi az -> CLOUDY  
                condition = "CLOUDY"
            else:
                condition = "NEUTRAL"
            
            return {
                "success": True,
                "result": {
                    "sky_condition": condition,
                    "brightness": round(brightness, 2),
                    "blue_ratio": round(blue_ratio, 4),
                    "std_dev": round(std_dev, 2)
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


def auto_label_day_images(
    input_folder: str,
    output_base: str = None,
    copy_files: bool = True,
    dry_run: bool = False,
    clear_threshold: float = DAY_CLEAR_THRESHOLD,
    neutral_threshold: float = DAY_NEUTRAL_THRESHOLD
) -> Dict:
    """
    Gündüz görüntülerini sınıflandırarak alt klasörlere ayırır.
    
    Args:
        input_folder: Giriş klasörü (dataset/day)
        output_base: Çıkış ana klasörü (None ise input_folder kullanılır)
        copy_files: True = kopyala, False = taşı
        dry_run: True ise sadece simülasyon yapar
        clear_threshold: CLEAR eşik değeri
        neutral_threshold: NEUTRAL eşik değeri
        
    Returns:
        İşlem özeti
    """
    input_path = Path(input_folder)
    output_base = Path(output_base) if output_base else input_path
    
    if not input_path.exists():
        logger.error(f"Klasör bulunamadı: {input_folder}")
        return {"success": False}
    
    # Alt klasörleri oluştur
    clear_folder = output_base / "clear"
    neutral_folder = output_base / "neutral"
    cloudy_folder = output_base / "cloudy"
    
    if not dry_run:
        clear_folder.mkdir(exist_ok=True)
        neutral_folder.mkdir(exist_ok=True)
        cloudy_folder.mkdir(exist_ok=True)
    
    # Classifier
    classifier = DaySkyClassifier(
        clear_threshold=clear_threshold,
        neutral_threshold=neutral_threshold
    )
    
    # Görüntüleri bul
    image_files = []
    for ext in [".jpg", ".jpeg", ".png"]:
        image_files.extend(input_path.glob(f"*{ext}"))
    
    total = len(image_files)
    logger.info(f"Toplam {total} gündüz görüntüsü etiketlenecek...")
    
    stats = {"CLEAR": 0, "NEUTRAL": 0, "CLOUDY": 0, "errors": 0}
    transfer_func = shutil.copy2 if copy_files else shutil.move
    action = "Kopyalanacak" if copy_files else "Taşınacak"
    
    if dry_run:
        logger.info("DRY RUN - Dosyalar taşınmayacak")
    
    for i, img_path in enumerate(image_files, 1):
        if i % 100 == 0 or i == total:
            logger.info(f"İşleniyor: {i}/{total}")
        
        try:
            result = classifier.analyze_image(str(img_path))
            
            if not result["success"]:
                stats["errors"] += 1
                continue
            
            condition = result["result"]["sky_condition"]
            stats[condition] += 1
            
            # Hedef klasör
            if condition == "CLEAR":
                dest_folder = clear_folder
            elif condition == "NEUTRAL":
                dest_folder = neutral_folder
            else:
                dest_folder = cloudy_folder
            
            dest_path = dest_folder / img_path.name
            
            if not dry_run:
                transfer_func(str(img_path), str(dest_path))
                
        except Exception as e:
            logger.error(f"Hata ({img_path.name}): {e}")
            stats["errors"] += 1
    
    # Sonuç özeti
    logger.info("=" * 50)
    logger.info("GÜNDÜZ GÖRÜNTÜ ETİKETLEME TAMAMLANDI")
    logger.info("=" * 50)
    
    total_classified = stats["CLEAR"] + stats["NEUTRAL"] + stats["CLOUDY"]
    
    if total_classified > 0:
        logger.info(f"CLEAR  : {stats['CLEAR']:5d} görüntü ({100*stats['CLEAR']/total_classified:.1f}%)")
        logger.info(f"NEUTRAL: {stats['NEUTRAL']:5d} görüntü ({100*stats['NEUTRAL']/total_classified:.1f}%)")
        logger.info(f"CLOUDY : {stats['CLOUDY']:5d} görüntü ({100*stats['CLOUDY']/total_classified:.1f}%)")
    
    if stats["errors"] > 0:
        logger.warning(f"Hatalar : {stats['errors']} görüntü işlenemedi")
    
    logger.info("=" * 50)
    
    return {
        "success": True,
        "stats": stats,
        "total_processed": total_classified
    }


def main():
    parser = argparse.ArgumentParser(
        description="Gündüz görüntülerini CLEAR/NEUTRAL/CLOUDY olarak etiketle"
    )
    parser.add_argument(
        "--input", "-i",
        default=DAY_FOLDER,
        help="Giriş klasörü (varsayılan: dataset/day)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Çıkış klasörü (varsayılan: giriş klasörü)"
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Dosyaları kopyalamak yerine taşı"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simülasyon modu - dosyaları taşıma"
    )
    parser.add_argument(
        "--clear-threshold",
        type=float,
        default=DAY_CLEAR_THRESHOLD,
        help=f"CLEAR eşik değeri (varsayılan: {DAY_CLEAR_THRESHOLD})"
    )
    parser.add_argument(
        "--neutral-threshold",
        type=float,
        default=DAY_NEUTRAL_THRESHOLD,
        help=f"NEUTRAL eşik değeri (varsayılan: {DAY_NEUTRAL_THRESHOLD})"
    )
    
    args = parser.parse_args()
    
    auto_label_day_images(
        input_folder=args.input,
        output_base=args.output,
        copy_files=not args.move,
        dry_run=args.dry_run,
        clear_threshold=args.clear_threshold,
        neutral_threshold=args.neutral_threshold
    )


if __name__ == "__main__":
    main()