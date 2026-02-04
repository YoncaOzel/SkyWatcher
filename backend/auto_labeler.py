"""
SkyWatcher - Otomatik Etiketleme Script'i (auto_labeler.py)
Gece klasöründeki görüntüleri CLEAR/NEUTRAL/CLOUDY alt klasörlerine ayırır.

Kullanım:
    python auto_labeler.py
    python auto_labeler.py --input ../dataset/night --dry-run
"""

import os
import shutil
import argparse
import logging
from pathlib import Path
from typing import Dict

from config import NIGHT_FOLDER, CLEAR_THRESHOLD, NEUTRAL_THRESHOLD
from sky_classifier import SkyAnalysisPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def auto_label_images(
    input_folder: str,
    output_base: str = None,
    copy_files: bool = True,
    dry_run: bool = False
) -> Dict:
    """
    Görüntüleri sınıflandırarak alt klasörlere ayırır.
    
    Args:
        input_folder: Giriş klasörü (dataset/night)
        output_base: Çıkış ana klasörü (None ise input_folder kullanılır)
        copy_files: True = kopyala, False = taşı
        dry_run: True ise sadece simülasyon yapar
        
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
    
    # Pipeline
    pipeline = SkyAnalysisPipeline()
    
    # Görüntüleri bul
    image_files = []
    for ext in [".jpg", ".jpeg", ".png"]:
        image_files.extend(input_path.glob(f"*{ext}"))
    
    total = len(image_files)
    logger.info(f"Toplam {total} görüntü etiketlenecek...")
    
    stats = {"CLEAR": 0, "NEUTRAL": 0, "CLOUDY": 0, "errors": 0}
    transfer_func = shutil.copy2 if copy_files else shutil.move
    action = "Kopyalanacak" if copy_files else "Taşınacak"
    
    if dry_run:
        logger.info("DRY RUN - Dosyalar taşınmayacak")
    
    for i, img_path in enumerate(image_files, 1):
        if i % 100 == 0 or i == total:
            logger.info(f"İşleniyor: {i}/{total}")
        
        try:
            result = pipeline.analyze_image(str(img_path))
            
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
                if not dest_path.exists():
                    transfer_func(str(img_path), str(dest_path))
                    
        except Exception as e:
            logger.error(f"Hata ({img_path.name}): {e}")
            stats["errors"] += 1
    
    summary = {
        "success": True,
        "total": total,
        "clear": stats["CLEAR"],
        "neutral": stats["NEUTRAL"],
        "cloudy": stats["CLOUDY"],
        "errors": stats["errors"],
        "dry_run": dry_run
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='SkyWatcher - Otomatik Etiketleyici'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=NIGHT_FOLDER,
        help=f'Giriş klasörü (varsayılan: {NIGHT_FOLDER})'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Çıkış klasörü (varsayılan: giriş klasörü)'
    )
    
    parser.add_argument(
        '--move',
        action='store_true',
        help='Dosyaları kopyalamak yerine taşı'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simülasyon modu (dosya taşımaz)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  SkyWatcher - Otomatik Etiketleyici")
    print("=" * 60)
    print(f"  Giriş Klasörü : {args.input}")
    print(f"  İşlem         : {'Taşı' if args.move else 'Kopyala'}")
    print(f"  Dry Run       : {'Evet' if args.dry_run else 'Hayır'}")
    print("=" * 60)
    
    summary = auto_label_images(
        input_folder=args.input,
        output_base=args.output,
        copy_files=not args.move,
        dry_run=args.dry_run
    )
    
    print(f"\n{'=' * 60}")
    print("  SONUÇ ÖZETİ")
    print("=" * 60)
    print(f"  Toplam       : {summary['total']}")
    print(f"  ☀️  CLEAR     : {summary['clear']}")
    print(f"  🌤️  NEUTRAL   : {summary['neutral']}")
    print(f"  ☁️  CLOUDY    : {summary['cloudy']}")
    print(f"  ⚠️  Hatalar   : {summary['errors']}")
    print("=" * 60)
    
    if not args.dry_run:
        print(f"\nKlasörler:")
        print(f"  {args.input}/clear/")
        print(f"  {args.input}/neutral/")
        print(f"  {args.input}/cloudy/")


if __name__ == "__main__":
    main()