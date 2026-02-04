"""
SkyWatcher - Gece/Gündüz Ayrıştırma Modülü (day_night_sorter.py)
Faz 1: Dosya isminden tarih okuyarak gece ve gündüz fotoğraflarını ayırır.

Kullanım:
    python day_night_sorter.py
    python day_night_sorter.py --input raw_dataset --copy
"""

import os
import re
import shutil
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List
import math

try:
    import ephem
except ImportError:
    print("HATA: 'ephem' kütüphanesi bulunamadı. Lütfen 'pip install ephem' komutunu çalıştırın.")
    exit(1)

from config import (
    OBSERVER_LATITUDE,
    OBSERVER_LONGITUDE, 
    OBSERVER_ELEVATION,
    SUN_ALTITUDE_THRESHOLD,
    RAW_DATA_FOLDER,
    DAY_FOLDER,
    NIGHT_FOLDER,
    SUPPORTED_EXTENSIONS
)

# =============================================================================
# LOGGING AYARLARI
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# TARİH ÇIKARMA FONKSİYONLARI
# =============================================================================

# Desteklenen tarih formatları için regex desenleri
DATE_PATTERNS = [
    # Format: YYYY_MM_DD__HH_MM_SS (mevcut veri seti formatı)
    (r'(\d{4})_(\d{2})_(\d{2})__(\d{2})_(\d{2})_(\d{2})', '%Y_%m_%d__%H_%M_%S'),
    # Format: YYYYMMDD_HHMMSS (PRD'de belirtilen format)
    (r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})', '%Y%m%d_%H%M%S'),
    # Format: YYYY-MM-DD_HH-MM-SS
    (r'(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})', '%Y-%m-%d_%H-%M-%S'),
    # Format: img_YYYYMMDD_HHMM
    (r'img_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})', 'img_%Y%m%d_%H%M'),
]


def extract_datetime_from_filename(filename: str) -> Optional[datetime]:
    """
    Dosya isminden tarih ve saat bilgisini çıkarır.
    
    Args:
        filename: Dosya adı (örn: "2021_09_27__23_12_47.jpg")
    
    Returns:
        datetime objesi veya None (tarih bulunamazsa)
    
    Raises:
        ValueError: Tarih formatı bozuksa (error handling için loglanır, None döner)
    """
    # Sadece dosya adını al (path varsa)
    basename = os.path.basename(filename)
    
    for pattern, date_format in DATE_PATTERNS:
        match = re.search(pattern, basename)
        if match:
            try:
                # Grupları birleştirerek datetime string'i oluştur
                groups = match.groups()
                
                # YYYY_MM_DD__HH_MM_SS formatı için
                if len(groups) == 6:
                    year, month, day, hour, minute, second = groups
                    dt = datetime(
                        int(year), int(month), int(day),
                        int(hour), int(minute), int(second)
                    )
                    return dt
                # img_YYYYMMDD_HHMM formatı için (saniye yok)
                elif len(groups) == 5:
                    year, month, day, hour, minute = groups
                    dt = datetime(
                        int(year), int(month), int(day),
                        int(hour), int(minute), 0
                    )
                    return dt
                    
            except ValueError as e:
                logger.warning(f"Tarih parse hatası ({basename}): {e}")
                return None
    
    logger.warning(f"Tarih formatı tanınamadı: {basename}")
    return None


# =============================================================================
# GÜNEŞ POZİSYONU HESAPLAMA
# =============================================================================

def create_observer() -> ephem.Observer:
    """
    TUG gözlemevi için ephem.Observer objesi oluşturur.
    
    Returns:
        ephem.Observer: Konfigüre edilmiş gözlemci objesi
    """
    observer = ephem.Observer()
    observer.lat = OBSERVER_LATITUDE
    observer.lon = OBSERVER_LONGITUDE
    observer.elevation = OBSERVER_ELEVATION
    return observer


def get_sun_altitude(dt: datetime, observer: ephem.Observer = None) -> float:
    """
    Belirli bir tarih/saat için Güneş'in yükseklik açısını hesaplar.
    
    Args:
        dt: Tarih ve saat (datetime objesi)
        observer: ephem.Observer objesi (None ise varsayılan oluşturulur)
    
    Returns:
        float: Güneş açısı (derece cinsinden, negatif = ufkun altında)
    """
    if observer is None:
        observer = create_observer()
    
    # Tarihi ephem formatına çevir (UTC varsayılır, Türkiye UTC+3)
    # Not: Veriler yerel saat olabilir, gerekirse düzeltme yapılabilir
    observer.date = ephem.Date(dt)
    
    sun = ephem.Sun(observer)
    
    # Radyandan dereceye çevir
    altitude_deg = math.degrees(float(sun.alt))
    
    return altitude_deg


def is_night(dt: datetime, observer: ephem.Observer = None) -> bool:
    """
    Verilen tarih/saatin astronomik gece olup olmadığını kontrol eder.
    
    Kural: Güneş açısı < -12° ise GECE
    
    Args:
        dt: Tarih ve saat
        observer: ephem.Observer objesi
    
    Returns:
        bool: True = Gece, False = Gündüz/Alacakaranlık
    """
    sun_alt = get_sun_altitude(dt, observer)
    return sun_alt < SUN_ALTITUDE_THRESHOLD


# =============================================================================
# DOSYA TARAMA VE SIRALAMA
# =============================================================================

def scan_image_files(input_folder: str, recursive: bool = True) -> List[str]:
    """
    Belirtilen klasördeki tüm görüntü dosyalarını tarar.
    
    Args:
        input_folder: Taranacak klasör yolu
        recursive: Alt klasörleri de tara (varsayılan: True)
    
    Returns:
        List[str]: Dosya yolları listesi
    """
    image_files = []
    input_path = Path(input_folder)
    
    if not input_path.exists():
        logger.error(f"Klasör bulunamadı: {input_folder}")
        return image_files
    
    if recursive:
        # Alt klasörler dahil tüm dosyaları tara
        for ext in SUPPORTED_EXTENSIONS:
            image_files.extend(input_path.rglob(f"*{ext}"))
            image_files.extend(input_path.rglob(f"*{ext.upper()}"))
    else:
        # Sadece ana klasördeki dosyalar
        for ext in SUPPORTED_EXTENSIONS:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    return [str(f) for f in image_files]


def sort_images(
    input_folder: str,
    copy_files: bool = False,
    recursive: bool = True
) -> Tuple[int, int, int]:
    """
    Görüntüleri gece/gündüz klasörlerine ayırır.
    
    Args:
        input_folder: Kaynak klasör
        copy_files: True = Kopyala, False = Taşı
        recursive: Alt klasörleri de tara
    
    Returns:
        Tuple[int, int, int]: (gece_sayısı, gündüz_sayısı, hata_sayısı)
    """
    # Hedef klasörleri oluştur
    os.makedirs(DAY_FOLDER, exist_ok=True)
    os.makedirs(NIGHT_FOLDER, exist_ok=True)
    
    # Dosyaları tara
    image_files = scan_image_files(input_folder, recursive)
    total_files = len(image_files)
    
    if total_files == 0:
        logger.warning(f"'{input_folder}' klasöründe görüntü dosyası bulunamadı.")
        return 0, 0, 0
    
    logger.info(f"Toplam {total_files} dosya bulundu. İşlem başlıyor...")
    
    # Observer'ı bir kere oluştur (performans için)
    observer = create_observer()
    
    night_count = 0
    day_count = 0
    error_count = 0
    
    # İşlem fonksiyonu seç
    transfer_func = shutil.copy2 if copy_files else shutil.move
    action_word = "kopyalandı" if copy_files else "taşındı"
    
    for i, filepath in enumerate(image_files, 1):
        filename = os.path.basename(filepath)
        
        # İlerleme göster (her 100 dosyada bir)
        if i % 100 == 0 or i == total_files:
            logger.info(f"İşleniyor: {i}/{total_files} ({(i/total_files)*100:.1f}%)")
        
        # Tarih çıkar
        dt = extract_datetime_from_filename(filename)
        
        if dt is None:
            error_count += 1
            logger.debug(f"Atlandı (tarih okunamadı): {filename}")
            continue
        
        # Gece mi gündüz mü?
        if is_night(dt, observer):
            dest_folder = NIGHT_FOLDER
            night_count += 1
        else:
            dest_folder = DAY_FOLDER
            day_count += 1
        
        # Dosyayı taşı/kopyala
        dest_path = os.path.join(dest_folder, filename)
        
        try:
            # Aynı isimde dosya varsa atla
            if os.path.exists(dest_path):
                logger.debug(f"Zaten mevcut, atlandı: {filename}")
                continue
            transfer_func(filepath, dest_path)
        except Exception as e:
            error_count += 1
            logger.error(f"Dosya transfer hatası ({filename}): {e}")
    
    return night_count, day_count, error_count


# =============================================================================
# ANA FONKSİYON
# =============================================================================

def main():
    """Ana fonksiyon - komut satırı argümanlarını işler ve sıralamayı başlatır."""
    
    parser = argparse.ArgumentParser(
        description='SkyWatcher - Gece/Gündüz Görüntü Ayrıştırıcı',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Örnekler:
  python day_night_sorter.py
  python day_night_sorter.py --input ham_veriler --copy
  python day_night_sorter.py --input raw_dataset --no-recursive
        '''
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=RAW_DATA_FOLDER,
        help=f'Kaynak klasör yolu (varsayılan: {RAW_DATA_FOLDER})'
    )
    
    parser.add_argument(
        '--copy', '-c',
        action='store_true',
        help='Dosyaları taşımak yerine kopyala'
    )
    
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Alt klasörleri tarama'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Detaylı log çıktısı'
    )
    
    args = parser.parse_args()
    
    # Verbose mod
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Başlık
    print("=" * 60)
    print("  SkyWatcher - Gece/Gündüz Ayrıştırma Modülü")
    print("=" * 60)
    print(f"  Kaynak Klasör : {args.input}")
    print(f"  İşlem Modu    : {'Kopyala' if args.copy else 'Taşı'}")
    print(f"  Alt Klasörler : {'Hayır' if args.no_recursive else 'Evet'}")
    print(f"  Güneş Eşiği   : {SUN_ALTITUDE_THRESHOLD}°")
    print("=" * 60)
    
    # Klasör kontrolü
    if not os.path.exists(args.input):
        logger.error(f"Kaynak klasör bulunamadı: {args.input}")
        return
    
    # İşlemi başlat ve süreyi ölç
    import time
    start_time = time.time()
    
    night_count, day_count, error_count = sort_images(
        args.input,
        copy_files=args.copy,
        recursive=not args.no_recursive
    )
    
    elapsed_time = time.time() - start_time
    
    # Sonuç özeti
    print("\n" + "=" * 60)
    print("  SONUÇ ÖZETİ")
    print("=" * 60)
    print(f"  🌙 Gece Fotoğrafları   : {night_count}")
    print(f"  ☀️  Gündüz Fotoğrafları : {day_count}")
    print(f"  ⚠️  Hatalar/Atlamalar  : {error_count}")
    print(f"  ⏱️  Toplam Süre        : {elapsed_time:.2f} saniye")
    
    total_processed = night_count + day_count
    if total_processed > 0:
        fps = total_processed / elapsed_time
        print(f"  📊 Hız                 : {fps:.1f} dosya/saniye")
    
    print("=" * 60)
    print(f"  Gece klasörü  : {NIGHT_FOLDER}")
    print(f"  Gündüz klasörü: {DAY_FOLDER}")
    print("=" * 60)


if __name__ == "__main__":
    main()