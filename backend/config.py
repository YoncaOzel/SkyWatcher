"""
SkyWatcher - Konfigürasyon Dosyası
TUG (Tübitak Ulusal Gözlemevi) için sabit değerler ve ayarlar.
"""

import os

# =============================================================================
# PROJE KÖK DİZİNİ
# =============================================================================
# Backend klasöründen bir üst dizine çık
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# GÖZLEMEVİ KOORDİNATLARI (TUG - Bakırtepe)
# =============================================================================
OBSERVER_LATITUDE = "36.8245"    # Kuzey enlemi (derece)
OBSERVER_LONGITUDE = "30.3353"   # Doğu boylamı (derece)
OBSERVER_ELEVATION = 2500        # Rakım (metre)

# =============================================================================
# GECE/GÜNDÜZ AYRIM EŞİĞİ
# =============================================================================
SUN_ALTITUDE_THRESHOLD = -12.0  # derece

# =============================================================================
# KLASÖR YAPISI (Proje köküne göre)
# =============================================================================
RAW_DATA_FOLDER = os.path.join(PROJECT_ROOT, "raw_dataset")
DATASET_FOLDER = os.path.join(PROJECT_ROOT, "dataset")
DAY_FOLDER = os.path.join(PROJECT_ROOT, "dataset", "day")
NIGHT_FOLDER = os.path.join(PROJECT_ROOT, "dataset", "night")

# =============================================================================
# BULUT SINIFLANDIRMA EŞİKLERİ (Parlaklık 0-255)
# =============================================================================
CLEAR_THRESHOLD = 20
NEUTRAL_THRESHOLD = 45

# =============================================================================
# DESTEKLENEN DOSYA FORMATLARI
# =============================================================================
SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png"]