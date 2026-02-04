"""
SkyWatcher - Gökyüzü Sınıflandırma Modülü (sky_classifier.py)
Faz 3: Maskelenmiş görüntülerin parlaklığını ölçerek bulut durumunu sınıflandırır.

Sınıflandırma:
- CLEAR (< 20): Açık gökyüzü - Gözlem yapılabilir
- NEUTRAL (20-45): Parçalı bulutlu - Riskli
- CLOUDY (> 45): Kapalı - Gözlem yapılamaz
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import json
import logging
from pathlib import Path

from config import CLEAR_THRESHOLD, NEUTRAL_THRESHOLD

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SkyCondition(Enum):
    """Gökyüzü durumu enum'u."""
    CLEAR = "CLEAR"
    NEUTRAL = "NEUTRAL"
    CLOUDY = "CLOUDY"


class ObservationStatus(Enum):
    """Gözlem durumu enum'u."""
    OBSERVABLE = "OBSERVABLE"
    RISKY = "RISKY"
    NOT_OBSERVABLE = "NOT_OBSERVABLE"


@dataclass
class ClassificationResult:
    """Sınıflandırma sonucu veri sınıfı."""
    filename: str
    mean_brightness: float
    sky_condition: str
    status: str
    moon_present: bool
    cloud_coverage_percent: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionary'e çevirir."""
        return {
            "filename": self.filename,
            "status": self.status,
            "sky_condition": self.sky_condition,
            "moon_present": self.moon_present,
            "cloud_coverage_percent": self.cloud_coverage_percent,
            "mean_brightness": round(self.mean_brightness, 2)
        }


class SkyClassifier:
    """
    Gökyüzü sınıflandırıcı sınıfı.
    
    Görüntünün ortalama parlaklığına göre bulut durumunu belirler.
    """
    
    def __init__(
        self,
        clear_threshold: int = CLEAR_THRESHOLD,
        neutral_threshold: int = NEUTRAL_THRESHOLD
    ):
        """
        Args:
            clear_threshold: CLEAR üst sınırı (varsayılan: 20)
            neutral_threshold: NEUTRAL üst sınırı (varsayılan: 45)
        """
        self.clear_threshold = clear_threshold
        self.neutral_threshold = neutral_threshold
    
    def calculate_mean_brightness(self, image: np.ndarray) -> float:
        """
        Görüntünün ortalama piksel parlaklığını hesaplar.
        
        Args:
            image: Görüntü (BGR veya grayscale)
            
        Returns:
            Ortalama parlaklık (0-255)
        """
        # Grayscale'e çevir
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return float(np.mean(gray))
    
    def calculate_cloud_coverage(self, image: np.ndarray, brightness_threshold: int = 30) -> int:
        """
        Bulut kaplama yüzdesini hesaplar.
        
        Parlak piksellerin (bulut) oranını hesaplar.
        
        Args:
            image: Görüntü
            brightness_threshold: Bulut sayılacak minimum parlaklık
            
        Returns:
            Bulut yüzdesi (0-100)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Parlak pikselleri say (bulut)
        cloud_pixels = np.sum(gray > brightness_threshold)
        total_pixels = gray.size
        
        coverage = (cloud_pixels / total_pixels) * 100
        return int(min(coverage, 100))
    
    def classify(self, mean_brightness: float) -> tuple:
        """
        Parlaklık değerine göre sınıflandırma yapar.
        
        Args:
            mean_brightness: Ortalama parlaklık (0-255)
            
        Returns:
            (SkyCondition, ObservationStatus)
        """
        if mean_brightness < self.clear_threshold:
            return SkyCondition.CLEAR, ObservationStatus.OBSERVABLE
        elif mean_brightness < self.neutral_threshold:
            return SkyCondition.NEUTRAL, ObservationStatus.RISKY
        else:
            return SkyCondition.CLOUDY, ObservationStatus.NOT_OBSERVABLE
    
    def process_image(
        self,
        image_path: str,
        moon_present: bool = False
    ) -> Optional[ClassificationResult]:
        """
        Görüntüyü analiz eder ve sınıflandırır.
        
        Args:
            image_path: Görüntü dosya yolu
            moon_present: Ay tespit edildi mi?
            
        Returns:
            ClassificationResult veya None
        """
        if not os.path.exists(image_path):
            logger.error(f"Dosya bulunamadı: {image_path}")
            return None
        
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Görüntü okunamadı: {image_path}")
            return None
        
        # Parlaklık hesapla
        mean_brightness = self.calculate_mean_brightness(image)
        
        # Bulut yüzdesi
        cloud_coverage = self.calculate_cloud_coverage(image)
        
        # Sınıflandır
        condition, status = self.classify(mean_brightness)
        
        filename = os.path.basename(image_path)
        
        result = ClassificationResult(
            filename=filename,
            mean_brightness=mean_brightness,
            sky_condition=condition.value,
            status=status.value,
            moon_present=moon_present,
            cloud_coverage_percent=cloud_coverage
        )
        
        logger.debug(f"{filename}: {condition.value} (brightness={mean_brightness:.1f})")
        
        return result


class SkyAnalysisPipeline:
    """
    Tam analiz pipeline'ı.
    
    Ay maskeleme + Sınıflandırma + JSON çıktısı
    """
    
    def __init__(self):
        from moon_masker import MoonMasker
        
        self.moon_masker = MoonMasker()
        self.classifier = SkyClassifier()
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Tek bir görüntüyü tam pipeline ile analiz eder.
        
        Args:
            image_path: Görüntü dosya yolu
            
        Returns:
            Analiz sonucu dictionary
        """
        # 1. Ay maskeleme
        mask_result = self.moon_masker.process_image(image_path)
        
        if not mask_result["success"]:
            return {
                "success": False,
                "error": mask_result.get("error", "Maskeleme hatası")
            }
        
        moon_present = mask_result["moon_detection"]["moon_detected"]
        masked_image = mask_result["masked_image"]
        
        # 2. Sınıflandırma (maskelenmiş görüntü üzerinde)
        mean_brightness = self.classifier.calculate_mean_brightness(masked_image)
        cloud_coverage = self.classifier.calculate_cloud_coverage(masked_image)
        condition, status = self.classifier.classify(mean_brightness)
        
        result = ClassificationResult(
            filename=os.path.basename(image_path),
            mean_brightness=mean_brightness,
            sky_condition=condition.value,
            status=status.value,
            moon_present=moon_present,
            cloud_coverage_percent=cloud_coverage
        )
        
        return {
            "success": True,
            "result": result.to_dict()
        }
    
    def analyze_folder(
        self,
        input_folder: str,
        output_json: str = None
    ) -> Dict[str, Any]:
        """
        Klasördeki tüm görüntüleri analiz eder.
        
        Args:
            input_folder: Giriş klasörü
            output_json: JSON çıktı dosyası (opsiyonel)
            
        Returns:
            Analiz özeti
        """
        input_path = Path(input_folder)
        
        if not input_path.exists():
            logger.error(f"Klasör bulunamadı: {input_folder}")
            return {"success": False, "error": "Klasör bulunamadı"}
        
        # Görüntüleri bul
        image_files = []
        for ext in [".jpg", ".jpeg", ".png"]:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        total = len(image_files)
        if total == 0:
            return {"success": True, "processed": 0, "results": []}
        
        logger.info(f"Toplam {total} görüntü analiz edilecek...")
        
        results = []
        stats = {"CLEAR": 0, "NEUTRAL": 0, "CLOUDY": 0, "moon_detected": 0}
        
        for i, img_path in enumerate(image_files, 1):
            if i % 100 == 0 or i == total:
                logger.info(f"Analiz ediliyor: {i}/{total}")
            
            try:
                analysis = self.analyze_image(str(img_path))
                
                if analysis["success"]:
                    result = analysis["result"]
                    results.append(result)
                    
                    # İstatistikler
                    stats[result["sky_condition"]] += 1
                    if result["moon_present"]:
                        stats["moon_detected"] += 1
                        
            except Exception as e:
                logger.error(f"Hata ({img_path.name}): {e}")
        
        # JSON'a kaydet
        if output_json and results:
            os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Sonuçlar kaydedildi: {output_json}")
        
        summary = {
            "success": True,
            "total": total,
            "processed": len(results),
            "statistics": stats,
            "clear_percent": f"{(stats['CLEAR']/len(results))*100:.1f}%" if results else "0%",
            "cloudy_percent": f"{(stats['CLOUDY']/len(results))*100:.1f}%" if results else "0%"
        }
        
        return summary


# =============================================================================
# KOMUT SATIRI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='SkyWatcher - Gökyüzü Sınıflandırıcı'
    )
    
    parser.add_argument('--input', '-i', type=str, help='Görüntü veya klasör yolu')
    parser.add_argument('--output', '-o', type=str, help='JSON çıktı dosyası')
    parser.add_argument('--verbose', '-v', action='store_true')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.input:
        parser.print_help()
        return
    
    pipeline = SkyAnalysisPipeline()
    
    print("=" * 60)
    print("  SkyWatcher - Gökyüzü Analiz Pipeline")
    print("=" * 60)
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Tek dosya
        result = pipeline.analyze_image(str(input_path))
        if result["success"]:
            print(json.dumps(result["result"], indent=2, ensure_ascii=False))
        else:
            print(f"Hata: {result.get('error')}")
    
    elif input_path.is_dir():
        # Klasör
        output_json = args.output or "analysis_results.json"
        summary = pipeline.analyze_folder(str(input_path), output_json)
        
        print(f"\n{'=' * 60}")
        print("  SONUÇ ÖZETİ")
        print(f"{'=' * 60}")
        print(f"  Toplam           : {summary.get('total', 0)}")
        print(f"  İşlenen          : {summary.get('processed', 0)}")
        print(f"  ☀️  CLEAR         : {summary['statistics']['CLEAR']}")
        print(f"  🌤️  NEUTRAL       : {summary['statistics']['NEUTRAL']}")
        print(f"  ☁️  CLOUDY        : {summary['statistics']['CLOUDY']}")
        print(f"  🌙 Ay Tespit     : {summary['statistics']['moon_detected']}")
        print(f"  Açık Hava Oranı  : {summary.get('clear_percent', '0%')}")
        print(f"{'=' * 60}")
        print(f"  JSON Çıktı: {output_json}")


if __name__ == "__main__":
    main()