"""
SkyWatcher - Ay Maskeleme Modülü (moon_masker.py)
Faz 2: Gece görüntülerinde Ay'ı tespit edip maskeler.

Algoritma:
1. Görüntüyü grayscale'e çevir
2. Gaussian blur ile gürültüyü azalt
3. HoughCircles ile dairesel parlak nesneleri (Ay) tespit et
4. Tespit edilen Ay'ı siyah daire ile maskele

Kullanım:
    from moon_masker import MoonMasker
    
    masker = MoonMasker()
    result = masker.process_image("image.jpg")
    
    veya komut satırından:
    python moon_masker.py --input image.jpg --output masked_image.jpg
"""

import os
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MoonDetectionResult:
    """Ay tespit sonucunu tutan veri sınıfı."""
    detected: bool
    center_x: Optional[int] = None
    center_y: Optional[int] = None
    radius: Optional[int] = None
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Sonucu dictionary olarak döndürür."""
        return {
            "moon_detected": self.detected,
            "center": (self.center_x, self.center_y) if self.detected else None,
            "radius": self.radius,
            "confidence": round(self.confidence, 2)
        }


class MoonMasker:
    """
    Ay tespit ve maskeleme sınıfı.
    
    HoughCircles algoritması kullanarak görüntüdeki Ay'ı tespit eder
    ve siyah daire ile maskeler.
    """
    
    def __init__(
        self,
        min_radius: int = 15,
        max_radius: int = 150,
        blur_kernel: int = 9,
        param1: int = 50,
        param2: int = 30,
        brightness_threshold: int = 200,
        mask_padding: float = 1.3
    ):
        """
        MoonMasker yapılandırması.
        
        Args:
            min_radius: Tespit edilecek minimum daire yarıçapı (piksel)
            max_radius: Tespit edilecek maksimum daire yarıçapı (piksel)
            blur_kernel: Gaussian blur kernel boyutu (tek sayı olmalı)
            param1: HoughCircles Canny edge üst eşiği
            param2: HoughCircles merkez tespit eşiği (düşük = daha hassas)
            brightness_threshold: Ay olarak kabul edilecek minimum parlaklık (0-255)
            mask_padding: Maske çapı çarpanı (halo etkisi için)
        """
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.blur_kernel = blur_kernel
        self.param1 = param1
        self.param2 = param2
        self.brightness_threshold = brightness_threshold
        self.mask_padding = mask_padding
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Görüntüyü yükler.
        
        Args:
            image_path: Görüntü dosya yolu
            
        Returns:
            numpy array (BGR formatında) veya None
        """
        if not os.path.exists(image_path):
            logger.error(f"Dosya bulunamadı: {image_path}")
            return None
        
        image = cv2.imread(image_path)
        
        if image is None:
            logger.error(f"Görüntü okunamadı: {image_path}")
            return None
        
        return image
    
    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Görüntüyü gri tonlamaya çevirir.
        
        Args:
            image: BGR formatında görüntü
            
        Returns:
            Grayscale görüntü
        """
        if len(image.shape) == 2:
            # Zaten grayscale
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def apply_blur(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Gürültü azaltma için Gaussian blur uygular.
        
        Args:
            gray_image: Grayscale görüntü
            
        Returns:
            Bulanıklaştırılmış görüntü
        """
        return cv2.GaussianBlur(
            gray_image, 
            (self.blur_kernel, self.blur_kernel), 
            0
        )
    
    def detect_moon(self, gray_image: np.ndarray) -> MoonDetectionResult:
        """
        HoughCircles ile Ay'ı tespit eder.
        
        Args:
            gray_image: Grayscale görüntü
            
        Returns:
            MoonDetectionResult objesi
        """
        # Blur uygula
        blurred = self.apply_blur(gray_image)
        
        # HoughCircles ile daire tespiti
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        
        if circles is None:
            logger.debug("Hiçbir daire tespit edilemedi")
            return MoonDetectionResult(detected=False)
        
        # Daireleri yuvarla ve numpy array'e çevir
        circles = np.uint16(np.around(circles))
        
        # En parlak daireyi bul (Ay muhtemelen en parlak)
        best_circle = None
        best_brightness = 0
        
        for circle in circles[0, :]:
            x, y, r = circle
            
            # Dairenin içindeki ortalama parlaklığı hesapla
            mask = np.zeros(gray_image.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            mean_brightness = cv2.mean(gray_image, mask=mask)[0]
            
            # Parlaklık eşiğini kontrol et
            if mean_brightness > self.brightness_threshold and mean_brightness > best_brightness:
                best_brightness = mean_brightness
                best_circle = circle
        
        if best_circle is None:
            logger.debug("Yeterince parlak daire bulunamadı")
            return MoonDetectionResult(detected=False)
        
        x, y, r = best_circle
        confidence = min(best_brightness / 255.0, 1.0)
        
        logger.info(f"Ay tespit edildi: merkez=({x}, {y}), yarıçap={r}, parlaklık={best_brightness:.1f}")
        
        return MoonDetectionResult(
            detected=True,
            center_x=int(x),
            center_y=int(y),
            radius=int(r),
            confidence=confidence
        )
    
    def apply_mask(
        self, 
        image: np.ndarray, 
        detection: MoonDetectionResult
    ) -> np.ndarray:
        """
        Tespit edilen Ay'ı siyah daire ile maskeler.
        
        Args:
            image: Orijinal görüntü (BGR veya grayscale)
            detection: Ay tespit sonucu
            
        Returns:
            Maskelenmiş görüntü
        """
        if not detection.detected:
            return image.copy()
        
        masked_image = image.copy()
        
        # Maske yarıçapını biraz büyüt (halo etkisi için)
        mask_radius = int(detection.radius * self.mask_padding)
        
        # Siyah daire çiz
        cv2.circle(
            masked_image,
            (detection.center_x, detection.center_y),
            mask_radius,
            (0, 0, 0) if len(image.shape) == 3 else 0,
            -1  # Dolu daire
        )
        
        logger.debug(f"Maske uygulandı: yarıçap={mask_radius}")
        
        return masked_image
    
    def process_image(
        self, 
        image_path: str,
        return_intermediate: bool = False
    ) -> Dict[str, Any]:
        """
        Görüntüyü işleyerek Ay'ı tespit eder ve maskeler.
        
        Args:
            image_path: Görüntü dosya yolu
            return_intermediate: Ara sonuçları da döndür (debug için)
            
        Returns:
            Dictionary: {
                "success": bool,
                "moon_detection": MoonDetectionResult dict,
                "masked_image": numpy array,
                "original_image": numpy array (opsiyonel),
                "grayscale_image": numpy array (opsiyonel)
            }
        """
        result = {
            "success": False,
            "moon_detection": None,
            "masked_image": None,
            "error": None
        }
        
        # Görüntüyü yükle
        image = self.load_image(image_path)
        if image is None:
            result["error"] = "Görüntü yüklenemedi"
            return result
        
        # Grayscale'e çevir
        gray = self.to_grayscale(image)
        
        # Ay tespiti
        detection = self.detect_moon(gray)
        
        # Maskeleme (orijinal renkli görüntü üzerinde)
        masked_image = self.apply_mask(image, detection)
        
        result["success"] = True
        result["moon_detection"] = detection.to_dict()
        result["masked_image"] = masked_image
        
        if return_intermediate:
            result["original_image"] = image
            result["grayscale_image"] = gray
        
        return result
    
    def process_and_save(
        self,
        input_path: str,
        output_path: str
    ) -> Dict[str, Any]:
        """
        Görüntüyü işler ve sonucu kaydeder.
        
        Args:
            input_path: Giriş görüntü yolu
            output_path: Çıkış görüntü yolu
            
        Returns:
            İşlem sonucu dictionary
        """
        result = self.process_image(input_path)
        
        if result["success"] and result["masked_image"] is not None:
            # Çıkış klasörünü oluştur
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            
            # Kaydet
            cv2.imwrite(output_path, result["masked_image"])
            logger.info(f"Maskelenmiş görüntü kaydedildi: {output_path}")
            result["output_path"] = output_path
        
        return result


def process_night_folder(
    input_folder: str,
    output_folder: str = None,
    masker: MoonMasker = None
) -> Dict[str, Any]:
    """
    Bir klasördeki tüm gece görüntülerini işler.
    
    Args:
        input_folder: Giriş klasörü (dataset/night)
        output_folder: Çıkış klasörü (None ise yerinde değiştirir)
        masker: MoonMasker instance (None ise varsayılan oluşturulur)
        
    Returns:
        İşlem özeti dictionary
    """
    from pathlib import Path
    
    if masker is None:
        masker = MoonMasker()
    
    input_path = Path(input_folder)
    
    if not input_path.exists():
        logger.error(f"Klasör bulunamadı: {input_folder}")
        return {"success": False, "error": "Klasör bulunamadı"}
    
    # Görüntü dosyalarını bul
    image_extensions = [".jpg", ".jpeg", ".png"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    total = len(image_files)
    if total == 0:
        logger.warning("İşlenecek görüntü bulunamadı")
        return {"success": True, "processed": 0, "moon_detected": 0}
    
    logger.info(f"Toplam {total} görüntü işlenecek...")
    
    processed = 0
    moon_detected = 0
    errors = 0
    
    for i, img_path in enumerate(image_files, 1):
        if i % 50 == 0 or i == total:
            logger.info(f"İşleniyor: {i}/{total}")
        
        # Çıkış yolunu belirle
        if output_folder:
            out_path = Path(output_folder) / img_path.name
        else:
            out_path = img_path
        
        try:
            result = masker.process_and_save(str(img_path), str(out_path))
            
            if result["success"]:
                processed += 1
                if result["moon_detection"]["moon_detected"]:
                    moon_detected += 1
            else:
                errors += 1
                
        except Exception as e:
            logger.error(f"Hata ({img_path.name}): {e}")
            errors += 1
    
    summary = {
        "success": True,
        "total": total,
        "processed": processed,
        "moon_detected": moon_detected,
        "errors": errors,
        "moon_detection_rate": f"{(moon_detected/processed)*100:.1f}%" if processed > 0 else "0%"
    }
    
    logger.info(f"İşlem tamamlandı: {processed}/{total} başarılı, {moon_detected} Ay tespit edildi")
    
    return summary


# =============================================================================
# KOMUT SATIRI ARAYÜZÜ
# =============================================================================

def main():
    """Komut satırı arayüzü."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='SkyWatcher - Ay Maskeleme Modülü',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Örnekler:
  # Tek bir görüntüyü işle
  python moon_masker.py --input image.jpg --output masked.jpg
  
  # Klasördeki tüm görüntüleri işle
  python moon_masker.py --folder dataset/night --output-folder dataset/night_masked
  
  # Yerinde değiştir (dikkatli kullanın!)
  python moon_masker.py --folder dataset/night --inplace
        '''
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Tek görüntü dosya yolu'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Çıkış dosya yolu (tek görüntü için)'
    )
    
    parser.add_argument(
        '--folder', '-f',
        type=str,
        help='İşlenecek klasör yolu'
    )
    
    parser.add_argument(
        '--output-folder',
        type=str,
        help='Çıkış klasörü (klasör işleme için)'
    )
    
    parser.add_argument(
        '--inplace',
        action='store_true',
        help='Görüntüleri yerinde değiştir'
    )
    
    parser.add_argument(
        '--min-radius',
        type=int,
        default=15,
        help='Minimum Ay yarıçapı (piksel)'
    )
    
    parser.add_argument(
        '--max-radius',
        type=int,
        default=150,
        help='Maksimum Ay yarıçapı (piksel)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Detaylı çıktı'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # MoonMasker oluştur
    masker = MoonMasker(
        min_radius=args.min_radius,
        max_radius=args.max_radius
    )
    
    print("=" * 60)
    print("  SkyWatcher - Ay Maskeleme Modülü")
    print("=" * 60)
    
    # Tek görüntü modu
    if args.input:
        output = args.output or args.input.replace('.jpg', '_masked.jpg')
        result = masker.process_and_save(args.input, output)
        
        if result["success"]:
            print(f"\n✓ İşlem tamamlandı")
            print(f"  Ay tespit edildi: {result['moon_detection']['moon_detected']}")
            if result['moon_detection']['moon_detected']:
                print(f"  Merkez: {result['moon_detection']['center']}")
                print(f"  Yarıçap: {result['moon_detection']['radius']}px")
            print(f"  Çıkış: {output}")
        else:
            print(f"\n✗ Hata: {result.get('error', 'Bilinmeyen hata')}")
    
    # Klasör modu
    elif args.folder:
        output_folder = None if args.inplace else args.output_folder
        
        if not args.inplace and not output_folder:
            print("HATA: --output-folder veya --inplace belirtmelisiniz")
            return
        
        summary = process_night_folder(args.folder, output_folder, masker)
        
        print(f"\n{'=' * 60}")
        print("  SONUÇ ÖZETİ")
        print(f"{'=' * 60}")
        print(f"  Toplam Görüntü    : {summary.get('total', 0)}")
        print(f"  Başarılı İşlenen  : {summary.get('processed', 0)}")
        print(f"  🌙 Ay Tespit Edilen: {summary.get('moon_detected', 0)}")
        print(f"  Tespit Oranı      : {summary.get('moon_detection_rate', '0%')}")
        print(f"  Hatalar           : {summary.get('errors', 0)}")
        print(f"{'=' * 60}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()