"""
SkyWatcher - Ana Uygulama (main.py)
Tüm modülleri birleştiren ana giriş noktası.

Kullanım:
    python main.py analyze --input image.jpg
    python main.py analyze --folder ../dataset/night --output results.json
    python main.py sort --input ../raw_dataset
    python main.py label --input ../dataset/night
"""

import argparse
import sys
import os

# Backend klasörünü path'e ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_analyze(args):
    """Görüntü analizi komutu."""
    from sky_classifier import SkyAnalysisPipeline
    import json
    from pathlib import Path
    
    pipeline = SkyAnalysisPipeline()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        result = pipeline.analyze_image(str(input_path))
        if result["success"]:
            print(json.dumps(result["result"], indent=2, ensure_ascii=False))
        else:
            print(f"Hata: {result.get('error')}")
            
    elif input_path.is_dir():
        output = args.output or "analysis_results.json"
        summary = pipeline.analyze_folder(str(input_path), output)
        
        print(f"\n✅ Analiz tamamlandı!")
        print(f"   CLEAR: {summary['statistics']['CLEAR']}")
        print(f"   NEUTRAL: {summary['statistics']['NEUTRAL']}")
        print(f"   CLOUDY: {summary['statistics']['CLOUDY']}")
        print(f"   Sonuçlar: {output}")


def cmd_sort(args):
    """Gece/gündüz sıralama komutu."""
    from day_night_sorter import sort_images
    
    night, day, errors = sort_images(
        args.input,
        copy_files=args.copy,
        recursive=not args.no_recursive
    )
    
    print(f"\n✅ Sıralama tamamlandı!")
    print(f"   🌙 Gece: {night}")
    print(f"   ☀️  Gündüz: {day}")
    print(f"   ⚠️  Hatalar: {errors}")


def cmd_label(args):
    """Otomatik etiketleme komutu."""
    from auto_labeler import auto_label_images
    
    summary = auto_label_images(
        args.input,
        copy_files=not args.move,
        dry_run=args.dry_run
    )
    
    print(f"\n✅ Etiketleme tamamlandı!")
    print(f"   ☀️  CLEAR: {summary['clear']}")
    print(f"   🌤️  NEUTRAL: {summary['neutral']}")
    print(f"   ☁️  CLOUDY: {summary['cloudy']}")


def main():
    parser = argparse.ArgumentParser(
        description='SkyWatcher - Teleskop Görüntü Analiz Sistemi',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Komutlar')
    
    # analyze komutu
    analyze_parser = subparsers.add_parser('analyze', help='Görüntü analizi')
    analyze_parser.add_argument('--input', '-i', required=True, help='Görüntü veya klasör')
    analyze_parser.add_argument('--output', '-o', help='JSON çıktı dosyası')
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # sort komutu
    sort_parser = subparsers.add_parser('sort', help='Gece/gündüz sıralama')
    sort_parser.add_argument('--input', '-i', required=True, help='Kaynak klasör')
    sort_parser.add_argument('--copy', action='store_true', help='Kopyala')
    sort_parser.add_argument('--no-recursive', action='store_true')
    sort_parser.set_defaults(func=cmd_sort)
    
    # label komutu
    label_parser = subparsers.add_parser('label', help='Otomatik etiketleme')
    label_parser.add_argument('--input', '-i', required=True, help='Gece klasörü')
    label_parser.add_argument('--move', action='store_true', help='Taşı')
    label_parser.add_argument('--dry-run', action='store_true', help='Simülasyon')
    label_parser.set_defaults(func=cmd_label)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  🔭 SkyWatcher - Teleskop Görüntü Analiz Sistemi")
    print("=" * 60)
    
    if args.command:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()