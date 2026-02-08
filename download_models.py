"""
EVS Navigation System - Model Download Script
Download ASR models independently before running the application.

Usage:
    python download_models.py              # Download all models
    python download_models.py crisperwhisper  # Download CrisperWhisper only
    python download_models.py funasr          # Download FunASR only
"""

import sys
import os
import time


def download_crisperwhisper():
    """Download CrisperWhisper model (~3GB)."""
    print("=" * 60)
    print("  Downloading CrisperWhisper (nyrahealth/CrisperWhisper)")
    print("  Size: ~3GB | English verbatim ASR")
    print("=" * 60)
    print()

    try:
        from huggingface_hub import snapshot_download, hf_hub_download

        # Check if already downloaded
        try:
            hf_hub_download("nyrahealth/CrisperWhisper", "config.json", local_files_only=True)
            print("[OK] CrisperWhisper model already downloaded (cached).")
            return True
        except Exception:
            pass

        print("[INFO] Downloading... this may take a while.")
        print()

        start = time.time()
        path = snapshot_download("nyrahealth/CrisperWhisper")
        elapsed = time.time() - start

        print()
        print(f"[OK] CrisperWhisper downloaded successfully!")
        print(f"     Location: {path}")
        print(f"     Time: {elapsed:.0f}s")
        return True

    except ImportError:
        print("[ERROR] huggingface_hub not installed.")
        print("        Run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return False


def download_funasr():
    """Download FunASR paraformer-zh model (~1GB)."""
    print("=" * 60)
    print("  Downloading FunASR Paraformer-ZH (Alibaba)")
    print("  Size: ~1GB | Chinese ASR")
    print("=" * 60)
    print()

    try:
        from funasr import AutoModel

        print("[INFO] Downloading... this may take a while.")
        print()

        start = time.time()
        model = AutoModel(model="paraformer-zh")
        elapsed = time.time() - start

        print()
        print(f"[OK] FunASR paraformer-zh downloaded successfully!")
        print(f"     Time: {elapsed:.0f}s")
        del model
        return True

    except ImportError:
        print("[ERROR] funasr not installed.")
        print("        Run: pip install funasr modelscope")
        return False
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return False


def main():
    print()
    print("============================================================")
    print("  EVS Navigation System - Model Downloader")
    print("============================================================")
    print()

    args = [a.lower() for a in sys.argv[1:]]

    # Determine what to download
    if not args or 'all' in args:
        targets = ['crisperwhisper', 'funasr']
    else:
        targets = args

    results = {}

    for target in targets:
        if target in ('crisperwhisper', 'cw', 'whisper'):
            results['CrisperWhisper'] = download_crisperwhisper()
            print()
        elif target in ('funasr', 'paraformer', 'chinese'):
            results['FunASR'] = download_funasr()
            print()
        else:
            print(f"[WARNING] Unknown model: {target}")
            print(f"          Available: crisperwhisper, funasr")
            print()

    # Summary
    print("============================================================")
    print("  Download Summary")
    print("============================================================")
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name}: {status}")
    print()

    if all(results.values()):
        print("  All models ready. You can start the app with start.bat")
    else:
        print("  Some downloads failed. Check errors above.")
    print()


if __name__ == "__main__":
    main()
