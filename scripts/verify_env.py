#!/usr/bin/env python
"""
EVS Navigation System - Environment Verification Script

Checks all required dependencies and system configurations.

Usage:
    python verify_env.py
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version is 3.8+."""
    print("[Python Version]")
    version = sys.version_info
    print(f"  Version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  Status: FAIL - Python 3.8+ required")
        return False

    print("  Status: OK")
    return True


def check_ffmpeg():
    """Check FFmpeg installation."""
    print("\n[FFmpeg]")
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"  {version_line}")
            print("  Status: OK")
            return True
    except FileNotFoundError:
        pass

    print("  Status: NOT FOUND")
    print("  Note: Required for audio processing")
    return False


def check_cuda():
    """Check CUDA availability for GPU acceleration."""
    print("\n[CUDA / GPU]")
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"  CUDA Available: Yes")
            print(f"  GPU: {device_name}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print("  Status: OK (GPU acceleration enabled)")
            return True
        else:
            print("  CUDA Available: No")
            print("  Status: OK (CPU mode)")
            return True
    except ImportError:
        print("  PyTorch not installed")
        print("  Status: OK (CPU mode)")
        return True


def check_package(package_name, import_name=None):
    """Check if a Python package is installed."""
    import_name = import_name or package_name
    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "unknown")
        return True, version
    except ImportError:
        return False, None


def check_core_packages():
    """Check core package installations."""
    print("\n[Core Packages]")

    packages = [
        ("streamlit", "streamlit"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("plotly", "plotly"),
        ("pydub", "pydub"),
        ("torch", "torch"),
        ("torchaudio", "torchaudio"),
    ]

    all_ok = True
    for package_name, import_name in packages:
        installed, version = check_package(package_name, import_name)
        status = f"v{version}" if installed else "NOT FOUND"
        symbol = "✓" if installed else "✗"
        print(f"  {symbol} {package_name}: {status}")
        if not installed:
            all_ok = False

    return all_ok


def check_asr_packages():
    """Check ASR-related package installations."""
    print("\n[ASR Packages]")

    packages = [
        ("transformers", "transformers"),
        ("funasr", "funasr"),
        ("google-cloud-speech", "google.cloud.speech"),
        ("tencentcloud-sdk-python", "tencentcloud"),
        ("ibm-watson", "ibm_watson"),
    ]

    for package_name, import_name in packages:
        installed, version = check_package(package_name, import_name)
        status = f"v{version}" if installed else "NOT FOUND (optional)"
        symbol = "✓" if installed else "○"
        print(f"  {symbol} {package_name}: {status}")

    return True  # ASR packages are optional


def check_nlp_packages():
    """Check NLP package installations."""
    print("\n[NLP Packages]")

    packages = [
        ("jieba", "jieba"),
        ("hanlp", "hanlp"),
    ]

    all_ok = True
    for package_name, import_name in packages:
        installed, version = check_package(package_name, import_name)
        status = f"v{version}" if installed else "NOT FOUND"
        symbol = "✓" if installed else "✗"
        print(f"  {symbol} {package_name}: {status}")
        if not installed:
            all_ok = False

    # Check spaCy models
    try:
        import spacy
        print(f"  ✓ spacy: v{spacy.__version__}")

        # Check Chinese model
        try:
            spacy.load("zh_core_web_sm")
            print("    ✓ zh_core_web_sm model: installed")
        except OSError:
            print("    ○ zh_core_web_sm model: NOT FOUND (optional)")

        # Check English model
        try:
            spacy.load("en_core_web_sm")
            print("    ✓ en_core_web_sm model: installed")
        except OSError:
            print("    ○ en_core_web_sm model: NOT FOUND (optional)")

    except ImportError:
        print("  ○ spacy: NOT FOUND (optional for advanced features)")

    return all_ok


def check_database():
    """Check database file and schema."""
    print("\n[Database]")

    db_path = Path(__file__).parent / "data" / "evs_repository.db"

    if db_path.exists():
        size_mb = db_path.stat().st_size / 1024 / 1024
        print(f"  Path: {db_path}")
        print(f"  Size: {size_mb:.2f} MB")

        # Check tables
        import sqlite3
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row[0] for row in cursor.fetchall()]
            tables = [t for t in tables if t != 'sqlite_sequence']
            conn.close()

            print(f"  Tables: {len(tables)}")
            print("  Status: OK")
            return True
        except sqlite3.Error as e:
            print(f"  Error: {e}")
            return False
    else:
        print(f"  Path: {db_path}")
        print("  Status: NOT FOUND")
        print("  Note: Run 'python init_database.py' to create")
        return False


def check_api_keys():
    """Check for API configuration files."""
    print("\n[API Configuration]")

    keys_dir = Path(__file__).parent / "keys"

    configs = [
        ("tencent_cloud_config.json", "Tencent Cloud ASR"),
        ("google_credentials.json", "Google Cloud Speech"),
    ]

    for filename, description in configs:
        path = keys_dir / filename
        if path.exists():
            print(f"  ✓ {description}: configured")
        else:
            print(f"  ○ {description}: not configured (optional)")

    return True


def main():
    """Run all environment checks."""
    print("=" * 60)
    print("  EVS Navigation System - Environment Verification")
    print("=" * 60)
    print()

    results = []

    results.append(("Python", check_python_version()))
    results.append(("FFmpeg", check_ffmpeg()))
    results.append(("CUDA", check_cuda()))
    results.append(("Core Packages", check_core_packages()))
    results.append(("ASR Packages", check_asr_packages()))
    results.append(("NLP Packages", check_nlp_packages()))
    results.append(("Database", check_database()))
    results.append(("API Keys", check_api_keys()))

    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "OK" if passed else "NEEDS ATTENTION"
        symbol = "✓" if passed else "!"
        print(f"  {symbol} {name}: {status}")
        if not passed and name in ["Python", "Core Packages"]:
            all_passed = False

    print()
    if all_passed:
        print("  Environment is ready!")
        print("  Run 'streamlit run app.py' to start the application.")
    else:
        print("  Some issues need to be resolved.")
        print("  Run 'pip install -r requirements.txt' to install missing packages.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
