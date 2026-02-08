#!/usr/bin/env python
"""
EVS Navigation System Deployment Script

Creates a deployment package with all necessary files for installation
on a new computer without Python.

Usage:
    python scripts/deploy.py --target ./deploy_package
    python scripts/deploy.py --target ./deploy_package --include-data
"""

import argparse
import shutil
import sys
from pathlib import Path


def create_deployment_package(target_dir: Path, include_data: bool = False):
    """Create a deployment package with all necessary files."""

    source_dir = Path(__file__).parent.parent.resolve()
    target_dir = target_dir.resolve()

    print(f"Source directory: {source_dir}")
    print(f"Target directory: {target_dir}")

    # Clean target directory if exists
    if target_dir.exists():
        print(f"Cleaning existing target directory...")
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    # Files and directories to include
    include_patterns = [
        # Main application
        "app.py",
        "requirements.txt",
        "requirements_advanced.txt",
        "create_evs_tables.sql",
        "init_database.py",

        # Core modules
        "db_utils.py",
        "db_manager.py",
        "user_utils.py",
        "paraverbal_process.py",
        "concordance_utils.py",
        "cache_utils.py",
        "file_alias_manager.py",
        "save_asr_results.py",
        "logger_config.py",
        "styles.py",
        "tencent_cloud_api.py",
        "llm_config.py",
        "asr_config.py",
        "edit_tracker.py",
        "email_queue.py",
        "email_metrics.py",
        "results_view.py",
        "privacy_settings.py",
        "file_display_utils.py",

        # Chinese NLP
        "chinese_nlp_unified.py",

        # Advanced analysis
        "advanced_interpretation_analysis.py",
        "advanced_analysis_methods.py",
        "app_integration.py",

        # Configuration
        "config/",

        # Utility modules
        "utils/",

        # Components
        "components/",

        # Pages
        "pages/",

        # Client modules
        "clients/",

        # Scripts
        "scripts/",
    ]

    # Directories to exclude
    exclude_patterns = [
        "__pycache__",
        ".git",
        ".vscode",
        "*.pyc",
        "*.pyo",
        ".env",
        "backup/",
        "logs/",
        "evs_resources/",
        "keys/",
        "www/",
        "tools/",
        "*.db",
    ]

    def should_exclude(path: Path) -> bool:
        """Check if path should be excluded."""
        name = path.name
        for pattern in exclude_patterns:
            if pattern.endswith('/'):
                if path.is_dir() and name == pattern[:-1]:
                    return True
            elif '*' in pattern:
                import fnmatch
                if fnmatch.fnmatch(name, pattern):
                    return True
            elif name == pattern:
                return True
        return False

    def copy_item(src: Path, dst: Path):
        """Copy file or directory."""
        if should_exclude(src):
            return

        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  Copied: {src.name}")
        elif src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            for item in src.iterdir():
                if not should_exclude(item):
                    copy_item(item, dst / item.name)

    print("\nCopying application files...")

    for pattern in include_patterns:
        src_path = source_dir / pattern
        if src_path.exists():
            if src_path.is_dir():
                copy_item(src_path, target_dir / pattern.rstrip('/'))
            else:
                copy_item(src_path, target_dir / pattern)
        else:
            print(f"  Warning: {pattern} not found")

    # Optionally include data directory
    if include_data:
        data_src = source_dir / "data"
        if data_src.exists():
            print("\nCopying data directory...")
            shutil.copytree(data_src, target_dir / "data", dirs_exist_ok=True)
    else:
        # Create empty data directory
        (target_dir / "data").mkdir(exist_ok=True)

    # Create keys directory placeholder
    keys_dir = target_dir / "keys"
    keys_dir.mkdir(exist_ok=True)
    (keys_dir / "README.txt").write_text(
        "Place your API key files here:\n"
        "- tencent_cloud_config.json (for Tencent ASR)\n"
        "- google_credentials.json (for Google Speech-to-Text)\n"
    )

    # Copy installation scripts
    scripts_src = Path(__file__).parent
    for script in ["setup.bat", "start.bat", "setup.sh", "start.sh", "verify_env.py"]:
        script_path = scripts_src / script
        if script_path.exists():
            shutil.copy2(script_path, target_dir / script)
            print(f"  Copied: {script}")

    # Copy installation guide
    guide_path = source_dir / "INSTALL_GUIDE.md"
    if guide_path.exists():
        shutil.copy2(guide_path, target_dir / "INSTALL_GUIDE.md")

    print(f"\n{'='*50}")
    print(f"Deployment package created: {target_dir}")
    print(f"{'='*50}")

    # Calculate package size
    total_size = sum(f.stat().st_size for f in target_dir.rglob('*') if f.is_file())
    print(f"Total size: {total_size / 1024 / 1024:.2f} MB")
    print(f"\nNext steps:")
    print(f"1. Copy the '{target_dir.name}' folder to the target computer")
    print(f"2. Follow instructions in INSTALL_GUIDE.md")


def main():
    parser = argparse.ArgumentParser(description="Create EVS deployment package")
    parser.add_argument(
        "--target", "-t",
        type=Path,
        default=Path("./deploy_package"),
        help="Target directory for deployment package"
    )
    parser.add_argument(
        "--include-data",
        action="store_true",
        help="Include existing database and data files"
    )

    args = parser.parse_args()
    create_deployment_package(args.target, args.include_data)


if __name__ == "__main__":
    main()
