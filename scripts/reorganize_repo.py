"""Reorganization script to clean redundant files and organize repository for public release.
"""

import shutil
from pathlib import Path


def reorganize():
    root = Path(__file__).resolve().parent.parent
    print(f"Project root: {root}")

    # 1. Ensure target directories exist
    docs_metrics = root / "docs" / "training_metrics"
    docs_metrics.mkdir(parents=True, exist_ok=True)

    legacy_dir = root / "legacy"
    legacy_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = root / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    notebooks_dir = root / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    source_dir = root / "Source"

    # 2. Copy metrics from Source to docs/training_metrics
    if source_dir.exists():
        metric_patterns = [
            "Box*.png",
            "confusion_matrix*.png",
            "results.*",
            "labels.jpg",
            "args.yaml",
            "fire_data.yaml",
            "val_batch*.jpg",
        ]
        for pattern in metric_patterns:
            for file_path in source_dir.glob(pattern):
                dest = docs_metrics / file_path.name
                shutil.copy2(file_path, dest)
                print(f"Copied {file_path.name} -> {dest.relative_to(root)}")

    # 3. Clean up root auto_detect2.py (already migrated to scripts/auto_label_fire_smoke.py)
    root_auto_detect = root / "auto_detect2.py"
    if root_auto_detect.exists():
        root_auto_detect.unlink()
        print("Removed root auto_detect2.py (migrated to scripts/auto_label_fire_smoke.py)")

    # 4. Remove root yolov8l-world.pt if exists (95MB binary)
    root_world_pt = root / "yolov8l-world.pt"
    if root_world_pt.exists():
        root_world_pt.unlink()
        print("Removed root yolov8l-world.pt (can be auto-downloaded by ultralytics)")

    # 5. Remove redundant New folder/ directories
    redundant_folders = [
        root / "New folder",
        source_dir / "New folder",
    ]
    for folder in redundant_folders:
        if folder.exists() and folder.is_dir():
            shutil.rmtree(folder, ignore_errors=True)
            print(f"Removed redundant directory: {folder.relative_to(root)}")

    # 6. Remove Source/ folder now that all assets are migrated
    if source_dir.exists() and source_dir.is_dir():
        shutil.rmtree(source_dir, ignore_errors=True)
        print("Removed legacy Source/ folder (assets migrated to src, legacy, weights, docs, notebooks)")

    print("\n--> Repository reorganization completed successfully!")


if __name__ == "__main__":
    reorganize()
