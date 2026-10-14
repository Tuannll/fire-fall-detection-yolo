"""Script to organize and centralize weights into weights/ directory."""
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = BASE_DIR / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

models = {
    "fire_best.pt": [
        BASE_DIR / "Source" / "fire_best.pt",
        BASE_DIR / "Computer-Vision" / "Models" / "fire_best.pt",
    ],
    "fall_lstm_best.pt": [
        BASE_DIR / "Source" / "fall_lstm_best.pt",
        BASE_DIR / "Computer-Vision" / "Models" / "fall_lstm_best.pt",
    ],
    "yolov8n-pose.pt": [
        BASE_DIR / "Source" / "yolov8n-pose.pt",
        BASE_DIR / "Source" / "New folder" / "yolov8n-pose.pt",
    ],
}

def main():
    print("Setting up weights directory...")
    for target_name, candidates in models.items():
        target_path = WEIGHTS_DIR / target_name
        if target_path.exists():
            print(f"[OK] Already exists: {target_path}")
            continue

        found = False
        for c in candidates:
            if c.exists():
                print(f"[COPY] Copying {c} -> {target_path}")
                shutil.copy2(c, target_path)
                found = True
                break

        if not found:
            print(f"[WARNING] Could not find source for {target_name}")

    print("Weights setup complete.")

if __name__ == "__main__":
    main()
