"""Auto-labeling tool for Fire & Smoke dataset using YOLO-World.
Scans images from an input folder and generates YOLO format annotation .txt files.
"""

from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from ultralytics import YOLOWorld


def run_auto_label(
    input_dir: str = "Img",
    output_dir: str = "Label_auto",
    model_name: str = "yolov8l-world.pt",
    conf: float = 0.15,
):
    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / input_dir
    output_path = project_root / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--> Running YOLO-World on device: {device}")

    weights_candidate = project_root / "weights" / model_name
    model_path = str(weights_candidate) if weights_candidate.exists() else model_name
    model = YOLOWorld(model_path)

    custom_prompts = [
        # Fire keywords
        "fire", "fire flame", "bonfire", "wildfire", "blaze", "burning fire", "glowing embers",
        # Smoke keywords
        "smoke", "smoke plume", "thick dark smoke", "white gray smoke", "wildfire smoke", "industrial smoke", "fume"
    ]
    model.set_classes(custom_prompts)

    class_mapping = {
        "fire": 0, "fire flame": 0, "bonfire": 0, "wildfire": 0, "blaze": 0, "burning fire": 0, "glowing embers": 0,
        "smoke": 1, "smoke plume": 1, "thick dark smoke": 1, "white gray smoke": 1, "wildfire smoke": 1, "industrial smoke": 1, "fume": 1
    }

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not input_path.exists():
        print(f"Input directory does not exist: {input_path}")
        return

    images = [f for f in input_path.iterdir() if f.suffix.lower() in valid_extensions]
    print(f"Found {len(images)} images to process in {input_path}")

    for img_path in tqdm(images, desc="Auto-labeling Fire & Smoke"):
        results = model.predict(str(img_path), conf=conf, device=device, verbose=False)[0]

        txt_lines = []
        if results.boxes is not None:
            for box in results.boxes:
                xywhn = box.xywhn[0].tolist()
                pred_cls_id = int(box.cls[0].item())
                pred_cls_name = custom_prompts[pred_cls_id]
                class_id = class_mapping.get(pred_cls_name, 1)

                x_center, y_center, width, height = xywhn
                line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                txt_lines.append(line)

        txt_path = output_path / f"{img_path.stem}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(txt_lines))

    print(f"\n--> Done! Annotations saved to: {output_path}")


if __name__ == "__main__":
    run_auto_label()
