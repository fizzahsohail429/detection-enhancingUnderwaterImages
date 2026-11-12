"""Train YOLOv8n for 50 epochs and save the validation table, PSNR/SSIM, and figures.

ResNet50 is not connected to this detector. This script records that separation.
"""
import json
import os
import platform
import sys

import cv2
import numpy as np
import torch
import ultralytics
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = r"C:\Users\Lenovo\underwater_dataset1\aquarium_pretrain\data.yaml"
TRAIN_IMAGES = r"C:\Users\Lenovo\underwater_dataset1\aquarium_pretrain\train\images"
VAL_IMAGES = r"C:\Users\Lenovo\underwater_dataset1\aquarium_pretrain\valid\images"
VAL_LABELS = r"C:\Users\Lenovo\underwater_dataset1\aquarium_pretrain\valid\labels"
OUT = os.path.join(ROOT, "final_experiment")
os.makedirs(OUT, exist_ok=True)

NAMES = ["fish", "jellyfish", "penguin", "puffin", "shark", "starfish", "stingray"]


def save_json(name, payload):
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"WROTE {path}")


def environment():
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "ultralytics": ultralytics.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }
    try:
        import skimage
        info["skimage"] = skimage.__version__
    except Exception as exc:
        info["skimage"] = str(exc)
    save_json("environment.json", info)
    print("ENVIRONMENT", info)
    return info


def table_i():
    psnr_values = []
    ssim_values = []
    n = 0
    for filename in os.listdir(TRAIN_IMAGES):
        original = cv2.imread(os.path.join(TRAIN_IMAGES, filename))
        if original is None:
            continue
        smoothed = cv2.GaussianBlur(original, (5, 5), 0)
        original_r = cv2.resize(original, (256, 256), interpolation=cv2.INTER_LINEAR)
        smoothed_r = cv2.resize(smoothed, (256, 256), interpolation=cv2.INTER_LINEAR)
        psnr_values.append(float(psnr(original_r, smoothed_r, data_range=255)))
        ssim_values.append(float(ssim(original_r, smoothed_r, channel_axis=-1, win_size=7, data_range=255)))
        n += 1
    result = {
        "images": n,
        "partition": "train",
        "reference": "original training images",
        "processed": "cv2.GaussianBlur kernel (5, 5), sigma argument 0",
        "resize": "256x256 INTER_LINEAR",
        "psnr_mean": float(np.mean(psnr_values)),
        "ssim_mean": float(np.mean(ssim_values)),
        "psnr_formatted": f"{np.mean(psnr_values):.2f}",
        "ssim_formatted": f"{np.mean(ssim_values):.4f}",
        "data_range": 255,
        "ssim_window": 7,
        "channels": "BGR color, channel_axis=-1",
        "used_by_detector": False,
    }
    save_json("table_i.json", result)
    print("TABLE_I", result["psnr_formatted"], result["ssim_formatted"], "n", n)
    return result


def draw_ground_truth(n_images=6):
    files = sorted(f for f in os.listdir(VAL_IMAGES) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    chosen = files[:n_images]
    tiles = []
    for filename in chosen:
        image = cv2.imread(os.path.join(VAL_IMAGES, filename))
        label_path = os.path.join(VAL_LABELS, os.path.splitext(filename)[0] + ".txt")
        h, w = image.shape[:2]
        if os.path.exists(label_path):
            with open(label_path, encoding="utf-8") as handle:
                for line in handle:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    class_id, x, y, bw, bh = map(float, parts[:5])
                    x1 = int((x - bw / 2) * w)
                    y1 = int((y - bh / 2) * h)
                    x2 = int((x + bw / 2) * w)
                    y2 = int((y + bh / 2) * h)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, NAMES[int(class_id)], (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(cv2.resize(image, (320, 320)))
    sheet = np.hstack(tiles)
    path = os.path.join(OUT, "figure1_validation_ground_truth.png")
    cv2.imwrite(path, sheet)
    print("WROTE", path)
    return path


def train_and_validate():
    model = YOLO("yolov8n.pt")
    model.train(
        data=DATA,
        epochs=50,
        imgsz=640,
        batch=8,
        device="cpu",
        workers=2,
        pretrained=True,
        patience=100,
        seed=42,
        project=OUT,
        name="yolov8n_50e",
        exist_ok=True,
        plots=True,
        cache=False,
    )
    best = os.path.join(OUT, "yolov8n_50e", "weights", "best.pt")
    print("BEST", best, "exists", os.path.exists(best))
    evaluated = YOLO(best)
    metrics = evaluated.val(data=DATA, split="val", imgsz=640, batch=8, device="cpu", plots=True, project=OUT, name="val_best")
    per_class = []
    precision = np.asarray(metrics.box.p).tolist()
    recall = np.asarray(metrics.box.r).tolist()
    map50 = np.asarray(metrics.box.ap50).tolist()
    map5095 = np.asarray(metrics.box.ap).tolist()
    for i, name in enumerate(NAMES):
        per_class.append({
            "class": name,
            "precision": precision[i] if i < len(precision) else None,
            "recall": recall[i] if i < len(recall) else None,
            "map50": map50[i] if i < len(map50) else None,
            "map50_95": map5095[i] if i < len(map5095) else None,
        })
    summary = {
        "checkpoint": best,
        "split": "val",
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class": per_class,
        "results_dict": {k: float(v) for k, v in metrics.results_dict.items()},
    }
    save_json("table_ii.json", summary)
    print("TABLE_II", summary["precision"], summary["recall"], summary["map50"], summary["map50_95"])

    sample = sorted(f for f in os.listdir(VAL_IMAGES) if f.lower().endswith((".jpg", ".jpeg", ".png")))[:6]
    evaluated.predict(
        source=[os.path.join(VAL_IMAGES, f) for f in sample],
        imgsz=640,
        device="cpu",
        save=True,
        project=OUT,
        name="figure3_validation_predictions",
        exist_ok=True,
    )
    return summary


if __name__ == "__main__":
    print("PYTHON", sys.version)
    environment()
    table_i()
    draw_ground_truth()
    train_and_validate()
    print("FINAL_EXPERIMENT_DONE")
