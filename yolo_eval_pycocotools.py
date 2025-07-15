import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.transforms import functional as F
from ultralytics import YOLO

# Define paths
image_dir = "/fml_v2/full_dataset/images/test"
label_dir = "/fml_v2/full_dataset/labels/yolo_format/test"
model_path = "/models/yolov8s_fml.pt"

# Load YOLO model
model = YOLO(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_ground_truths(label_dir):
    """Convert YOLO-style labels to COCO format."""
    gt_annotations = []
    gt_images = []
    annotation_id = 1

    for idx, filename in enumerate(sorted(os.listdir(label_dir))):
        if not filename.endswith(".txt"):
            continue

        image_filename = filename.replace(".txt", ".jpg")
        img_path = os.path.join(image_dir, image_filename)
        if not os.path.exists(img_path):
            image_filename = filename.replace(".txt", ".png")
            img_path = os.path.join(image_dir, image_filename)

        if not os.path.exists(img_path):
            print(f"Warning: Image file {image_filename} not found!")
            continue

        img = Image.open(img_path)
        width, height = img.size
        image_id = idx + 1
        gt_images.append({
            "id": image_id,
            "file_name": image_filename,
            "width": width,
            "height": height
        })

        with open(os.path.join(label_dir, filename), "r") as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0]) + 1
                x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])

                x_min = (x_center - bbox_width / 2) * width
                y_min = (y_center - bbox_height / 2) * height
                bbox_w = bbox_width * width
                bbox_h = bbox_height * height

                gt_annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [x_min, y_min, bbox_w, bbox_h],
                    "area": bbox_w * bbox_h,
                    "iscrowd": 0
                })
                annotation_id += 1

    categories = [{"id": 1, "name": "garbage"}]
    gt_coco_format = {"images": gt_images, "annotations": gt_annotations, "categories": categories}
    with open("ground_truths.json", "w") as f:
        json.dump(gt_coco_format, f)
    return gt_coco_format


def predict(image_paths, image_ids):
    """Run YOLO inference and return detections in COCO format."""
    results = []
    predictions = model(image_paths, conf=0.5)
    for i, pred in enumerate(predictions):
        for j in range(len(pred.boxes.xyxy)):
            bbox = pred.boxes.xyxy[j].cpu().numpy()
            score = float(pred.boxes.conf[j].item())
            category_id = int(pred.boxes.cls[j].item()) + 1
            if score > 0.5:
                x_min, y_min, x_max, y_max = map(float, bbox)
                width, height = x_max - x_min, y_max - y_min
                results.append({
                    "image_id": image_ids[i],
                    "category_id": category_id,
                    "bbox": [x_min, y_min, width, height],
                    "score": score
                })
    return results


def evaluate(batch_size=16):
    """Evaluate YOLO model using COCO metrics."""
    ground_truths = load_ground_truths(label_dir)
    coco_gt = COCO("ground_truths.json")

    predictions = []
    image_paths = []
    image_ids = []
    for idx, img in enumerate(ground_truths["images"]):
        img_path = os.path.join(image_dir, img["file_name"])
        image_paths.append(img_path)
        image_ids.append(img["id"])
        if len(image_paths) == batch_size or idx == len(ground_truths["images"]) - 1:
            preds = predict(image_paths, image_ids)
            predictions.extend(preds)
            image_paths, image_ids = [], []
            print(f"Processed {idx + 1}/{len(ground_truths['images'])} images...")

    with open("predictions.json", "w") as f:
        json.dump(predictions, f)

    coco_dt = coco_gt.loadRes("predictions.json")
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    precisions = coco_eval.eval["precision"]
    recalls = np.linspace(0.0, 1.0, precisions.shape[1])
    plt.figure(figsize=(10, 6))
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    colors = plt.cm.viridis(np.linspace(0, 1, len(iou_thresholds)))

    for i, iou in enumerate(iou_thresholds):
        plt.plot(recalls, precisions[i, :, 0, 0, 2], label=f'IoU={iou:.2f}', color=colors[i])

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid()
    plt.show()

    avg_precisions = np.mean(precisions[:, :, 0, 0, 2], axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, avg_precisions, label="Averaged PR Curve", color="b")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Averaged over IoU & Area)")
    plt.legend()
    plt.grid()
    plt.show()


# Run evaluation
evaluate(batch_size=1)
