import os
import torch
import torchvision
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from tqdm import tqdm  # To show progress bar

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths
image_dir = "/fml_version2/full_dataset/images/test"
label_dir = "/fml_version2/full_dataset/labels/yolo_format/test"

# Load trained model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
model.load_state_dict(torch.load(
    "/models/fasterrcnn_fml.pth",
    map_location=device))
model.to(device)
model.eval()


def load_ground_truths(label_dir):
    """Convert YOLO-style labels to COCO format."""
    gt_annotations = []
    gt_images = []
    annotation_id = 1

    for idx, filename in enumerate(sorted(os.listdir(label_dir))):
        if not filename.endswith(".txt"):
            continue

        # Check if corresponding image is .jpg, if not, replace with .png
        image_filename = filename.replace(".txt", ".jpg")
        img_path = os.path.join(image_dir, image_filename)

        # If .jpg doesn't exist, replace .txt with .png
        if not os.path.exists(img_path):
            image_filename = filename.replace(".txt", ".png")
            img_path = os.path.join(image_dir, image_filename)

        if not os.path.exists(img_path):
            print(f"Warning: Image file {image_filename} not found!")
            continue

        # Get image size
        img = Image.open(img_path)
        width, height = img.size

        image_id = idx + 1  # Unique image ID
        gt_images.append({
            "id": image_id,
            "file_name": image_filename,
            "width": width,
            "height": height
        })

        # Read and parse the ground truth annotations
        with open(os.path.join(label_dir, filename), "r") as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0]) + 1  # YOLO format uses 0-based, COCO uses 1-based
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

    gt_coco_format = {
        "images": gt_images,
        "annotations": gt_annotations,
        "categories": categories
    }

    with open("ground_truths.json", "w") as f:
        json.dump(gt_coco_format, f)

    return gt_coco_format


def predict(image_paths, image_ids):
    """Run inference and return detections in COCO format for a batch of images."""
    images = []
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        images.append(F.to_tensor(image).to(device))

    images_tensor = torch.stack(images).to(device)

    with torch.no_grad():
        predictions = model(images_tensor)

    results = []
    for i, pred in enumerate(predictions):
        for j in range(len(pred["boxes"])):
            bbox = pred["boxes"][j].cpu().numpy()  # Convert to numpy array
            score = float(pred["scores"][j].item())  # Convert to native Python float
            category_id = int(pred["labels"][j].item())  # Ensure category_id is an int

            if score > 0.5:  # Only include confident detections
                # Convert bbox values from numpy.float32 to native Python float
                x_min, y_min, x_max, y_max = map(float, bbox)
                width = float(x_max - x_min)  # Ensure width is a float
                height = float(y_max - y_min)  # Ensure height is a float

                results.append({
                    "image_id": image_ids[i],  # Use the passed image_id
                    "category_id": category_id,
                    "bbox": [x_min, y_min, width, height],
                    "score": score
                })

    return results


def evaluate(batch_size=16):
    """Evaluate model performance using COCO metrics."""
    ground_truths = load_ground_truths(label_dir)
    coco_gt = COCO("ground_truths.json")

    # Generate predictions
    predictions = []

    # Batch processing
    image_paths = []
    image_ids = []
    for idx, img in enumerate(ground_truths["images"]):
        img_path = os.path.join(image_dir, img["file_name"])
        image_paths.append(img_path)
        image_ids.append(idx + 1)  # Unique ID based on index

        if len(image_paths) == batch_size or idx == len(ground_truths["images"]) - 1:
            preds = predict(image_paths, image_ids)  # Pass the batch
            predictions.extend(preds)
            image_paths = []  # Reset for next batch
            image_ids = []

            # Print progress
            print(f"Processed {idx + 1}/{len(ground_truths['images'])} images...")

    # Save predictions in COCO format
    with open("predictions.json", "w") as f:
        json.dump(predictions, f)

    coco_dt = coco_gt.loadRes("predictions.json")
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract Precision-Recall Data Correctly
    precisions = coco_eval.eval["precision"]  # Shape: (10, 101, 1, 4, 3)

    # Generate recall values (evenly spaced)
    recalls = np.linspace(0.0, 1.0, precisions.shape[1])

    plt.figure(figsize=(10, 6))

    # Plot each IoU threshold separately
    iou_thresholds = np.linspace(0.5, 0.95, 10)  # Standard COCO IoU thresholds
    colors = plt.cm.viridis(np.linspace(0, 1, len(iou_thresholds)))  # Generate colors

    for i, iou in enumerate(iou_thresholds):
        plt.plot(
            recalls, precisions[i, :, 0, 0, 2],
            label=f'IoU={iou:.2f}',
            color=colors[i]
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="best")
    plt.grid(True)

    plt.show()


    # Averaging over IoU thresholds (axis=0) and area thresholds (axis=3)
    avg_precisions = np.mean(precisions[:, :, 0, 0, 2], axis=0)  # Shape: (101,)

    # Generate recall values (evenly spaced)
    recalls = np.linspace(0.0, 1.0, precisions.shape[1])

    # Plot the averaged PR curve
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, avg_precisions, label="Averaged PR Curve", color="b")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Averaged over IoU & Area)")
    plt.legend(loc="best")
    plt.grid(True)

    plt.show()


# Run evaluation
evaluate(batch_size=1)
