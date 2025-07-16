import os
import torch
import glob
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
import time
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class YoloDataset(Dataset):
    def __init__(self, image_dir, label_dir, transforms=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

        print(f"📂 Found {len(self.image_files)} images in {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = os.path.join(self.label_dir, os.path.basename(img_path).replace(".jpg", ".txt"))

        # Load image
        image = Image.open(img_path).convert("RGB")
        w, h = image.size  # Get image dimensions

        # Load labels
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    data = list(map(float, line.split()))
                    class_id, x_center, y_center, width, height = data

                    # Convert from YOLO to absolute pixel values
                    x_min = (x_center - width / 2) * w
                    y_min = (y_center - height / 2) * h
                    x_max = (x_center + width / 2) * w
                    y_max = (y_center + height / 2) * h

                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(int(class_id) + 1)  # Class IDs start from 1 (0 is background)

        # 🔹 Instead of recursion, return an empty target if no annotations
        if len(boxes) == 0:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),  # Empty tensor
                "labels": torch.zeros((0,), dtype=torch.int64),  # Empty tensor
            }
        else:
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
            }

        if self.transforms:
            image = self.transforms(image)

        return image, target

# Define transformations
transform = T.Compose([T.ToTensor()])

training_images_path = "/fml_version2/full_dataset/images/train"
training_labels_path = "/fml_version2/full_dataset/labels/yolo_format/train"
validation_images_path = "/fml_version2/full_dataset/images/val"
validation_labels_path = "/fml_version2/full_dataset/labels/yolo_format/val"

# Create datasets
train_dataset = YoloDataset(training_images_path, training_labels_path, transform)
val_dataset = YoloDataset(validation_images_path,validation_labels_path, transform)

# DataLoader (collate function to handle variable-sized batches)
def collate_fn(batch):
    return tuple(zip(*batch))

batch_size = 2

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Load pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

# Modify classifier head
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def load_ground_truths(image_dir, label_dir):
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


# Training function with progress print
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0
    print(f"\n Epoch {epoch+1}: Training started...")

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        if batch_idx % 10 == 0:  # Print every 10 batches
            print(f"   Batch {batch_idx}/{len(data_loader)} - Loss: {losses.item():.4f}")

    avg_loss = total_loss / len(data_loader)
    print(f"✅ Epoch {epoch+1}: Training finished. Avg Loss: {avg_loss:.4f}")
    return avg_loss


def predict(image_paths, image_ids):
    """Run inference and return detections in COCO format for a batch of images."""
    model.eval()  # 🔹 Set to evaluation mode before inference

    images = []
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        images.append(F.to_tensor(image).to(device))

    images_tensor = list(images)  # Keep it as a list, Faster R-CNN expects a list of tensors

    with torch.no_grad():  # 🔹 Disable gradient computation for inference
        predictions = model(images_tensor)

    results = []
    for i, pred in enumerate(predictions):
        for j in range(len(pred["boxes"])):
            bbox = pred["boxes"][j].cpu().numpy()
            score = float(pred["scores"][j].item())
            category_id = int(pred["labels"][j].item())

            if score > 0.5:  # Only include confident detections
                x_min, y_min, x_max, y_max = map(float, bbox)
                width = float(x_max - x_min)
                height = float(y_max - y_min)

                results.append({
                    "image_id": image_ids[i],
                    "category_id": category_id,
                    "bbox": [x_min, y_min, width, height],
                    "score": score
                })

    model.train()  # 🔹 Set back to training mode after inference
    return results


def evaluate(batch_size,image_dir, label_dir):
    """Evaluate model performance using COCO metrics."""
    ground_truths = load_ground_truths(image_dir,label_dir)
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
    return coco_eval.stats[0]  # Return mAP


# Training loop
best_map = 0
patience = 30
wait = 0
num_epochs = 100
start_time = time.time()

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
    print(f"\n📊 Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")
    current_map = evaluate(batch_size, validation_images_path, validation_labels_path)
    print(f"Epoch {epoch + 1}: Validation mAP = {current_map:.4f}")

    if current_map > best_map:
        best_map = current_map
        wait = 0
        # Save model checkpoint
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        checkpoint_path = f"/fasterrcnn/fasterrcnn_epoch_{epoch + 1}_{timestamp}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 Model saved: {checkpoint_path}")
    else:
        wait += 1
        print(f"⏳ No improvement for {wait}/{patience} epochs.")

    if wait >= patience:
        print("🛑 Early stopping triggered.")
        break

end_time = time.time()
print(f"\n🎉 Training Complete! Total Time: {((end_time - start_time) / 60):.2f} minutes")
