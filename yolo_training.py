from ultralytics import YOLO
import torch


model = YOLO("yolov8s.pt")

yaml_filename = "train.yaml"

results = model.train(data=yaml_filename, epochs=200, patience=30, augment=True, name='bts_200_epochs_30_patience_1024_yolov8s', pretrained=True, batch=8, imgsz=(1024, 1024), device=device)

