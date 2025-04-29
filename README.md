# Floating Marine Litter (FML) Dataset

The Floating Marine Litter (FML) Dataset comprises approximately 5500 floating marine litter optical images and 14000 bounding box single-class "garbage" annotations, tailored for the object detection task. The images were taken from a first-person perspective using a camera mounted on an Unmanned Surface Vehicle (USV). The images were collected in diverse lighting conditions and camera setups across inland and coastal water bodies. The dataset is published on the SEA scieNtific Open data Edition SEANOE repository (https://www.seanoe.org/) and it is available with the following DOI https://doi.org/10.17882/106148. The images are all in 1920 x 1080 resolution and .jpg format.

![folder_structure](https://github.com/user-attachments/assets/6320dc76-f88c-43dc-883d-6b5710ba608a)

The figure reports the folder structure of the dataset: images are divided into the train, val, and test folders following the 70-20-10 dataset split. Annotations are provided both in .txt (YOLO standard) and JSON (COCO standard) formats. The YOLO labels are contained inside the train, val, and test label folders and have a 1 to 1 correspondence with the images (they have the same filename as the image, but the extension is .txt). The COCO labels are condensed in three .json files train.jason, val.jason, and test.json.


# Training Models

To train the models run the yolo_training.py and fasterrcnn_training.py scripts.

# Evaluate Models

To evaluate the models on a custom dataset run the yolo_eval_pycocotools.py and fasterrcnn_eval.py.fasterrcnn_eval.py
The trained models can be downloaded at the following link: https://drive.google.com/drive/folders/1QTGO9rDWCKzvo5DlphT6a4IzraDRwJ4o?usp=drive_link

