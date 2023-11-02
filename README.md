# LabelBox Handler

This repo hosts scripts that download annotations from LabelBox and convert into format in YOLOv8, COCO, etc.

## Install dependencies

```bash
pip install -r requirements.txt
```

## Preparation

1. Get LabelBox API key from [LabelBox](https://app.labelbox.com/settings/account/api-keys).
2. Find LabelBox Project ID from [LabelBox](https://app.labelbox.com/projects).
3. Put API key and Project ID into `labelbxo_config.yaml` file.

## From LabelBox to YOLOv8

1. Download labels json file from LabelBox:
    ```bash
    python3 download_labels_json_from_lbv2_project.py
    ```
   Labels will be saved in `labelbox_labels/{project_name}/labels.json`, and will be split into train, val, and test
   sets (0.7, 0.15, 0.15) as `labelbox_labels/{project_name}/labels_{train/val/test}.json`.
2. Convert labels json file into YOLOv8 format
    ```bash
    python3 lbv2_to_yolov8.py
    ```
   Labels will be saved in `yolov8_labels/{project_name}/{train/val/test}/labels/{image_id}.txt`, and images will be
   saved in `yolov8_labels/{project_name}/{train/val/test}/images/{image_id}.jpg`