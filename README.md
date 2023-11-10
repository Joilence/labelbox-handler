# LabelBox Handler

This repo hosts scripts that download annotations from LabelBox and convert into format in YOLOv8, COCO, etc.

## Install dependencies

```bash
pip install -r requirements.txt
```

## Preparation

1. Get LabelBox API key from [LabelBox](https://app.labelbox.com/settings/account/api-keys).
2. Find LabelBox Project ID from [LabelBox](https://app.labelbox.com/projects).
3. Put API key and Project ID into `./labelbxo_config.yaml` file.

## From LabelBox to YOLOv8

1. Download labels json file from LabelBox:
    ```bash
    python3 download_labels_json_from_lbv2_project.py
    ```
   Labels will be saved in `labelbox_labels/{project_name}/{project_name}.json`

2. Create new ontology, split and convert labels json file into YOLOv8 format
    ```bash
    python3 filter_split_convert_lbv2_to_yolov8.py --labels labelbox_labels/{prject_name}/{prject_name}.json
    ```
   Converted labels will be saved in `yolov8_labels`.