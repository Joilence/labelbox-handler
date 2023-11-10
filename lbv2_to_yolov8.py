import argparse
from pathlib import Path

import yaml

from utility import labelboxv2_to_yolov8, load_lb_labels_json, count_cls_in_labels

parser = argparse.ArgumentParser()
parser.add_argument("--label_dir", type=str, required=True,
                    help="dir where labelbox labels json files are stored")
parser.add_argument("-k", "--fn_key", type=str, default="id", choices=["global_key", "external_id", "id"],
                    help="filename key, default to \"id\"")
args = parser.parse_args()
label_dir = Path(args.label_dir)
filename_key = args.fn_key
override = True  # override existing files except for images

# load labelbox config
lb_cfg = yaml.load(Path("labelbox_config.yaml").read_text(), Loader=yaml.FullLoader)
PROJECT_ID = lb_cfg["PROJECT_ID"]

# load labels
labels = load_lb_labels_json(label_dir / f"{label_dir.stem}.json")
classes = list(count_cls_in_labels(labels=labels, project_id=PROJECT_ID).keys())
print("Classes in labels: ", classes)

label_name = label_dir.stem
split_path_map = {
    split: label_dir / f"{label_name}_{split}.json"
    for split in ["train", "val", "test"]
}

labelboxv2_to_yolov8(split_path_map, label_name, classes, filename_key, True)
