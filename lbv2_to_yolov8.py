import argparse
import os
from pathlib import Path

import labelbox as lb
import yaml
from labelformat.formats import LabelboxObjectDetectionInput
from labelformat.model.bounding_box import BoundingBoxFormat

from utility import get_ontology, load_lb_labels_json, download_images_from_lbv2

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--fn_key", type=str, default="id", choices=["global_key", "external_id", "id"],
                    help="filename key, default to \"id\"")
args = parser.parse_args()
filename_key = args.fn_key
override = True  # override existing files except for images

# load labelbox config
lb_cfg = yaml.load(Path("labelbox_config.yaml").read_text(), Loader=yaml.FullLoader)
PROJECT_ID = lb_cfg["PROJECT_ID"]
api_key = lb_cfg["LABELBOX_API_KEY"] or os.getenv("LABELBOX_API_KEY")

# get classes from ontology
client = lb.Client(api_key=api_key)
project = client.get_project(PROJECT_ID)
print(f"Fetching classes from ontology of \"{project.name}\" ({PROJECT_ID})...")
_, classes = get_ontology(client, PROJECT_ID)
print(f"Classes fetched: {classes}")

labels_name = 'labels'
split_path_map = {
    split: f"labelbox_labels/{project.name}/{labels_name}_{split}.json"
    for split in ["train", "val", "test"]
}

# Create a YOLOv8 compatible label file
save_dir = "yolov8_labels"
dest_dir = Path(save_dir) / project.name
dest_dir.mkdir(parents=True, exist_ok=override)

# set up data dicts for dataset yaml file
data_dicts = {
    "names": dict(enumerate(classes)),
    "nc": len(classes),
    "path": ".",
}

for split, labels_path in split_path_map.items():
    print(f"Processing {split} split...")

    # set up directories
    labels_dir = dest_dir / split / "labels"
    data_dicts[split] = f"{split}/images"
    images_dir = dest_dir / f"{split}/images"

    # load labelbox labels json
    label_input = LabelboxObjectDetectionInput(
        input_file=Path(labels_path),
        category_names=','.join(classes),
    )

    # convert labelbox labels json into yolov8 label files
    for label in label_input.get_labels(filename_key=filename_key):
        label_path = (labels_dir / label.image.filename).with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=override)
        with label_path.open("w") as file:
            for obj in label.objects:
                cx, cy, w, h = obj.box.to_format(format=BoundingBoxFormat.CXCYWH)
                rcx = cx / label.image.width
                rcy = cy / label.image.height
                rw = w / label.image.width
                rh = h / label.image.height
                file.write(f"{obj.category.id} {rcx} {rcy} {rw} {rh}\n")

    # download images
    download_images_from_lbv2(
        labels=load_lb_labels_json(labels_path),
        filename_key=filename_key,
        dest_dir=images_dir,
        override=True
    )

# save data dicts to yaml
with open(Path(dest_dir) / "data.yaml", "w") as f:
    yaml.dump(data_dicts, f)
