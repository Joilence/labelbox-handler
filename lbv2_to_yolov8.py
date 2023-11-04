import argparse
import os
from pathlib import Path

import labelbox as lb
import yaml

from utility import get_ontology, labelboxv2_to_yolov8

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

labelboxv2_to_yolov8(split_path_map, project.name, classes, filename_key, True)
