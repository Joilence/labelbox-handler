import os
from pathlib import Path

import labelbox as lb
import yaml

from utility import (
    count_cls_in_labels,
    remove_cls_in_labels,
    replace_cls_in_labels,
    remove_empty_labels,
    get_ontology,
    load_lb_labels_json,
    split_lbv2_labels,
    labelboxv2_to_yolov8,
)

labels_path = 'labelbox_labels/Detection_2023/labels.json'

# read labelbox config
lb_cfg = yaml.load(Path("labelbox_config.yaml").read_text(), Loader=yaml.FullLoader)
PROJECT_ID = lb_cfg["PROJECT_ID"]
api_key = lb_cfg["LABELBOX_API_KEY"] or os.getenv("LABELBOX_API_KEY")

# get classes from ontology
client = lb.Client(api_key=api_key)
project = client.get_project(PROJECT_ID)
print(f"Fetching classes from ontology of \"{project.name}\" ({PROJECT_ID})...")
_, classes = get_ontology(client, PROJECT_ID)
print(f"Classes fetched: {classes}")

# load labels
labels = load_lb_labels_json(labels_path)

# making multi-class labels
mc_labels = remove_cls_in_labels(
    labels=labels,
    cls=['territory', 'unknown'],
    project_id=PROJECT_ID
)
mc_labels = remove_empty_labels(mc_labels, PROJECT_ID)
print(count_cls_in_labels(mc_labels, classes + ['bb'], PROJECT_ID))

# making gender-only class labels
gender_cls = ['bbmale', 'bbfemale']
gender_labels = remove_cls_in_labels(
    labels=labels,
    cls=[cls for cls in classes if cls not in gender_cls],
    project_id=PROJECT_ID
)
gender_labels = remove_empty_labels(gender_labels, PROJECT_ID)
print(count_cls_in_labels(gender_labels, classes + ['bb'], PROJECT_ID))

# making single class labels
sc_labels = replace_cls_in_labels(
    labels=gender_labels,
    cls_map={'bbmale': 'bb', 'bbfemale': 'bb'},
    project_id=PROJECT_ID)
sc_labels = remove_empty_labels(sc_labels, PROJECT_ID)
print(count_cls_in_labels(sc_labels, classes + ['bb'], PROJECT_ID))

for labels, labels_name, selected_cls in zip(
        [mc_labels, gender_labels, sc_labels],
        ['mc_labels', 'gd_labels', 'sc_labels'],
        [
            # mutli-class
            [cls for cls in classes if cls not in ['territory', 'unknown']],
            # gender classes
            ['bbmale', 'bbfemale'],
            # single-class
            ['bb']
        ]
):
    print(f"Splitting {labels_name}...")
    split_lbv2_labels(labels, labels_name, f'labelbox_labels/Detection_2023/{labels_name}', True)

    split_path_map = {
        split: f"labelbox_labels/Detection_2023/{labels_name}/{labels_name}_{split}.json"
        for split in ["train", "val", "test"]
    }

    labelboxv2_to_yolov8(split_path_map, labels_name, selected_cls, 'id', True)
