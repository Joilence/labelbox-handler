"""
This script trys to manipulate the labelbox labels based on different classes to make it suitable for training.
It generates 3 different types of labels:
- multi-class labels: labels with all classes except 'territory' and 'unknown'
- gender-only class labels: labels with only 'bbmale' and 'bbfemale'
- single-class labels: labels with only 'bb'
"""

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
    train_val_test_split,
    labelboxv2_to_yolov8,
    save_lb_labels_json,
)

# arguments
labels_path = 'labelbox_labels/Detection_2023/labels.json'
cls_to_exclude = ['territory']
override = True

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

########################################################################################################################
# Making Ontology ######################################################################################################
########################################################################################################################

# making multi-class labels
mc_labels = remove_cls_in_labels(
    labels=labels,
    cls=cls_to_exclude,
    project_id=PROJECT_ID
)
mc_labels = remove_empty_labels(mc_labels, PROJECT_ID)
print('mc_labels: ', count_cls_in_labels(mc_labels, classes + ['bb'], PROJECT_ID))

# making gender-only class labels
gender_cls = ['bbmale', 'bbfemale']
gender_labels = remove_cls_in_labels(
    labels=labels,
    cls=[cls for cls in classes if cls not in gender_cls],
    project_id=PROJECT_ID
)
gender_labels = remove_empty_labels(gender_labels, PROJECT_ID)
print('gd_labels: ', count_cls_in_labels(gender_labels, classes + ['bb'], PROJECT_ID))

# making single class labels
sc_labels = replace_cls_in_labels(
    labels=gender_labels,
    cls_map={'bbmale': 'bb', 'bbfemale': 'bb'},
    project_id=PROJECT_ID)
sc_labels = remove_empty_labels(sc_labels, PROJECT_ID)
print('sc_labels: ', count_cls_in_labels(sc_labels, classes + ['bb'], PROJECT_ID))

########################################################################################################################
# Split Labels
########################################################################################################################

for labels, labels_name, selected_cls in zip(
        [mc_labels, gender_labels, sc_labels],
        ['mc_dtc2023', 'gd_dtc2023', 'sc_dtc2023'],
        [
            # mutli-class
            [cls for cls in classes if cls not in cls_to_exclude],
            # gender classes
            ['bbmale', 'bbfemale'],
            # single-class
            ['bb']
        ]
):
    print(f"Splitting {labels_name}...")
    train, val, test = train_val_test_split(labels)
    print(f"{labels_name}: Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    label_dir = Path(f'labelbox_labels/Detection_2023/{labels_name}')
    label_dir.mkdir(parents=True, exist_ok=override)
    for split, split_labels in zip(['train', 'val', 'test'], [train, val, test]):
        save_lb_labels_json(label_dir / f"{labels_name}_{split}.json", split_labels)

    split_path_map = {
        split: f"labelbox_labels/Detection_2023/{labels_name}/{labels_name}_{split}.json"
        for split in ["train", "val", "test"]
    }

    labelboxv2_to_yolov8(split_path_map, labels_name, selected_cls, 'id', True)
