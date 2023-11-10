"""
This script:
 1) take LabelBox v2 labels json file,
 2) manipulate the ontology
 3) split the labels into train, val, test.
 4) convert the split labels to yolov8 format.

It generates 3 different types of ontology:
- multi-class labels: labels with all classes except for classes excluded by `--exclude` argument
- gender-only class labels: labels with only 'bbmale' and 'bbfemale'
- single-class labels: labels with only 'bb'
"""
import argparse
import os
from pathlib import Path

import yaml

from utility import (
    count_cls_in_labels,
    remove_cls_in_labels,
    replace_cls_in_labels,
    remove_empty_labels,
    load_lb_labels_json,
    train_val_test_split,
    labelboxv2_to_yolov8,
    save_lb_labels_json,
    lbv2_labels_with_cls,
)

parser = argparse.ArgumentParser()
parser.add_argument('--labels', type=str, required=True,
                    help='path to labelbox labels json file')
parser.add_argument('--exclude', type=str, default=['territory'], nargs='*',
                    help='classes to exclude, separated by space')
parser.add_argument('-f', '--override', action='store_true', help='force to override existing files')
args = parser.parse_args()

# arguments
src_labels_path = Path(args.labels)
cls_to_exclude = args.exclude
override = args.override

# read labelbox config
lb_cfg = yaml.load(Path("labelbox_config.yaml").read_text(), Loader=yaml.FullLoader)
PROJECT_ID = lb_cfg["PROJECT_ID"]
api_key = lb_cfg["LABELBOX_API_KEY"] or os.getenv("LABELBOX_API_KEY")

# load labels
src_labels = load_lb_labels_json(src_labels_path)
classes = list(count_cls_in_labels(labels=src_labels, project_id=PROJECT_ID).keys())
print("Classes in labels: ", classes)

########################################################################################################################
# Making Ontology
########################################################################################################################

# making multi-class labels
mc_labels = remove_cls_in_labels(
    labels=src_labels,
    cls=cls_to_exclude,
    project_id=PROJECT_ID
)
mc_labels = remove_empty_labels(mc_labels, PROJECT_ID)
print('mc_labels: ', count_cls_in_labels(mc_labels, PROJECT_ID, classes + ['bb']))

# making gender-only class labels
gender_cls = ['bbmale', 'bbfemale']
gender_labels = remove_cls_in_labels(
    labels=src_labels,
    cls=[cls for cls in classes if cls not in gender_cls],
    project_id=PROJECT_ID
)
gender_labels = remove_empty_labels(gender_labels, PROJECT_ID)
print('gd_labels: ', count_cls_in_labels(gender_labels, PROJECT_ID, classes + ['bb']))

# making single class labels
sc_labels = replace_cls_in_labels(
    labels=gender_labels,
    cls_map={'bbmale': 'bb', 'bbfemale': 'bb'},
    project_id=PROJECT_ID)
print('sc_labels: ', count_cls_in_labels(sc_labels, PROJECT_ID, classes + ['bb']))

# save altered ontology labels
for labels, labels_name in zip([mc_labels, gender_labels, sc_labels], ['mc_dtc2023', 'gd_dtc2023', 'sc_dtc2023']):
    labels_save_dir = src_labels_path.parent / labels_name
    labels_save_dir.mkdir(parents=True, exist_ok=override)
    save_json_file_path = labels_save_dir / f'{labels_name}.json'
    save_lb_labels_json(save_json_file_path, labels)

########################################################################################################################
# Split Labels
########################################################################################################################

for labels, labels_name, selected_cls in zip(
        # labels as lists of dicts
        [mc_labels, gender_labels, sc_labels],
        # labels names
        ['mc_dtc2023', 'gd_dtc2023', 'sc_dtc2023'],
        # ontologies
        [
            # mutli-class
            [cls for cls in classes if cls not in cls_to_exclude],
            # gender classes
            ['bbmale', 'bbfemale'],
            # single-class
            ['bb']
        ]
):  # TODO: decouple ontology manipulation, splitting, and conversion to yolov8 format

    print(f"Splitting '{labels_name}' with {len(labels)} labels.")

    # sort classes from least to most
    selected_cls = sorted(selected_cls,
                          key=lambda x: count_cls_in_labels(labels=labels, cls=[x], project_id=PROJECT_ID)[x])

    # split labels within each class from least to most, ensure each class has at least one label in each split
    labels_remained = labels
    train, val, test = [], [], []
    for cls in selected_cls:
        if cls == selected_cls[-1]:  # no need to check labels of the last
            labels_with_cls = labels_remained
        else:
            labels_with_cls, labels_remained = lbv2_labels_with_cls(labels=labels_remained, cls=cls,
                                                                    project_id=PROJECT_ID)
        print(f"- Splitting {len(labels_with_cls)} labels with '{cls}', {len(labels_remained)} labels remaining.")
        _train, _val, _test = train_val_test_split(labels=labels_with_cls,
                                                   train_size=0.7, val_size=0.15, test_size=0.15,
                                                   random_seed=42)
        train.extend(_train)
        val.extend(_val)
        test.extend(_test)

    print(f"Splitted {labels_name}: Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # save splitted labels in LabelBox v2 format
    labels_dir = src_labels_path.parent / labels_name
    labels_dir.mkdir(parents=True, exist_ok=override)
    split_path_map = {}  # for converting to yolov8 format
    for split, split_labels in zip(['train', 'val', 'test'], [train, val, test]):
        split_labels_path = labels_dir / f"{labels_name}_{split}.json"
        split_path_map[split] = split_labels_path
        save_lb_labels_json(split_labels_path, split_labels)

    # convert to yolov8 format
    labelboxv2_to_yolov8(split_path_map, labels_name, selected_cls, 'id', True)
