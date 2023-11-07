import json
import os
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Tuple, List, Union, Dict

import labelbox as lb
import requests
import yaml
from labelformat.formats import LabelboxObjectDetectionInput
from labelformat.model.bounding_box import BoundingBoxFormat
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


def get_ontology(client: lb.Client, project_id: str):
    """get project ontology from labelbox
    :param client: labelbox client
    :param project_id: project id
    :return: ontology and classes
    """

    response = client.execute(
        """
                query getOntology (
                    $project_id : ID!){
                    project (where: { id: $project_id }) {
                        ontology {
                            normalized
                        }
                    }
                }
                """,
        {"project_id": project_id},
    )

    ontology = response["project"]["ontology"]["normalized"]["tools"]

    # Return list of tools and embed category id to be used to map classname during training and inference
    mapped_ontology = []
    thing_classes = []
    for i, item in enumerate(ontology):
        item.update({"category": i})
        mapped_ontology.append(item)
        thing_classes.append(item["name"])

    return mapped_ontology, thing_classes


def download_files(filemap: Tuple[str, str]) -> str:
    """Generic data download function
    :param filemap: tuple of file path and url
    """
    path, uri = filemap
    # Download data
    if not os.path.exists(path):
        r = requests.get(uri, stream=True)
        if r.status_code == 200:
            with open(path, "wb") as f:
                for chunk in r:
                    f.write(chunk)
    return path


def download_images_from_lbv2(
        labels: List[dict],
        filename_key: str,
        dest_dir: Union[str, Path],
        n_threads: int = 8,
        override: bool = False
):
    """Download images from URLs in labels to the destination directory
    :param labels: list of labels from labelbox export API v2
    :param filename_key: key to use for filename, must be one of ['id', 'global_key', 'external_id']
    :param dest_dir: destination directory to save images
    :param n_threads: number of threads to use
    :param override: override existing files
    """
    dest_dir = Path(dest_dir) if isinstance(dest_dir, str) else dest_dir
    dest_dir.mkdir(parents=True, exist_ok=override)

    filename_key_options = ['id', 'global_key', 'external_id']
    assert filename_key in filename_key_options, f"filename_key must be one of {filename_key_options}"

    # use download url to determine image extension
    url = labels[0]['data_row']["row_data"]
    img_ext_options = ['.jpg', '.png']
    img_ext = next((ext for ext in img_ext_options if ext in url.lower()), None)
    assert img_ext, f"Could not determine image extension from url: {url}"

    fp_url_maps = [
        (
            # file save path
            dest_dir / f"{label['data_row'][filename_key]}{img_ext}",
            # file download url
            label['data_row']["row_data"]
        )
        for label in labels
    ]
    results = ThreadPool(n_threads).imap_unordered(download_files, fp_url_maps)
    for _ in tqdm(results, desc=f"Downloading images to {dest_dir}", total=len(fp_url_maps)):
        pass


def load_lb_labels_json(filepath: Union[str, Path]):
    """Load labels from a LabelBox json file, with each line as a json object
    :param filepath: path to json file
    :return: list of labels
    """
    labels = []
    with open(filepath, "r") as f:
        labels.extend(json.loads(line) for line in f)
    return labels


def save_lb_labels_json(filepath: Union[str, Path], labels: List[dict]):
    """ Save labels to a LabelBox json file, with each line as a json object
    :param filepath: path to json file
    :param labels: list of LabelBox v2 labels
    """
    text = '\n'.join(json.dumps(label) for label in labels)
    with open(filepath, 'w') as file:
        file.write(text)


def cls_stat_in_labels(labels: List[dict], cls: Union[List[str], str], project_id: str):
    """print class statistics in LabelBox v2 labels
    :param labels: a list of LabelBox v2 labels
    :param cls: a list of classes to count
    :param project_id: project id of the annotations
    """
    if isinstance(cls, str):
        cls = [cls]
    elif not cls:
        return labels

    cls_stat = {c: 0 for c in cls}

    for label in labels:
        assert project_id in label['projects'], f"project_id {project_id} not in label"
        annotations = label['projects'][project_id]['labels'][0]['annotations']
        for obj in annotations['objects']:
            if obj['name'] in cls:
                cls_stat[obj['name']] += 1

    print(cls_stat)


def remove_cls_in_labels(labels: List[dict], cls: Union[List[str], str], project_id: str) -> List[dict]:
    """remove annotation of classes in LabelBox v2 labels
    :param labels: a list of LabelBox v2 labels
    :param cls: a list of classes to remove
    :param project_id: project id of the annotations
    :return: a new list of labels with classes removed
    """
    if isinstance(cls, str):
        cls = [cls]
    elif not cls:
        return labels
    return [_remove_cls_in_label(label, cls, project_id) for label in labels]


def _remove_cls_in_label(label: dict, cls: List[str], project_id: str) -> dict:
    """remove annotation of classes in a LabelBox v2 label
    :param label: a LabelBox v2 label
    :param cls: a list of classes to remove
    :param project_id: project id of the annotations
    :return: a new label with classes removed
    """
    assert project_id in label['projects'], f"project_id {project_id} not in label"

    label_copy = deepcopy(label)
    annotations = label_copy['projects'][project_id]['labels'][0]['annotations']
    annotations['objects'] = [obj for obj in annotations['objects'] if obj['name'] not in cls]
    return label_copy


def replace_cls_in_labels(labels: List[dict], cls_map: dict, project_id: str) -> List[dict]:
    """replace annotation of classes in LabelBox v2 labels
    :param labels: a list of LabelBox v2 labels
    :param cls_map: a map from old class to new class
    :param project_id: project id of the annotations
    :return: a new list of labels with classes replaced
    """
    return [_replace_cls_in_label(label, cls_map, project_id) for label in labels]


def _replace_cls_in_label(label: dict, cls_map: dict, project_id: str) -> dict:
    """replace annotation of classes in a LabelBox v2 label
    :param label: a LabelBox v2 label
    :param cls_map: a map from old class to new class
    :param project_id: project id of the annotations
    :return: a new label with classes replaced
    """
    assert project_id in label['projects'], f"project_id {project_id} not in label"

    label_copy = deepcopy(label)
    annotations = label_copy['projects'][project_id]['labels'][0]['annotations']
    for obj in annotations['objects']:
        if obj['name'] in cls_map:
            obj['name'] = cls_map[obj['name']]
    return label_copy


def remove_empty_labels(labels: List[dict], project_id: str) -> List[dict]:
    """remove empty labels in LabelBox v2 labels
    :param labels: a list of LabelBox v2 labels
    :param project_id: project id of the annotations
    :return: a new list of labels with empty labels removed
    """
    return [label for label in labels if not _is_label_empty(label, project_id)]


def _is_label_empty(label: dict, project_id: str) -> bool:
    """check if a LabelBox v2 label is empty
    :param label: a LabelBox v2 label
    :param project_id: project id of the annotations
    :return: True if the label is empty, False otherwise
    """
    assert project_id in label['projects'], f"project_id {project_id} not in label"

    annotations = label['projects'][project_id]['labels'][0]['annotations']
    return len(annotations['objects']) == 0


def split_lbv2_labels(
        labels: List[dict],
        labels_name: str,
        dest_dir: Union[str, Path],
        override: bool = False,
        random_seed: int = 42
):
    """ Split labels into train, val, test sets
    :param labels: list of labelbox labels
    :param labels_name: name of the labels
    :param dest_dir: destination directory
    :param override: override existing files
    :param random_seed: random seed
    :return: None"""

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=override)

    # try to split into train, val, test, 0.7, 0.15, 0.15
    train, test = train_test_split(labels, test_size=15 / 100, random_state=random_seed)
    train, val = train_test_split(train, test_size=15 / 85, random_state=random_seed)
    print(f"{labels_name}: Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    for split, split_labels in zip(['train', 'val', 'test'], [train, val, test]):
        save_lb_labels_json(dest_dir / f"{labels_name}_{split}.json", split_labels)


def labelboxv2_to_yolov8(
        split_path_map: Dict[str, str],
        dataset_name: str,
        classes: List[str],
        filename_key: str,
        override: bool = False,
):
    """ convert LabelBox v2 labels to YOLOv8 labels
    :param split_path_map: dict of split name to labelbox v2 labels path
    :param dataset_name: name of dataset
    :param classes: list of classes
    :param filename_key: key in data_rows to use as image/label file name
    :param override: override existing files
    """
    # Create a YOLOv8 compatible label file
    save_dir = "yolov8_labels"
    dest_dir = Path(save_dir) / dataset_name
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
    with open(Path(dest_dir) / f"{dataset_name}.yaml", "w") as f:
        yaml.dump(data_dicts, f)
