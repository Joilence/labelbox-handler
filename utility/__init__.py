import json
import os
from multiprocessing.pool import ThreadPool
from pathlib import Path

import labelbox as lb
import requests
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


def download_files(filemap: tuple[str, str]) -> str:
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
        labels: list[dict],
        filename_key: str,
        dest_dir: Path | str,
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


def load_lb_labels_json(filepath: str | Path):
    """Load labels from a LabelBox json file, with each line as a json object
    :param filepath: path to json file
    :return: list of labels
    """
    labels = []
    with open(filepath, "r") as f:
        labels.extend(json.loads(line) for line in f)
    return labels


def save_lb_labels_json(filepath: str | Path, labels: list[dict]):
    """ Save labels to a LabelBox json file, with each line as a json object
    :param filepath: path to json file
    """
    text = '\n'.join(json.dumps(label) for label in labels)
    with open(filepath, 'w') as file:
        file.write(text)
