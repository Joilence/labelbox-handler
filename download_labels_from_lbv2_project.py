import os
import time
from pathlib import Path

import labelbox as lb
import yaml

from utility import save_lb_labels_json

# arguments
override = True
dataset_name = 'dtc2023'

# read labelbox config
lb_cfg = yaml.load(Path("labelbox_config.yaml").read_text(), Loader=yaml.FullLoader)
PROJECT_ID = lb_cfg["PROJECT_ID"]
api_key = lb_cfg["LABELBOX_API_KEY"] or os.getenv("LABELBOX_API_KEY")

# set up labelbox labels paths
save_dir = "labelbox_labels"
client = lb.Client(api_key)
project = client.get_project(PROJECT_ID)
dest_dir = Path(save_dir) / dataset_name or project.name
dest_dir.mkdir(parents=True, exist_ok=override)
json_file_path = dest_dir / f"{dest_dir.stem}.json"

# export labelbox labels
project_export_task = project.export_v2()
print(f"Waiting for project \"{project.name}\" export to complete... (could take a several minutes)")
start_time = time.time()
project_export_task.wait_till_done()

# save labelbox labels
if project_export_task.errors:
    print(project_export_task.errors)
else:
    print(f"Project export completed in {time.time() - start_time:.2f} seconds")
    labels = project_export_task.result
    save_lb_labels_json(json_file_path, labels)
    print(f"Saved labels to {json_file_path}")
