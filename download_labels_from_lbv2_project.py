import os
import time
from pathlib import Path

import labelbox as lb
import yaml

from utility import save_lb_labels_json, split_lbv2_labels

# arguments
override = True

# read labelbox config
lb_cfg = yaml.load(Path("labelbox_config.yaml").read_text(), Loader=yaml.FullLoader)
PROJECT_ID = lb_cfg["PROJECT_ID"]
api_key = lb_cfg["LABELBOX_API_KEY"] or os.getenv("LABELBOX_API_KEY")

# set up labelbox labels paths
save_dir = "labelbox_labels"
client = lb.Client(api_key)
project = client.get_project(PROJECT_ID)
dest_dir = Path(save_dir) / project.name
dest_dir.mkdir(parents=True, exist_ok=override)

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
    save_lb_labels_json(dest_dir / "labels.json", labels)
    print(f"Saved labels to {dest_dir / 'labels.json'}")
    # split and save labels
    split_lbv2_labels(labels, project.name, dest_dir, override)
