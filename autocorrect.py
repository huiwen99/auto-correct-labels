"""
Uses the csv output from correct_labels.py and automatically relabels the data
"""
import yaml

import utils.data_utils as data_utils

# load configs
with open(r"configs/autocorrect_config.yaml") as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

results_path = cfg['results_path']
sensitivity = cfg['sensitivity']

df = data_utils.get_labels_to_correct(results_path, sensitivity)
data_utils.do_autocorrect(df)

print("Finished auto-correcting!")