import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def relabel(img_path, new_class):
    """
    Shift image from one class into another class folder
    """
    path = Path(img_path)
    root_dir = path.parent.parent
    img_name = path.name
    new_path = root_dir / new_class / img_name
    shutil.move(path, new_path)
    
    
def get_labels_to_correct(corrected_labels_path, sensitivity):
    """
    Parameters
    ----------
    corrected_labels_path: str
        Path to corrected labels csv file
    sensitivity: int or float
        Sensitivity threshold to filter corrected labels. 
        E.g. if sensitivity=60, then only rows with % > 60 is returned.
    
    Returns
    -------
    df: DataFrame
        DataFrame containing rows of labels to be corrected
    """
    corrected_labels = pd.read_csv(corrected_labels_path, index_col=0)
    labels_to_correct_df = corrected_labels[
        corrected_labels['label']!=corrected_labels['corrected_label']
    ]
    labels_to_correct_df = labels_to_correct_df[labels_to_correct_df['%']>=sensitivity]
    return labels_to_correct_df
    

def do_autocorrect(labels_to_correct_df):
    for img_path, row in tqdm(labels_to_correct_df.iterrows()):
        corrected_label = row['corrected_label']
        relabel(img_path, corrected_label)