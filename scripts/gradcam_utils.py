import os
import re
from .utils import f1_m, precision_m, recall_m

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1

def prepare_gradcam_inputs(data, input_path='cnn_input_data/'):
    """
    Prepare GradCAM++ inputs: class file names and custom metric objects.

    Parameters
    ----------
    data : np.ndarray
        CNN input data (not used but included for consistency).
    input_path : str
        Path to directory containing .npz input files.

    Returns
    -------
    class_files : list of str
        Sorted list of .npz input files.
    custom_objects : dict
        Dictionary of metric functions for model loading.
    """
    class_files = sorted(
        [f for f in os.listdir(input_path) if f.endswith('.npz')],
        key=extract_number
    )

    custom_objects = {
        'f1_m': f1_m,
        'precision_m': precision_m,
        'recall_m': recall_m
    }

    return class_files, custom_objects
