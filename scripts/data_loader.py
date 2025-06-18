
import numpy as np
import os
import re

def extract_number(filename):
    return int(re.search(r'\d+', filename).group())

def load_data(path='cnn_input_data/'):
    class_files = sorted([f for f in os.listdir(path) if f.endswith(".npz")], key=extract_number)

    data, labels = [], []
    for i, class_file in enumerate(class_files):
        loaded = np.load(os.path.join(path, class_file))
        class_data = np.stack([loaded[k] for k in loaded.files])
        class_labels = np.ones(class_data.shape[0]) * i

        data.append(class_data)
        labels.append(class_labels)
        print("Loaded:", class_file)

    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    data[np.isnan(data)] = 0
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
    return data, labels, len(class_files), data.shape[2], class_files
