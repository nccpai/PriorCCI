# PriorCCI

**Prioritization of the ligand-receptor pairs for the specific cell-cell interaction from single-cell RNA-seq data using deep learning.**  
This repository provides the full pipeline of the PriorCCI framework, including preprocessing, CNN-based classification, and GradCAM++-based interpretability.

---

## Installation

We recommend using a Python 3.11 environment with TensorFlow 2.17.

### 1. Create a conda environment
```bash
conda create -n priorcci python=3.11 -y
conda activate priorcci
```

### 2. Install required packages
```bash
pip install -r requirements_python.txt
```

### 3. Toy data download
```python
import gdown

url = "https://drive.google.com/file/d/1p80kgvtsOD4YAcmSefo_xGwQQuRS9vUr/view?usp=drive_link"
gdown.download(url, output='DB/CCA_Lung_toy.h5ad', quiet=False, fuzzy=True)
```
---

## Usage

### 1: Preprocess input data

```python
# Step 1: Preprocess input data for CNN
adata = sc.read('DB/CCA_Lung_toy.h5ad')
input_data_preprocess(
    adata,
    celltype_col='cell_type_major',
    output_folder='cnn_input_data',
    n_repeat=1000
)

```

Generates `.txt` and `.npz` input files under `cnn_input_data/`.

---

### 2: Train CNN models

```python
# Step 2: Load .npz samples and prepare training labels
data, labels, num_classes, num_lrpair = load_data(path='cnn_input_data/')

# Step 3: Train and save CNN models
train_and_save_model(data, labels, num_classes, num_lrpair)

# Step 4: Evaluate all trained models
evaluate_saved_models(data, labels, num_classes)

# Step 5: Visualize final model (v10)
model_path = 'cnn_model/data_cnn-model_v010.h5'
model = load_model(model_path, custom_objects={
    'f1_m': f1_m,
    'precision_m': precision_m,
    'recall_m': recall_m
})

_, test_data, _, test_labels = train_test_split(data, labels, test_size=0.2, random_state=51)
visualize_final_model_results(model, test_data, test_labels)
```

Trains and saves 10 CNN models with different splits.

---

### 3: Run GradCAM++ per model and class

```python
# Step 6: Run GradCAM++ analysis
import os
import re

def extract_number(filename):
    return int(re.search(r'\d+', filename).group())

class_files = sorted([f for f in os.listdir('cnn_input_data/') if f.endswith('.npz')], key=extract_number)

custom_objs = {'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m}

run_gradcam_analysis(
    data=data,
    gene_list_csv='DB/filtered_CCIdb.csv',
    model_dir='cnn_model/',
    save_dir='gcam_res/',
    class_files=class_files,
    custom_objects=custom_objs
)
```

Generates per-class importance scores for each trained model.

---

### 4: Merge GradCAM++ results

```python
# Step 7: Merge GradCAM++ results
merge_gradcam_results(
    path='gcam_res/',
    save_path='CCI_res/'
)
```

Outputs `.csv` files summarizing mean, std, and frequency of top L-R pairs.

---

## Key dependencies

Installed via `requirements_python.txt`. Key packages:

- `tensorflow==2.17.0`
- `keras==3.5.0`
- `tf-keras-vis==0.8.7`
- `scanpy`, `anndata`
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `seqeval`

---

## Citation

If you use this repository, please cite:

> **Kim H., et al.**  
> not yet published  
> (In submission to ---, 2025)

---

## Contact

- **Maintainer**: Hanbyeol Kim (googija92@ncc.re.kr) and Joonha Kwon (joon2k@ncc.re.kr)
- **Institution**: Bioinformatics Branch, Division of Cancer Data Science, Research Institute, National Cancer Center
