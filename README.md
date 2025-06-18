# PriorCCI

**Prioritization of tumor-specific cell-cell interactions from single-cell RNA-seq data using deep learning.**  
This repository provides the full pipeline of the PriorCCI framework, including preprocessing, CNN-based classification, and GradCAM++-based interpretability.

---

## 🔧 Installation

We recommend using a Python 3.11 environment with TensorFlow 2.17.

### 1. Create a conda environment
```bash
conda create -n priorcci311 python=3.11 -y
conda activate priorcci311
```

### 2. Install required packages
```bash
pip install -r requirements_python311.txt
```

---

## 📁 Project structure

```
PriorCCI/
├── DB/                        # Input .h5ad and ligand-receptor CSVs
├── cnn_input_data/           # Preprocessed input files (.txt, .npz)
├── cnn_model/                # Trained CNN models (.h5)
├── gcam_res/                 # GradCAM++ results per class and model
├── CCI_res/                  # Aggregated GradCAM++ statistics
├── scripts/                  # Python functions and pipelines
│   ├── preprocess.py
│   ├── train.py
│   ├── gradcam.py
│   └── merge_gradcam.py
├── requirements_python311.txt
└── README.md
```

---

## 🧬 Usage

### Step 1: Preprocess input data

```python
from scripts.preprocess import input_data_preprocess
import scanpy as sc

adata = sc.read('DB/CCA_Lung_gs_notip.h5ad')
input_data_preprocess(adata, celltype_col='cell_type_major')
```

Generates `.txt` and `.npz` input files under `cnn_input_data/`.

---

### Step 2: Train CNN models

```python
from scripts.train import train_and_save_model, load_data

data, labels, num_classes, num_lrpair = load_data(path='cnn_input_data/')
train_and_save_model(data, labels, num_classes, num_lrpair, base_save_path='cnn_model/')
```

Trains and saves 10 CNN models with different splits.

---

### Step 3: Run GradCAM++ per model and class

```python
from scripts.gradcam import run_gradcam_analysis
from scripts.train import load_data

# Load input data and .npz file list
data, _, _, _ = load_data()
from os import listdir
class_files = sorted([f for f in listdir('cnn_input_data/') if f.endswith('.npz')])

run_gradcam_analysis(data=data,
                     gene_list_csv='DB/filtered_CCIdb.csv',
                     model_dir='cnn_model/',
                     save_dir='gcam_res/',
                     class_files=class_files)
```

Generates per-class importance scores for each trained model.

---

### Step 4: Merge GradCAM++ results

```python
from scripts.merge_gradcam import merge_gradcam_results

merge_gradcam_results(path='gcam_res/', save_path='CCI_res/')
```

Outputs `.csv` files summarizing mean, std, and frequency of top L-R pairs.

---

## 📊 Output example

- `gcam_res/gcamplus_result_Tip_Cells_v05.txt`: GradCAM++ weights per gene pair for Tip Cells in model v5
- `CCI_res/gcam_Tip_Cells_res.csv`: Aggregated results across models for Tip Cells

---

## 📦 Key dependencies

Installed via `requirements_python311.txt`. Key packages:

- `tensorflow==2.17.0`
- `keras==3.5.0`
- `tf-keras-vis==0.8.7`
- `scanpy`, `anndata`
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `seqeval`

---

## 📌 Citation

If you use this repository, please cite:

> **Kim H., et al.**  
> _제목_  
> (In submission to ---, 2025)

---

## 🧑‍💻 Contact

- **Maintainer**: Hanbyeol Kim (googija92@ncc.re.kr)  
- **Institution**: NCC Bioinformatics Research Division
