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

All steps are provided in the `PriorCCI_pipeline.ipynb` notebook, including:

1. Preprocessing input data (`input_data_preprocess`)
2. Generating `.npz` input files
3. CNN training and evaluation (10 repetitions)
4. Model visualization using confusion matrix and ROC curves
5. GradCAM++ analysis across models and classes
6. Merging GradCAM++ results into final ranked ligand-receptor pairs

To run the full pipeline:

```bash
jupyter notebook PriorCCI_pipeline.ipynb
```
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

Kim H., Choi E., Shim Y., and Kwon J.
PriorCCI: Interpretable Deep Learning Framework for Identifying Key Ligand–Receptor Interactions Between Specific Cell Types from Single-Cell Transcriptomes.
*International Journal of Molecular Sciences*, 26(15), 7110, 2025. https://doi.org/10.3390/ijms26157110

---

## Contact

- **Maintainer**: Hanbyeol Kim (googija92@ncc.re.kr) and Joonha Kwon (joon2k@ncc.re.kr)
- **Institution**: Bioinformatics Branch, Division of Cancer Data Science, Research Institute, National Cancer Center
