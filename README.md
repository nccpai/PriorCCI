# PriorCCI

**Prioritization of tumor-specific cell-cell interactions from single-cell RNA-seq data using deep learning.**  
This repository provides the full pipeline of the PriorCCI framework, from preprocessing to CNN-based classification and GradCAM++-based interpretability.

---

## 🔧 Installation

We recommend using a Python 3.11 environment with TensorFlow 2.17.

### 1. Create a conda environment
```bash
conda create -n priorcci python=3.11 -y
conda activate priorcci

pip install -r requirements_python.txt
