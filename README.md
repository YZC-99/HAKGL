<p align="center">
  <img src="https://img.shields.io/badge/Accepted-IEEE%20TMI-blue" alt="status">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="license">
</p>

<h1 align="center">HAKGL</h1>

<p align="center">
  <b>Hierarchy-Aware and Knowledge-Guided Learning for Multi-Label Classification of Retinal Diseases from Fundus Images</b>
</p>

<p align="center">
  Official implementation of our IEEE Transactions on Medical Imaging paper.
</p>

---

## ✨ Overview

**HAKGL** is a hierarchy-aware and knowledge-guided learning framework for **multi-label retinal disease classification** from fundus images.  
It explicitly incorporates both **hierarchical disease structure** and **medical prior knowledge** to improve label correlation modeling and enhance recognition performance in real-world retinal disease analysis.

The framework mainly consists of:

- **Hierarchical Transformer** for structured representation learning
- **HCCL**: *Hierarchically Consistent Correlation Learning*
- **KGCL**: *Knowledge-Guided Correlation Learning*

> This repository is the official implementation of our paper.  
---

## 🖼️ Framework

<p align="center">
  <img src="framework.png" alt="HAKGL Framework" width="95%">
</p>

<p align="center">
  <em>Overall framework of HAKGL. Please place your main architecture figure at <code>./framework.png</code>.</em>
</p>

---

---

## 🧠 Method Components

### 1. Hierarchical Transformer
Structured visual representation learning with explicit hierarchical awareness.

- **File**: `./models/hierarchicaltransformer.py`

### 2. HCCL — Hierarchically Consistent Correlation Learning
Encourages prediction consistency with hierarchical disease relations.

- **File**: `./loss.py`

### 3. KGCL — Knowledge-Guided Correlation Learning
Introduces external medical knowledge to regularize disease correlation learning.

- **File**: `./loss.py`

---

## 📂 Supported Datasets

We train and evaluate HAKGL on the following public retinal fundus datasets:

### **ODIR**
> Li, N., Li, T., et al.  
> *A benchmark of ocular disease intelligent recognition: One shot for multi-disease detection.*  
> BenchCouncil International Symposium, 2021.

### **RFMiD**
> Pachade, S., Porwal, P., et al.  
> *Retinal fundus multi-disease image dataset (RFMiD): A dataset for multi-disease detection research.*  
> *Data*, 2021.

### **Kaggle Diabetic Retinopathy**
> Dugas, E., Jared, J., Jorge, et al.  
> *Diabetic Retinopathy Detection.* Kaggle, 2015.  
> [Competition Link](https://kaggle.com/competitions/diabetic-retinopathy-detection)

Related processing protocol:
> Ju, L., Wang, X., et al.  
> *Improving medical images classification with label noise using dual-uncertainty estimation.*  
> *IEEE Transactions on Medical Imaging*, 2022.

---

## 🌳 Hierarchical Prior Construction

The hierarchical tree design in this work is inspired by prior retinal disease recognition studies, including:

- **Ju et al., IEEE TMI 2022**  
  *Improving medical images classification with label noise using dual-uncertainty estimation*

- **Ju et al., IEEE TMI 2023**  
  *Hierarchical knowledge guided learning for real-world retinal disease recognition*

---


## 📌 Citation

If you find this work useful in your research, please cite:

```bibtex
@article{yang2026hierarchy,
  title={Hierarchy-Aware and Knowledge-Guided Learning for Multi-Label Classification of Retinal Diseases from Fundus Images},
  author={Yang, Zhaocan and Li, Yan and Liu, Yang},
  journal={IEEE Transactions on Medical Imaging},
  year={2026},
  publisher={IEEE}
}
```

---

## 🙏 Acknowledgements

We sincerely thank the authors of the following open-source projects:

- [Query2Label](https://github.com/SlongLiu/query2labels)
- [HLEG](https://github.com/ShiQingHongYa/HLEG)

for their valuable contributions to the community.


