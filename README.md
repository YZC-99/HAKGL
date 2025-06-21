
<p align="center">
  <img src="https://img.shields.io/badge/status-under review-orange" alt="status">
  <img src="https://img.shields.io/badge/python-3.9+-blue" alt="python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="license">
</p>

<h1 align="center">🔬 HAKGL</h1>
<p align="center">
  <b>Hierarchy-Aware and Knowledge-Guided Learning for Multi-Label Classification of Retinal Diseases</b>
</p>

---

> 🔧 This is the official implementation of our paper **HAKGL**. The full code will be further refined and released upon our paper's acceptance and publication.

[comment]: <> (## 📘 Overview)

[comment]: <> (**HAKGL** introduces a novel dual-guided learning framework for fundus image analysis:)

[comment]: <> (- **HCCL**: Hierarchically Consistent Correlation Learning)

[comment]: <> (- **KGCL**: Knowledge-Guided Correlation Learning)

[comment]: <> (These components work jointly to enhance multi-label classification performance on retinal diseases using hierarchical prior and medical knowledge.)

[comment]: <> (---)

## 🧠 Key Components

### 📂 Hierarchical Transformer
> File: `./models/hierarchicaltransformer.py`

### 🔁 HCCL - Hierarchically Consistent Correlation Learning
> File: `./loss.py`

### 📚 KGCL - Knowledge-Guided Correlation Learning
> File: `./loss.py`

---

## 🗂️ Datasets

We train and evaluate on the following public datasets:

- 📑 **[ODIR Dataset]**  
  *N. Li, T. Li et al., “A benchmark of ocular disease intelligent recognition:
One shot for multi-disease detection,” in Benchmarking, Measuring,
and Optimizing: Third BenchCouncil International Symposium, Bench
2020, Virtual Event, November 15–16, 2020, Revised Selected Papers 3.
Springer, 2021, pp. 177–193.*

- 📑 **[RFMID Dataset]**  
  *S. Pachade, P. Porwal et al., “Retinal fundus multi-disease image dataset
(rfmid): A dataset for multi-disease detection research,” Data, vol. 6,
no. 2, p. 14, 2021.*

- 📑 **[Kaggle Diabetic Retinopathy]**  
  - *E. Dugas, Jared, Jorge et al., “Diabetic retinopathy detection,” 2015,
kaggle.* [Kaggle Competition Link](https://kaggle.com/competitions/diabetic-retinopathy-detection)
  
  - *L. Ju, X. Wang et al., “Improving medical images classification with
label noise using dual-uncertainty estimation,” IEEE Transactions on
Medical Imaging, vol. 41, no. 6, pp. 1533–1546, 2022.*
  
 

---

## 🌳 Hierarchical Tree Construction References

- **Ju et al., IEEE TMI 2022**: Improving medical images classification with
label noise using dual-uncertainty estimation 
- **Ju et al., IEEE TMI 2023**: Hierarchical knowledge guided learning for
real-world retinal disease recognition

---

## 🙏 Acknowledgements

We thank the authors of:
- [Query2Label](https://github.com/SlongLiu/query2labels)
- [HLEG](https://github.com/ShiQingHongYa/HLEG)

for their open-source contributions.

---

## 🚧 Ongoing Work

📢 Our paper is currently under peer review. We will further improve the code and contribute to the community in the future.
