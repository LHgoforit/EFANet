# EFANet

This repository provides all source code, configurations, and evaluation scripts necessary to reproduce the results of our paper.

---


## 📦 Environment Setup
### Option 1: Use Conda (recommended)
```bash
conda env create -f environment.yml
conda activate efanet
```
### Option 2: Use pip
```bash
pip install -r requirements.txt
```


## 📂 Dataset Preparation
EFANet supports three face datasets—CelebA, Helen and FFHQ, please manually download and organize datasets as follows:
 <!--EFANet supports three face datasets—[CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) [1], [Helen](https://exposing.ai/helen/) [2] and [FFHQ](https://github.com/NVlabs/ffhq-dataset) [3], please manually download and organize datasets as follows: --> 
```
EFANet/
└── data/
    ├── celeba/
    ├── helen/
    └── ffhq/
```


## 🚀 Training EFANet
EFANet supports training on multiple super-resolution scales (×4, ×8, ×16). For example, to train a ×4 model on the CelebA dataset:
```bash
python scripts/train.py --cfg configs/efanet_x4_celeba.yaml
```


## 🔍 Testing & Evaluation
Perform inference and evaluation on the Helen dataset for ×8 super-resolution:
```bash
python scripts/test.py --cfg configs/efanet_x8_helen.yaml
```
Generate quantitative comparisons:
```bash
python evaluation/compare_sota.py
```


## 📤 ONNX Export & Latency Test
Export EFANet to ONNX:
```bash
python scripts/export_onnx.py
```
Benchmark inference latency:
```bash
python scripts/benchmark_latency.py
```

 <!--
## References

[1] Liu, Ziwei, et al. *Deep learning face attributes in the wild*. Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2015.

[2] Le, Vuong, et al. *Interactive facial feature localization*. In: Computer Vision–ECCV 2012. 12th European Conference on Computer Vision, Florence, Italy, October 7–13, 2012. Proceedings, Part III. Springer Berlin Heidelberg, 2012.

[3] Karras, Tero, Samuli Laine, and Timo Aila. *A style-based generator architecture for generative adversarial networks*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.
 --> 
