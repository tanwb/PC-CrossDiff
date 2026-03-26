# PC_CrossDiff

## News

- **2026-03-26**: Official code release of **PC_CrossDiff**.
- **AAAI 2026**: Our paper **"PC-CrossDiff: Point-Cluster Dual-Level Cross-Modal Differential Attention for Unified 3D Referring and Segmentation"** is accepted by **Proceedings of the AAAI Conference on Artificial Intelligence**.

:tada::tada::tada:
This repository provides the PyTorch implementation of **PC_CrossDiff**, proposed in the paper ["PC-CrossDiff: Point-Cluster Dual-Level Cross-Modal Differential Attention for Unified 3D Referring and Segmentation"] (AAAI 2026).
If you are interested in our work and have any questions, please feel free to contact us at `wbtan@stu.xmu.edu.cn`. Discussions are welcome.
**Paper Framework**

![Paper Framework](./img/fig_overall_network.pdf)

## 0. Installation

+ **(1)** Install environment:
  ```bash
  conda create -n PC_CrossDiff python=3.8 -y
  conda activate PC_CrossDiff
  pip install torch==1.12.0+cu113 torchvision==0.13+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
  pip install numpy ipython psutil traitlets timm matplotlib termcolor ipdb scipy tensorboardX h5py wandb plyfile tabulate einops
  ```

+ **(2)** Set compatibility packages for `torch==1.12` (important)
  ```bash
  # fix: ImportError: cannot import name 'packaging' from 'pkg_resources'
  conda activate PC_CrossDiff
  python -m pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple --force-reinstall "setuptools==65.6.3" "packaging>=23,<25"
  ```

+ **(3)** Install spacy for text parsing
  ```bash
  pip install spacy
  pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.3.0/en_core_web_sm-3.3.0.tar.gz
  # or local file:
  # pip install ./en_core_web_sm-3.3.0.tar.gz
  ```

+ **(4)** Install `KNN_CUDA` and `pointnet2` (CUDA)
  ```bash
  cd ~/PC_CrossDiff

  # 4.1 KNN_CUDA
  cd KNN_CUDA
  chmod +x ninja
  export PATH="$(pwd):${PATH}"
  pip install -r requirements.txt
  python setup.py install --user
  cd ..

  # 4.2 pointnet2 ops
  cd pointnet2
  python setup.py install --user
  cd ..

  # 4.3 verify
  python - << 'PY'
import torch
assert torch.cuda.is_available(), "CUDA is required."
from knn_cuda import KNN
import pointnet2._ext
print("KNN_CUDA + pointnet2 installed successfully.")
PY
  ```

+ **(5)** Install segmentator from https://github.com/Karbo123/segmentator

## 0.1 Environment/runtime fixes 

- In `PC_CrossDiff`, it is recommended to pin `setuptools==65.6.3` and `packaging>=23,<25` to avoid the `pkg_resources.packaging` import error in `torch==1.12`.
- `KNN_CUDA` and `pointnet2` should both be installed and verified with CUDA enabled.

## 1. Quick visualization demo

We provide visualization through `wandb` for superpoints, keypoints, bad-case analysis, predicted masks, ground-truth masks, and boxes.
+ superpoints in `src/joint_det_dataset.py` line 71
```python
self.visualization_superpoint = False
```


## 2. Data preparation

The final required files are as follows:
```text
├── [DATA_ROOT]
│   ├── [1] train_v3scans.pkl # Packaged ScanNet training set
│   ├── [2] val_v3scans.pkl   # Packaged ScanNet validation set
│   ├── [3] ScanRefer/        # ScanRefer utterance data
│   │   ├── ScanRefer_filtered_train.json
│   │   ├── ScanRefer_filtered_val.json
│   │   └── ...
│   ├── [4] ReferIt3D/        # NR3D/SR3D utterance data
│   │   ├── nr3d.csv
│   │   ├── sr3d.csv
│   │   └── ...
│   ├── [5] group_free_pred_bboxes/  # detected boxes (optional)
│   ├── [6] gf_detector_l6o256.pth   # pointnet++ checkpoint (optional)
│   ├── [7] roberta-base/     # roberta pretrained language model
│   ├── [8] checkpoints/      # PC_CrossDiff pretrained models
```

+ **[1] [2] Prepare ScanNet Point Clouds Data**
  + **1)** Download ScanNet v2 data. Follow the [ScanNet instructions](https://github.com/ScanNet/ScanNet) to apply for dataset permission, and you will get the official download script `download-scannet.py`. Then use the following command to download the necessary files:
    ```bash
    python2 download-scannet.py -o [SCANNET_PATH] --type _vh_clean_2.ply
    python2 download-scannet.py -o [SCANNET_PATH] --type _vh_clean_2.labels.ply
    python2 download-scannet.py -o [SCANNET_PATH] --type .aggregation.json
    python2 download-scannet.py -o [SCANNET_PATH] --type _vh_clean_2.0.010000.segs.json
    python2 download-scannet.py -o [SCANNET_PATH] --type .txt
    ```
    where `[SCANNET_PATH]` is the output folder. The ScanNet dataset structure should look like below:
    ```text
    ├── [SCANNET_PATH]
    │   ├── scans
    │   │   ├── scene0000_00
    │   │   │   ├── scene0000_00.txt
    │   │   │   ├── scene0000_00.aggregation.json
    │   │   │   ├── scene0000_00_vh_clean_2.ply
    │   │   │   ├── scene0000_00_vh_clean_2.labels.ply
    │   │   │   ├── scene0000_00_vh_clean_2.0.010000.segs.json
    │   │   ├── scene.......
    ```
  + **2)** Package the above files into two `.pkl` files (`train_v3scans.pkl` and `val_v3scans.pkl`):
    ```bash
    python Pack_scan_files.py --scannet_data [SCANNET_PATH] --data_root [DATA_ROOT]
    ```
+ **[3] ScanRefer**: Download ScanRefer annotations following the instructions [HERE](https://github.com/daveredrum/ScanRefer). Unzip inside `[DATA_ROOT]`.
+ **[4] ReferIt3D**: Download ReferIt3D annotations following the instructions [HERE](https://github.com/referit3d/referit3d). Unzip inside `[DATA_ROOT]`.
+ **[5] group_free_pred_bboxes**: Download [object detector's outputs](https://drive.google.com/drive/folders/1vfOeTLKdW2AFoQPoivxT5sFloeZSXnEf). Unzip inside `[DATA_ROOT]`. (not used in single-stage method)
+ **[6] gf_detector_l6o256.pth**: Download PointNet++ [checkpoint](https://1drv.ms/u/s!AsnjK0KGPk10gYBXZWDnWle7SvCNBg?e=SNyUK8) into `[DATA_ROOT]`.
+ **[7] roberta-base**: Download the RoBERTa PyTorch model:
  ```bash
  cd [DATA_ROOT]
  git clone https://huggingface.co/roberta-base
  cd roberta-base
  rm -rf pytorch_model.bin
  wget https://huggingface.co/roberta-base/resolve/main/pytorch_model.bin
  ```
+ **[8] checkpoints**: Our pre-trained models (see Section 3).
+ **[9] ScanNetv2**: Prepare the preprocessed ScanNetv2 dataset by following the "Data Preparation" section from https://github.com/sunjiahao1999/SPFormer, obtaining the dataset file with the following structure:
```text
ScanNetv2
├── data
│   ├── scannetv2
│   │   ├── scans
│   │   ├── scans_test
│   │   ├── train
│   │   ├── val
│   │   ├── test
│   │   ├── val_gt
```
+ **[10] superpoints**: Prepare superpoints for each scene preprocessed from Step 9.
  ```bash
  cd [DATA_ROOT]
  python superpoint_maker.py  # modify data_root & split
  ```

## 3. Models

|         Dataset/Model         | RES mAP@0.25 | RES mAP@0.5 | RES mIoU | Model |
|:-----------------------------:|:------------:|:-----------:|:--------:|:---:|
|    ScanRefer/PC_CrossDiff     |    60.41     |    52.52    |  46.39   |  |
| ScanRefer/PC_CrossDiff (best) |    61.45     |    54.44    |  47.63   | [GoogleDrive](https://drive.google.com/file/d/1CPA1qzyiwdDPA-ma9T2ill6iijutSWqF/view?usp=sharing) |



|   Dataset/Model   | easy mAP@0.25 | hard mAP@0.25 | vd mAP@0.25 | vid mAP@0.25 | overall mAP@0.25 | Model |
|:-----------------:|:-------------:|:-------------:|:------------:|:-------------:|:-----------------:|:-----:|
| Nr3D/PC_CrossDiff |     62.22     |     57.52     |    60.55     |    59.60     |      59.91        | [GoogleDrive](https://drive.google.com/file/d/1QuYU6CpcnpTD8G1kVsU0au2nxwrzf5Mo/view?usp=sharing) |


## 4. Training

+ Please specify the paths of `--data_root`, `--log_dir`, `--pp_checkpoint` in the `train_*.sh` script first.
+ Logs are saved under `.../all_logs/PC_CrossDiff/<dataset>/<timestamp>/`, for example `.../all_logs/PC_CrossDiff/scanrefer/<timestamp>/`.
+ 
+ For **ScanRefer** training
  ```bash
  sh scripts/train_scanrefer_PC_CrossDiff_sp.sh
  ```
+ For **ScanRefer (single stage)** training
  ```bash
  sh scripts/train_scanrefer_PC_CrossDiff_sp_single.sh
  ```
+ For **SR3D** training
  ```bash
  sh scripts/train_sr3d_PC_CrossDiff_sp.sh
  ```
+ For **NR3D** training
  ```bash
  sh scripts/train_nr3d_PC_CrossDiff_sp.sh
  ```

## 5. Evaluation

+ Please specify the paths of `--data_root`, `--log_dir`, `--checkpoint_path` in the `test_*.sh` script first.
+ Test checkpoints should match the new log layout, for example `.../all_logs/PC_CrossDiff/scanrefer/<timestamp>/ckpt_best.pth`.
+ For **ScanRefer** evaluation
  ```bash
  sh scripts/test_scanrefer_PC_CrossDiff_sp.sh
  ```
+ For **ScanRefer (single stage)** evaluation
  ```bash
  sh scripts/test_scanrefer_PC_CrossDiff_sp_single.sh
  ```
+ For **SR3D** evaluation
  ```bash
  sh scripts/test_sr3d_PC_CrossDiff_sp.sh
  ```
+ For **NR3D** evaluation
  ```bash
  sh scripts/test_nr3d_PC_CrossDiff_sp.sh
  ```

## 6. Acknowledgements

This repository is built by reusing code from [EDA](https://github.com/yanmin-wu/EDA) and [3DRefTR](https://github.com/Leon1207/3DRefTR). We also thank [SPFormer](https://github.com/sunjiahao1999/SPFormer), [BUTD-DETR](https://github.com/nickgkan/butd_detr), [GroupFree](https://github.com/zeliu98/Group-Free-3D), [ScanRefer](https://github.com/daveredrum/ScanRefer), [SceneGraphParser](https://github.com/vacancy/SceneGraphParser), and [MCLN](https://github.com/qzp2018/MCLN).

## 7. Citation

If you find our work useful in your research, please consider citing:
```
@article{tanPCCrossDiffPointClusterDualLevel2026a,
  title = {{{PC-CrossDiff}}: {{Point-Cluster Dual-Level Cross-Modal Differential Attention}} for {{Unified 3D Referring}} and {{Segmentation}}},
  shorttitle = {{{PC-CrossDiff}}},
  author = {Tan, Wenbin and Lin, Jiawen and Wang, Fangyong and Xie, Yuan and Xie, Yong and Zhang, Yachao and Qu, Yanyun},
  year = 2026,
  month = mar,
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume = {40},
  number = {11},
  pages = {9332--9340},
  issn = {2374-3468},
  doi = {10.1609/aaai.v40i11.37892},
  urldate = {2026-03-21},
  abstract = {3D Visual Grounding (3DVG) aims to localize the referent of natural language referring expressions through two core tasks: Referring Expression Comprehension (3DREC) and Segmentation (3DRES). While existing methods achieve high accuracy in simple, single-object scenes, they suffer from severe performance degradation in complex, multi-object scenes that are common in real-world settings, hindering practical deployment. Existing methods face two key challenges in complex, multi-object scenes: inadequate parsing of implicit localization cues critical for disambiguating visually similar objects, and ineffective suppression of dynamic spatial interference from co-occurring objects, resulting in degraded grounding accuracy. To address these challenges, we propose PC-CrossDiff, a unified dual-task framework with a dual-level cross-modal differential attention architecture for 3DREC and 3DRES. Specifically, the framework introduces: (i) Point-Level Differential Attention (PLDA) modules that apply bidirectional differential attention between text and point clouds, adaptively extracting implicit localization cues via learnable weights to improve discriminative representation; (ii) Cluster-Level Differential Attention (CLDA) modules that establish a hierarchical attention mechanism to adaptively enhance localization-relevant spatial relationships while suppressing ambiguous or irrelevant spatial relations through a localization-aware differential attention block. To address the scale disparity and conflicting gradients in joint 3DREC--3DRES training, we propose L\_DGTL, a unified loss function that explicitly reduces multi-task crosstalk and enables effective parameter sharing across tasks. Our method achieves state-of-the-art performance on the ScanRefer, NR3D, and SR3D benchmarks. Notably, on the Implicit subsets of ScanRefer, it improves the Overall@0.50 score by +10.16\% for the 3DREC task, highlighting its strong ability to parse implicit spatial cues.},
  copyright = {Copyright (c) 2026 Association for the Advancement of Artificial Intelligence},
  langid = {english},
}
```
