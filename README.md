[![Durham](https://img.shields.io/badge/UK-Durham-blueviolet)](https://durham-repository.worktribe.com/output/3704622)
[![arXiv](https://img.shields.io/badge/arXiv-2503.00675-b31b1b.svg)](https://arxiv.org/abs/2503.00675)

# Dur360BEV: A Real-world 360-degree Single Camera Dataset and Benchmark for Bird-Eye View Mapping in Autonomous Driving

![Image Description](./ICRA_2025_Head_Image.png)



https://github.com/user-attachments/assets/5e436f12-50df-485e-9938-7570144f2a29



## Abstract  
We present **Dur360BEV**, a novel spherical camera autonomous driving dataset featuring a **128-channel 3D LiDAR** and an **RTK-refined GNSS/INS system**, along with a benchmark for **Bird’s Eye View (BEV) map generation** using a **single spherical camera**. This dataset addresses the challenges of BEV generation by **reducing hardware complexity**, replacing multiple perspective cameras with a **single 360-degree camera**.  

Our benchmark includes **SI2BEV (Spherical-Image-to-BEV)**, a novel module that projects spherical imagery into BEV space with a **refined sampling strategy**, as well as an **adaptive Focal Loss** formulation to handle **extreme class imbalance** in BEV segmentation. Extensive experiments demonstrate that **Dur360BEV** simplifies the sensor setup while achieving **competitive performance**.

## News
- [2025/05/29] Dur360BEV-Extended has been released, providing new driving data and additional frames to complement the original Dur360BEV benchmark.
- [2025/01/27] Dur360BEV has been accepted by ICRA 2025.


## Sensor placement

Dur360BEV is equipped with a **spherical camera, 3D LiDAR, and a high-precision GNSS/INS system**, providing comprehensive environmental perception.

- **LiDAR**: [Ouster OS1-128 LiDAR sensor](https://ouster.com/products/os1-lidar-sensor/) with **128 channels** vertical resolution
- **Spherical Camera**: [Ricoh Theta S](https://www.ricoh-imaging.co.jp/english/products/theta_s/) featuring a **dual 1/2.3" 12M CMOS sensor**, **1280 × 640** resolution, and **15 Hz capture frequency**
- **GNSS/INS**: [OxTS RT3000v3](https://www.oxts.com/products/rt3000-v3/) global navigation satellite and inertial navigation system, supporting localization from GPS, GLONASS, BeiDou, Galileo, PPP and SBAS constellations

## Dataset File Description

```
dataset/ 
├── image
│   ├── data
│   └── timestamps.txt
├── labels
│   ├── data
│   ├── dataformat.txt
│   └── timestamps.txt
├── md5sums.txt
├── metadata
│   ├── dataset_indices.pkl
│   └── os1.json
├── ouster_points
│   ├── data
│   ├── dataformat.txt
│   └── timestamps.txt
└── oxts
    ├── data
    ├── dataformat.txt
    └── timestamps.txt
```

# Dataset Download
## Links
### Train and validation split used in our paper
- [Download from Hugging Face](https://huggingface.co/datasets/TomEeee/Dur360BEV)

### Dur360BEV_Extended
- [Download from Hugging Face](https://huggingface.co/datasets/TomEeee/Dur360BEV-Extended)

### 3D model used in our LiDAR-Camera Setup
To do...

## Download Guidance
### Downloading the full dataset from Hugging Face
#### Prerequisites
- Before starting, ensure you have the [**Hugging Face CLI (huggingface_hub)**](https://huggingface.co/docs/huggingface_hub/en/guides/cli) installed:

  ​	Install it via: ```pip install -U "huggingface_hub[cli]"```

#### Download the dataset using scripts
We provide a [pre-written Bash script](dur360bev_hf_download_script.sh) to download the dataset from Hugging Face. You need to manually modify the **User Configuration (Modify as Needed)** section at the beginning of [dur360bev_hf_download_script.sh](dur360bev_hf_download_script.sh) to match your desired paths and features.

- Dur360BEV dataset (used in our paper)
  
    ```bash dur360bev_hf_download_script.sh```

- Dur360BEV_Extended dataset only
  
    ```bash dur360bev_hf_download_script.sh extended```

- Dur360BEV_Complete dataset
  
    ```bash dur360bev_hf_download_script.sh complete```

    Then run [merge_datasets.py](merge_datasets.py) to combine **Dur360BEV_Extended** and **Dur360BEV** into a unified **Dur360BEV_Complete** dataset:

    ```python3 merge_datasets.py --folder_dir /path/to/Dur360BEV_datasets```

If you encounter any issues (e.g., network problems or unexpected interruptions), you can also modify this script to fit your needs. For example, if the extraction process is interrupted, you can manually comment out the dataset download section and resume from extraction.


> **Note**: The Hugging Face service may be restricted in certain countries or regions. In such cases, you may consider using a Hugging Face mirror (Hugging Face 镜像) as an alternative.

# Environment Setup  

To ensure reproducibility and ease of installation, we provide a structured virtual environment setup. This includes all required dependencies and the local `fisheye_tools` library.

### 🔐 Git LFS Notice

This repository uses Git LFS to manage large files (e.g. model checkpoints).

Please run the following before cloning or pulling:

```bash
git lfs install
```

```bash
git clone https://github.com/Tom-E-Durham/Dur360BEV.git
cd Dur360BEV
```

## Virtual Envrionment setup
```bash
./script/setup_env.sh
```

# Dataset Setup

After downloading the dataset, follow these steps to place it correctly:

1. **Move the dataset folder** into the project directory:


2. **Move the dataset split file** [dataset_indices.pkl](Dur360BEV_dataset/dataset_indices.pkl) into the dataset's `metadata` folder:

This ensures the evaluation scripts can correctly access both the dataset and the split definitions.

# Reproducing the Models
We provide checkpoints for the two models from our paper.
## Evaluate our models
To reproduce the results in our paper, run the following scripts:
```bash
./scripts/eval_coarse_fine.sh
./scripts/eval_dense.sh
```
These scripts will load pre-trained weights and perform evaluation on the corresponding test set of our dataset.
