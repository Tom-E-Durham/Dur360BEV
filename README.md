[![arXiv](https://img.shields.io/badge/arXiv-2503.00675-b31b1b.svg)](https://arxiv.org/abs/2503.00675)

# Dur360BEV: A Real-world 360-degree Single Camera Dataset and Benchmark for Bird-Eye View Mapping in Autonomous Driving

![Image Description](./ICRA_2025_Head_Image.png)

# Abstract  
We present **Dur360BEV**, a novel spherical camera autonomous driving dataset featuring a **128-channel 3D LiDAR** and an **RTK-refined GNSS/INS system**, along with a benchmark for **Bird’s Eye View (BEV) map generation** using a **single spherical camera**. This dataset addresses the challenges of BEV generation by **reducing hardware complexity**, replacing multiple perspective cameras with a **single 360-degree camera**.  

Our benchmark includes **SI2BEV (Spherical-Image-to-BEV)**, a novel module that projects spherical imagery into BEV space with a **refined sampling strategy**, as well as an **adaptive Focal Loss** formulation to handle **extreme class imbalance** in BEV segmentation. Extensive experiments demonstrate that **Dur360BEV** simplifies the sensor setup while achieving **competitive performance**.

# Dataset

## Sensor Setup  

Dur360BEV is equipped with a **spherical camera, 3D LiDAR, and a high-precision GNSS/INS system**, providing comprehensive environmental perception.

### **Camera**  
The dataset uses a **Ricoh Theta S** spherical dual-fisheye camera, featuring a **dual 1/2.3" 12M CMOS sensor**. It captures **RGB images** at a resolution of **1280 × 640** with a **15 Hz capture frequency**. The camera operates with **auto exposure**, and images are stored in **JPEG format** with **factory calibration** applied.

### **LiDAR**  
Dur360BEV includes an **Ouster OS1-128** LiDAR sensor with **128 vertical channels** and a **2048 horizontal resolution**. It captures data at **10 Hz** with a **360° horizontal field of view (HFOV)** and a **-21.2° to 21.2° vertical field of view (VFOV)**. The LiDAR achieves a detection range of **120m with >50% probability** and **100m with >90% probability**, with a **range resolution of 0.3 cm**.

### **GNSS/INS**  
For high-precision localization, the dataset incorporates an **OxTS RT3000v3** GNSS/INS system, operating at **100 Hz**. It provides **0.03° pitch/roll accuracy** and **0.15° slip angle accuracy**, achieving **centimeter-level positioning accuracy** with **RTK corrections received via NTRIP**.

## File Description

```
dataset/ 
├── image/  
│   ├── data/  
│   │   └── <frame_number.png>   [ ..... ]   
│   └── timestamp.txt  
├── gps/  
│   └── data.csv  
├── imu/  
│   └── data.csv  
├── ouster_points/  
│   ├── data/  
│   │   └── <frame_number.bin>   [ ..... ]   
│   └── timestamp.txt  
```

## Download Links

### Mini-version dataset

### Train and validation split used in our paper

### 3D model used in our LiDAR-Camera Setup

# Environment Setup  

To ensure reproducibility and ease of installation, we provide a structured virtual environment setup. This includes all required dependencies and the local `fisheye_tools` library.

```bash
git clone https://github.com/yourusername/icra2025-dur360bev.git
cd icra2025-dur360bev
```

## Virtual Envrionment setup
```bash
python3 -m venv dur360bev
source dur360bev/bin/activate
pip install -r requirements.txt
```



