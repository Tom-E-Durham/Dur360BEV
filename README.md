[![arXiv](https://img.shields.io/badge/arXiv-2503.00675-b31b1b.svg)](https://arxiv.org/abs/2503.00675)

# Dur360BEV: A Real-world 360-degree Single Camera Dataset and Benchmark for Bird-Eye View Mapping in Autonomous Driving

![Image Description](./ICRA_2025_Head_Image.png)

## Abstract  
We present **Dur360BEV**, a novel spherical camera autonomous driving dataset featuring a **128-channel 3D LiDAR** and an **RTK-refined GNSS/INS system**, along with a benchmark for **Bird’s Eye View (BEV) map generation** using a **single spherical camera**. This dataset addresses the challenges of BEV generation by **reducing hardware complexity**, replacing multiple perspective cameras with a **single 360-degree camera**.  

Our benchmark includes **SI2BEV (Spherical-Image-to-BEV)**, a novel module that projects spherical imagery into BEV space with a **refined sampling strategy**, as well as an **adaptive Focal Loss** formulation to handle **extreme class imbalance** in BEV segmentation. Extensive experiments demonstrate that **Dur360BEV** simplifies the sensor setup while achieving **competitive performance**.

## News
- [2025/01/27] Dur360BEV has been accepted by ICRA 2025.


## Sensor placement

Dur360BEV is equipped with a **spherical camera, 3D LiDAR, and a high-precision GNSS/INS system**, providing comprehensive environmental perception.

- **LiDAR**: [Ouster OS1-128 LiDAR sensor](https://ouster.com/products/os1-lidar-sensor/) with 128 channels vertical resolution
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

## Download Links

### Train and validation split used in our paper
[Download from OneDrive](https://durhamuniversity-my.sharepoint.com/:f:/g/personal/hhgb23_durham_ac_uk/Eucqrf1f7GlJuLYVxMqDNSUBDFkJv14P_i-4_S8fQFfeuQ?e=RRemIL)

### 3D model used in our LiDAR-Camera Setup
To do...

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



