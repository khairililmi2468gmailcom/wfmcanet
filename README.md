# WF-MCANet

WF-MCANet (Wavelet Fusion Multi-scale Cross-Axis Attention Network) is a deep learning framework for medical image segmentation that combines wavelet-based feature fusion and cross-axis attention mechanisms to improve lesion segmentation performance.

## Demo

![WF-MCANet Demo](demo.png)

Web-based interface for medical image segmentation using the final trained WF-MCANet model.

---

## Features

* MSCAN-T Backbone
* Wavelet Fusion Module
* Cross-Axis Multi-Context Attention
* Real-Time Segmentation Demo
* Overlay & Binary Mask Visualization
* Medical Image Analysis

---

## Repository Structure

```text
.
├── train.py
├── mscan_official.py
├── app_demo
│   ├── app.py
│   ├── model.py
│   ├── index.html
│   ├── run_demo.sh
│   ├── requirements.txt
│   └── checkpoints
│       └── D_WFMCANet_Full_best.pt
├── demo.png
└── README.md
```

---

## Included Model

The repository includes the final trained WF-MCANet checkpoint:

```text
D_WFMCANet_Full_best.pt
```

---

## Installation

```bash
git clone https://github.com/khairililmi2468gmailcom/wfmcanet.git
cd wfmcanet
```

Install dependencies:

```bash
cd app_demo
pip install -r requirements.txt
```

---

## Run Demo

```bash
cd app_demo
bash run_demo.sh
```

Open your browser:

```text
http://localhost:5000
```

---

## Usage

1. Launch the demo application.
2. Upload one or multiple medical images.
3. Click **Run Analysis**.
4. View:

   * Segmentation Mask
   * Overlay Visualization
   * Coverage Statistics
   * Inference Time

---

## Training

The final model checkpoint included in this repository was generated using:

```text
train.py
```

---

## Author

Khairil Ilmi

Master of Artificial Intelligence
Universitas Syiah Kuala
Banda Aceh, Indonesia

---

## License

This project is released for research and educational purposes.
