# WF-MCANet

A Lightweight Wavelet Frequency Network with Cross-Axis Attention for Medical Image Segmentation

Khairil Ilmi, Juwita
Department of Informatics, Universitas Syiah Kuala

---

## Demo

![WF-MCANet Demo](demo.png)

Web-based medical image segmentation system powered by the final WF-MCANet model.

---

## Abstract

WF-MCANet is a lightweight medical image segmentation architecture that integrates a Wavelet Frequency Module (WFM) into the MCANet framework. The proposed model introduces frequency-domain information through Haar wavelet decomposition and combines it with cross-axis attention mechanisms to improve lesion boundary delineation.

The model was evaluated on the ISIC 2018 skin lesion dataset and further tested in a zero-shot setting on the Kvasir-SEG polyp dataset without retraining. Experimental results demonstrate improved Dice score and boundary accuracy (HD95) while requiring only a small increase in model parameters.

---

## Key Contributions

* Wavelet Frequency Module (WFM)
* Haar Wavelet Decomposition (LL, LH, HL, HH)
* Learnable Gated Fusion Mechanism
* Cross-Axis Multi-Context Attention Decoder
* HD95-Oriented Evaluation Protocol
* Cross-Domain Zero-Shot Generalization Analysis

---

## Main Results (ISIC 2018)

| Model                      | Dice ↑     | HD95 ↓    | Parameters |
| -------------------------- | ---------- | --------- | ---------- |
| MCANet Baseline            | 0.9100     | 23.79     | 4.04 M     |
| WF-MCANet (Ours)           | **0.9123** | **23.52** | 4.24 M     |
| Edge Loss Only             | 0.9088     | 24.48     | 4.04 M     |
| Full WF-MCANet (EffNet-B3) | 0.9097     | 23.69     | 13.10 M    |

WF-MCANet improves lesion-boundary accuracy while adding only **0.20 M parameters** compared to the baseline.

---

## Zero-Shot Evaluation (Kvasir-SEG)

The model was trained only on ISIC 2018 and directly evaluated on Kvasir-SEG without retraining.

| Model                      | Dice ↑    |
| -------------------------- | --------- |
| MCANet Baseline            | 0.287     |
| WF-MCANet (Ours)           | **0.296** |
| Edge Loss Only             | 0.281     |
| Full WF-MCANet (EffNet-B3) | 0.289     |

These results indicate improved cross-domain generalization through frequency-aware feature representations.

---

## Model Architecture

WF-MCANet consists of:

* MobileNetV2 Encoder
* MSCAN-T Feature Extraction
* Wavelet Frequency Module (WFM)
* Channel Attention
* Learnable Gated Fusion
* Cross-Axis MCA Decoder
* Segmentation Head

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

The repository includes the final trained checkpoint:

```text
D_WFMCANet_Full_best.pt
```

Generated using:

```text
train.py
```

---

## Installation

```bash
git clone https://github.com/khairililmi2468gmailcom/wfmcanet.git
cd wfmcanet/app_demo

pip install -r requirements.txt
```

---

## Run Demo

```bash
bash run_demo.sh
```

Open:

```text
http://localhost:5000
```

---

## Usage

1. Launch the application.
2. Upload a medical image.
3. Click **Run Analysis**.
4. View:

   * Segmentation Mask
   * Overlay Visualization
   * Coverage Statistics
   * Inference Time

---

## Citation

If you use this work in your research, please cite:

```bibtex
@article{ilmi2026wfmcanet,
  title={A Lightweight Wavelet Frequency Network with Cross-Axis Attention for Medical Image Segmentation},
  author={Ilmi, Khairil and Juwita},
  year={2026}
}
```

---

## Author

Khairil Ilmi
Juwita
Master of Artificial Intelligence
Universitas Syiah Kuala
Banda Aceh, Indonesia

---

## License

This project is released for research and educational purposes.

