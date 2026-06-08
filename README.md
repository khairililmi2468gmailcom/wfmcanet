# WF-MCANet

### A Wavelet Frequency Module for Cross-Domain Generalisation in Cross-Axis Attention Medical Image Segmentation

> Bringing frequency-domain edge evidence into a cross-axis attention decoder. The module gives no benefit when in-domain accuracy is already saturated, but it measurably improves cross-domain (zero-shot) boundary accuracy.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c)
![Backbone](https://img.shields.io/badge/Backbone-EfficientNet--B3-2e7d32)
![Params](https://img.shields.io/badge/Parameters-~10.3M-555)
![Status](https://img.shields.io/badge/Status-Research%20%2F%20Manuscript%20under%20review-orange)

**Authors:** Khairil Ilmi, Juwita · Department of Informatics, Universitas Syiah Kuala, Banda Aceh, Indonesia

---

## Overview

Most deep-learning segmentation models reason only in the spatial domain and never look at the
frequency content of an image, even though high-frequency components encode the sharp transitions
that correspond to lesion edges. **WF-MCANet** inserts a **Wavelet Frequency Module (WFM)** into the
decoder of the MCANet cross-axis attention framework. The WFM decomposes decoder features with a
Haar wavelet transform, reweights the four sub-bands with Squeeze-and-Excitation channel attention,
and merges them back into the spatial stream through a **learnable gate**.

The model (EfficientNet-B3 backbone, ~10.3 M parameters) is trained on **ISIC 2018** (dermoscopy) and
tested zero-shot, without retraining, on **Kvasir-SEG** (colonoscopy), using a boundary-focused
protocol built around the 95th-percentile Hausdorff Distance (HD95).

**Main finding (stated honestly).** On the in-domain ISIC 2018 task, performance is already saturated
(~0.96 Dice for every configuration including the baseline), so the WFM provides **no in-domain gain**.
Its value appears under **cross-domain transfer**: the full model improves zero-shot Kvasir-SEG Dice
from 0.548 to 0.601 and reduces HD95 from 179.4 px to 143.5 px versus the spatial baseline.

---

## Highlights and Contributions

- **Wavelet Frequency Module (WFM)** placed *inside the decoder*. The contribution is not the Haar
  transform itself (which is classical) but a learnable module that uses Haar decomposition to
  strengthen edge evidence in an attention-based decoder.
- **Haar decomposition** into four sub-bands: LL (Low–Low), LH (Low–High), HL (High–Low), HH
  (High–High); the diagonal HH band captures edges that axis-aligned convolutions recover least well.
- **Squeeze-and-Excitation channel attention** over the sub-bands (after Hu et al.).
- **Learnable gated fusion** `F_new = F_spatial + σ(g) · Upsample(F_wav)` that protects the spatial
  stream from noisy high-frequency activations.
- **HD95-oriented evaluation** plus a **cross-domain zero-shot** protocol (ISIC 2018 → Kvasir-SEG).
- **Key result:** the frequency module improves *cross-domain generalisation*, not in-domain accuracy.

---

## Architecture

**Overall network.** Multi-scale features from the EfficientNet-B3 encoder are decoded by a cross-axis
attention block. The WFM supplies frequency-domain edge evidence, merged into the spatial stream by a
learnable gate before the segmentation head.

![Overall architecture of WF-MCANet](assets/architecture_overall.png)

**Wavelet Frequency Module (WFM).** The input feature is decomposed by a Haar transform into four
sub-bands, reweighted by Squeeze-and-Excitation channel attention, reduced by a 1×1 convolution,
upsampled, and fused with the spatial feature through the learnable gate.

![Detailed structure of the Wavelet Frequency Module](assets/wfm_module.png)

---

## Demo

A web-based demonstration application runs the trained checkpoint and shows the predicted mask,
overlay, lesion coverage, and inference time for an uploaded medical image.

![WF-MCANet demo application](assets/demo.png)

---

## Results (real, measured)

### In-domain ablation — ISIC 2018

| Configuration             | WFM | Edge | Dice ↑          | IoU ↑  | HD95 (px) ↓    | Param. (M) |
| ------------------------- | :-: | :--: | --------------- | ------ | -------------- | ---------- |
| A: Baseline (MCANet)      |  –  |  –   | 0.9603 ± 0.059  | 0.9283 | 11.28 ± 17.2   | 10.22      |
| B: Baseline + WFM         |  ✓  |  –   | 0.9600 ± 0.061  | 0.9279 | 11.90 ± 19.8   | 10.27      |
| C: Baseline + Edge loss   |  –  |  ✓   | 0.9617 ± 0.051  | 0.9301 | 11.42 ± 18.9   | 10.22      |
| D: Full (WFM + Edge)      |  ✓  |  ✓   | 0.9600 ± 0.059  | 0.9278 | 11.93 ± 19.2   | 10.27      |

> **In-domain, the four configurations are statistically indistinguishable** (gaps are far smaller
> than the standard deviation). The task is saturated and the WFM gives no in-domain benefit.
> Note: a Dice of ~0.96 is above typical ISIC 2018 results, so verify the held-out test split
> (no train/test overlap) before over-interpreting the absolute in-domain numbers.

![Ablation on ISIC 2018](assets/results_ablation.png)

### Cross-domain zero-shot — Kvasir-SEG (no retraining)

| Configuration             | Dice ↑    | IoU ↑     | HD95 (px) ↓ |
| ------------------------- | --------- | --------- | ----------- |
| A: Baseline (MCANet)      | 0.5477    | 0.4295    | 179.44      |
| B: Baseline + WFM         | 0.5573    | 0.4437    | 178.93      |
| C: Baseline + Edge loss   | 0.6049    | 0.4877    | 157.53      |
| **D: Full (WFM + Edge)**  | **0.6011**| **0.4856**| **143.47**  |

> **Cross-domain is where the module helps.** The full model achieves the best boundary accuracy
> (HD95 143.5 px vs 179.4 px for the baseline). Comparing C (edge only) with D (edge + WFM) isolates
> the WFM: similar Dice, but HD95 drops from 157.5 to 143.5 px — the module tightens the boundary in
> an unseen domain. Absolute Dice (~0.60) is modest, as expected across a large modality gap; this is
> a generalisation analysis, not clinical-grade polyp segmentation.

![Zero-shot on Kvasir-SEG](assets/results_zeroshot.png)

### Evaluation Protocol

- **Dice / IoU** — area-overlap metrics.
- **HD95** — 95th-percentile symmetric boundary distance; worst-case boundary error, most relevant
  for surgical-margin planning.

---

## Repository Structure

```text
.
├── train.py                  # training entry point
├── generate_figures.py       # measures real metrics + renders all journal figures
├── setup.sh                  # dataset download + environment setup
├── app_demo
│   ├── app.py                # demo server
│   ├── model.py              # WF-MCANet model definition
│   ├── index.html            # demo web UI
│   ├── run_demo.sh
│   ├── requirements.txt
│   └── checkpoints
│       └── D_WFMCANet_Full_best.pt
├── assets                    # figures used in this README
└── README.md
```

---

## Installation

```bash
git clone https://github.com/khairililmi2468gmailcom/wfmcanet.git
cd wfmcanet/app_demo
pip install -r requirements.txt
```

## Run the Demo

```bash
bash run_demo.sh
# then open http://localhost:5000
```

## Reproduce the Figures and Metrics

```bash
export KAGGLE_USERNAME=...   # use a freshly generated token
export KAGGLE_KEY=...
bash setup.sh
python generate_figures.py --checkpoints ./checkpoints --out ./journal_assets
```

---

## Datasets

- **ISIC 2018** — skin-lesion segmentation: https://challenge.isic-archive.com/data/
- **Kvasir-SEG** — colorectal polyp segmentation (zero-shot only): https://datasets.simula.no/kvasir-seg/

---

## Citation

```bibtex
@article{ilmi2026wfmcanet,
  title   = {A Wavelet Frequency Module for Cross-Domain Generalisation in Cross-Axis Attention Medical Image Segmentation},
  author  = {Ilmi, Khairil and Juwita},
  year    = {2026},
  note    = {Manuscript under review}
}
```

---

## Authors

**Khairil Ilmi** — Master of Artificial Intelligence, Department of Informatics, Universitas Syiah Kuala, Banda Aceh, Indonesia
**Juwita** — Lecturer, Department of Informatics, Universitas Syiah Kuala, Banda Aceh, Indonesia

**Corresponding author:** Khairil Ilmi · khairililmi2468@gmail.com

---

## License

Released for research and educational purposes.
