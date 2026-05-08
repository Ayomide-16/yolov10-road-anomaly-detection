# YOLOv10 Road Anomaly Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4.21-1a56db)
![Model](https://img.shields.io/badge/Model-YOLOv10--M-purple)
![mAP@0.5](https://img.shields.io/badge/mAP%400.5-0.611-brightgreen)
![FPS](https://img.shields.io/badge/FPS-44.8%20(T4)-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Real-time detection of road surface anomalies using a fine-tuned **YOLOv10-M** model. Detects **potholes**, **speed bumps**, and **cracks** from road images at 44.8 FPS on a Tesla T4 GPU. Includes a learning rate sensitivity study across 5 configurations, a full analysis pipeline, and a zero-dependency desktop inference app built with Tkinter.

---

## Results

### Learning Rate Comparison

| LR | Best Epoch | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1e-05 | 76 | 0.5908 | 0.3900 | 0.6688 | 0.5433 |
| **0.0001** | **98** | **0.6107** | **0.4106** | **0.7378** | **0.5565** |
| 0.001 | 100 | 0.5466 | 0.3526 | 0.5992 | 0.5280 |
| 0.01 | 95 | 0.4611 | 0.2762 | 0.4941 | 0.4308 |
| 0.1 | 82 | 0.3558 | 0.1569 | 0.7277 | 0.3224 |

### Per-Class Performance (Best LR = 0.0001, test set)

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|:---:|:---:|:---:|:---:|:---:|
| Pothole | 0.877 | 0.570 | 0.743 | 0.443 |
| Speedbump | 0.917 | 0.893 | **0.947** | 0.733 |
| Crack | 0.383 | 0.174 | 0.129 | 0.046 |
| **Overall** | **0.726** | **0.546** | **0.606** | **0.408** |

### Inference Speed

| Hardware | Image Size | FPS | Latency |
|:---:|:---:|:---:|:---:|
| NVIDIA Tesla T4 | 640 x 640 | 44.8 +/- 0.9 | 22.3 ms |

---

## Model

**YOLOv10-M** fine-tuned from COCO pre-trained weights.

| Attribute | Value |
|---|---|
| Architecture | YOLOv10-M |
| Layers | 288 |
| Parameters | 16,576,768 |
| GFLOPs | 64.5 |
| Input Size | 640 x 640 |
| Framework | Ultralytics 8.4.21 |

YOLOv10 introduces a **dual-assignment training strategy** (one-to-many + one-to-one heads) enabling **NMS-free inference**, reducing end-to-end latency compared to prior YOLO versions.

---

## Dataset

A unified dataset assembled from three public sources:

| Source | Class | Link |
|---|---|---|
| Kaggle (rajdalsaniya) | Pothole | [Pothole Detection Dataset](https://www.kaggle.com/datasets/rajdalsaniya/pothole-detection-dataset) |
| Roboflow Universe | Speedbump | [Speed Unmarked Bump](https://universe.roboflow.com/pothole-detection-1nczj/speed-unmarked-bumb) |
| Google Drive | Crack | [Crack Dataset](https://drive.google.com/drive/folders/1a1ecTXoQpmzEXd4Uj4z9mAPJnmRGUbd) |

**Total: 6,203 images | Split: 70/15/15 (train/val/test) | Seed: 42**

| Split | Pothole | Speedbump | Crack | Images |
|:---:|:---:|:---:|:---:|:---:|
| Train | 3,010 | 1,765 | 2,051 | 4,342 |
| Val | 583 | 416 | 426 | 930 |
| Test | 653 | 354 | 457 | 931 |

### Preprocessing

The raw datasets required significant preprocessing before unification (`prepare_dataset.py`):

- **Crack dataset**: labels were supplied as polygon segmentation coordinates — converted to axis-aligned bounding boxes by computing the minimum enclosing rectangle over all polygon vertices
- **Speedbumps dataset**: contained 5 heterogeneous classes — filtered to 2 relevant classes (Speed Bump, Unmarked Bump) and remapped to unified class 1
- **All sources**: filename deduplication via source prefix, coordinate clamping to [0, 1], malformed annotation filtering, and a post-split validation pass

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/yolov10-road-anomaly-detection.git
cd yolov10-road-anomaly-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> Tkinter is built into Python — no extra UI package needed.

### 3. Add weights

Download the trained weights and place them in the `weights/` folder:

```
weights/
  lr_1e-05/best.pt
  lr_0.0001/best.pt      <- best performing
  lr_0.001/best.pt
  lr_0.01/best.pt
  lr_0.1/best.pt
```

---

## Inference

### Desktop App (Recommended)

```bash
python inference_app.py
```

A native desktop window opens immediately — no browser, no server, no internet required.

**Features:**
- Dropdown to switch between all 5 trained models
- Confidence and IoU threshold sliders
- Side-by-side input / output image panels
- Per-detection log table (class, confidence, position, size)
- Run summary (FPS, inference time, per-class counts)
- Save annotated result to disk

### Command Line

```bash
# Single image
python inference.py --weights weights/lr_0.0001/best.pt --image road.jpg

# Folder of images
python inference.py --weights weights/lr_0.0001/best.pt --image ./test_images/

# Custom thresholds
python inference.py --weights weights/lr_0.0001/best.pt --image road.jpg --conf 0.4 --iou 0.5

# All options
python inference.py --help
```

---

## Training

All runs: 100 epochs, batch 16, AdamW, Mosaic + MixUp + CutMix augmentation, seed 42, Tesla T4 GPU.

```bash
yolo train \
  model=yolov10m.pt \
  data=dataset.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  lr0=0.0001 \
  optimizer=AdamW \
  seed=42 \
  mosaic=1.0 \
  mixup=0.15 \
  copy_paste=0.10
```

Full training notebooks for all 5 learning rates are in `Training And Analysis Notebooks/`.

---

## Repository Structure

```
yolov10-road-anomaly-detection/
│
├── inference_app.py                  <- Tkinter desktop inference app
├── inference.py                      <- CLI inference script
├── prepare_dataset.py                <- Dataset preprocessing pipeline
├── requirements.txt
├── README.md
│
├── weights/
│   ├── lr_1e-05/best.pt
│   ├── lr_0.0001/best.pt
│   ├── lr_0.001/best.pt
│   ├── lr_0.01/best.pt
│   └── lr_0.1/best.pt
│
└── Training And Analysis Notebooks/
    ├── Group8_Train_LR_0_00001.ipynb
    ├── Group8_Train_LR_0_0001.ipynb
    ├── Group8_Train_LR_0_001.ipynb
    ├── Group8_Train_LR_0_01.ipynb
    ├── Group8_Train_LR_0_1.ipynb
    └── Group8_Analysis.ipynb
```

---

## Notes on Crack Detection

Crack detection achieves a lower mAP@0.5 (0.129) compared to the other classes, due to three compounding factors:

1. **Annotation format**: polygon labels converted to bounding boxes produce looser boxes for elongated crack structures
2. **Class imbalance**: fewer annotations relative to visual complexity
3. **Visual subtlety**: fine linear surface textures are easily confused with road background at 640 x 640 resolution

The confusion matrix shows 78% of crack instances misclassified as background. Higher-resolution inference, class-weighted loss, or a dedicated crack segmentation model would likely improve this substantially.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| No weights found | Confirm `.pt` files are in `weights/<lr_folder>/best.pt` |
| `CUDA out of memory` | Use `--imgsz 320` or select 320 in the app |
| Slow on CPU | Normal — install CUDA PyTorch for GPU speed (see `requirements.txt`) |

---

## License

MIT — free to use, modify, and distribute with attribution.

---

*Built with [Ultralytics YOLOv10](https://github.com/ultralytics/ultralytics)*