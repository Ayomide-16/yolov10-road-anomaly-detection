# Group 8 - MCE 415: YOLOv10 Road Anomaly Detection
## Submission Inference Package

This folder is ready for evaluation and contains a one-click launcher for Windows.

## Package Contents

```
submit/
├── Run_Inference_App.bat
├── inference_app.py
├── inference.py
├── requirements.txt
├── README.md
└── weights/
    ├── lr_1e-05/best.pt
    ├── lr_0.0001/best.pt
    ├── lr_0.001/best.pt
    ├── lr_0.01/best.pt
    └── lr_0.1/best.pt
```

## Recommended Run Method (Windows)

1. Ensure Python 3.10+ is installed.
2. Double-click `Run_Inference_App.bat`.

The launcher will automatically:
- Create a local `.venv` (first run only)
- Install dependencies if they are missing
- Start the interactive app

## Manual Run (Alternative)

If you prefer terminal commands:

```bash
pip install -r requirements.txt
python inference_app.py
```

## How To Use The App

1. Click Browse Image and choose a road image.
2. Select the model/learning-rate option from the dropdown.
3. Adjust confidence and IoU sliders if needed.
4. Click Run Detection.
5. Review detections and click Save Result to export the annotated image.

## Command-Line Usage

```bash
# Single image
python inference.py --weights weights/lr_0.001/best.pt --image road.jpg

# Folder of images
python inference.py --weights weights/lr_0.001/best.pt --image ./test_images/

# Custom thresholds
python inference.py --weights weights/lr_0.001/best.pt --image road.jpg --conf 0.4 --iou 0.5
```

## Detected Classes

- Class 0: Pothole
- Class 1: Speedbump
- Class 2: Crack

## Troubleshooting

- `Python not found`: install Python 3.10+ and reopen the folder.
- `pip not found`: run `python -m pip install -r requirements.txt`.
- `No .pt weight files found`: ensure the `weights/` folder contains `best.pt` files.
- Slow inference on CPU: expected behavior; GPU will be faster.

## Submission Status

- One-click launcher included
- Inference app included
- CLI inference script included
- Dependencies listed
- All 5 trained weight files included

Group 8 - MCE 415 - Federal University of Technology Minna
