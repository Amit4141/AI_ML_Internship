# Face Mask Detection System

Real-time face mask detection using CNN and OpenCV.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. Train the model:
```bash
python train_model.py
```

2. Run real-time detection:
```bash
python detect_mask.py
```

3. Run Flask web app (optional):
```bash
python app.py
```

## Dataset

Download from Kaggle: [Face Mask Detection Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)

Place in `dataset/` folder with structure:
- dataset/with_mask/
- dataset/without_mask/
