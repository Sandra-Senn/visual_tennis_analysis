# CDA2-3DA TENNIS CHALLENGE

A comprehensive data science and machine learning challenge focused on tennis match analysis, featuring YOLO-based object detection and interactive dashboarding.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![YOLO](https://img.shields.io/badge/YOLO-v8%20%7C%20v11-green.svg)
![Streamlit](https://img.shields.io/badge/streamlit-dashboard-red.svg)

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Folder Structure](#folder-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Sources](#data-sources)
- [Usage Guide](#usage-guide)
- [Model Training](#model-training)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview

This repository contains a complete pipeline for tennis match analysis using computer vision and machine learning. The project processes tennis match footage to detect and track players, rackets, balls, and court boundaries, then presents insights through an interactive dashboard.

### Use Cases
- **Performance Analysis**: Track player movements and shot patterns
- **Coaching Tools**: Analyze technique and strategy
- **Broadcasting**: Generate real-time match statistics
- **Research**: Study tennis biomechanics and tactics

## Key Features

- **Computer Vision**: YOLO-based detection for rackets, balls, and court lines
- **Data Processing**: Automated calibration and data cleaning pipelines
- **Machine Learning**: Custom-trained models for tennis-specific object detection
- **Interactive Dashboard**: Streamlit-based visualization of match statistics
- **Analysis Notebooks**: Jupyter notebooks for exploratory data analysis
- **UI Design**: Professional dashboard mockups and prototypes

## Folder Structure

```
CDA2-3DA_TENNIS-CHALLENGE/
│
├── dashboard/                      # Interactive visualization app
│   ├── Dashboard.py               # Main Streamlit dashboard
│   └── Match_Daten.csv           # Dashboard data source
│
├── data/
│   └── daw/                       # Data processing & cleaning
│       ├── Datenausarbeitung.ipynb    # Data prep notebook
│       ├── Kalibrierung.csv           # Calibration parameters
│       ├── Match_Daten_ungereinigt.csv # Raw match data
│       └── Match_Daten.csv            # Cleaned match data
│
├── runs/
│   └── detect/                    # YOLO detection outputs
│       ├── track/                 # Tracking experiments
│       ├── track2/
│       ├── track8/
│       ├── train16/               # Training runs
│       ├── train19/
│       └── train21/
│
├── try_ml/                        # ML experimentation workspace
│   ├── _input/                    # Input data for training
│   ├── models/                    # Trained model checkpoints
│   ├── training_ball/             # Ball detection training data
│   ├── training_court/            # Court detection training data
│   ├── training_racket/           # Racket detection training data
│   ├── ml_test_tennis.ipynb      # ML testing notebook
│   ├── tennis_match_analysis.csv  # Analysis results
│   └── yolo11n-pose.pt           # Pose estimation model
│
├── ui_design/                     # Dashboard design mockups
│   └── Erster_Entwurf_Dashboard.pdf
│
├── yolo/                          # YOLO implementation files
│
├── .gitignore
├── README.md
├── requirements.txt               # Python dependencies
└── yolov8x.pt                    # Pre-trained YOLO model
```

## Requirements

### System Requirements
- **Python**: 3.8 or higher (recommended: 3.10)
- **GPU**: CUDA-compatible GPU recommended for training (optional)
- **RAM**: Minimum 8GB, 16GB+ recommended
- **Storage**: Approximately 5GB for models and data

### Core Dependencies
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `matplotlib` - Visualization
- `opencv-python` - Computer vision operations
- `torch` - Deep learning framework
- `ultralytics` - YOLO implementation
- `streamlit` - Dashboard framework
- `jupyter` - Interactive notebooks
- `seaborn` - Statistical visualization
- `pillow` - Image processing

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/CDA2-3DA_TENNIS-CHALLENGE.git
cd CDA2-3DA_TENNIS-CHALLENGE
```

### 2. Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import torch; import ultralytics; print('Installation successful!')"
```

## Quick Start

### Running the Dashboard
```bash
streamlit run dashboard/Dashboard.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`

### Running Jupyter Notebooks
```bash
jupyter notebook
```
Navigate to `data/daw/Datenausarbeitung.ipynb` or `try_ml/ml_test_tennis.ipynb`

### Quick Object Detection Test
```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8x.pt')

# Run inference
results = model.predict(source='path/to/tennis/video.mp4', save=True)
```

## Data Sources

### Input Data
- **Kalibrierung.csv**: Camera calibration parameters for court perspective transformation
- **Match_Daten_ungereinigt.csv**: Raw match data with potential inconsistencies
- **Match_Daten.csv**: Cleaned and validated match data ready for analysis

### Model Weights
- **yolov8x.pt**: Pre-trained YOLOv8 extra-large model
- **yolo11n-pose.pt**: YOLO11 nano model for pose estimation
- **Custom Models**: Located in `try_ml/models/` directory

### Output Data
- Detection results stored in `runs/detect/`
- Training metrics and logs in respective `train*/` folders
- Tracking results in `track*/` folders

## Usage Guide

### 1. Data Preparation
Open and run the data preparation notebook:
```bash
jupyter notebook data/daw/Datenausarbeitung.ipynb
```
This notebook handles data cleaning, calibration, and preprocessing.

### 2. Model Training
For training custom YOLO models on tennis-specific data:
```python
from ultralytics import YOLO

# Initialize model
model = YOLO('yolo11n.pt')

# Train on custom dataset
model.train(
    data='training_ball/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='tennis_ball_detector'
)
```

### 3. Object Detection
Run detection on tennis match videos:
```python
model = YOLO('path/to/trained/model.pt')
results = model.predict(
    source='match_video.mp4',
    conf=0.5,
    save=True,
    project='runs/detect',
    name='match_analysis'
)
```

### 4. Tracking Objects
For continuous tracking across frames:
```python
results = model.track(
    source='match_video.mp4',
    tracker='botsort.yaml',
    save=True
)
```

### 5. Dashboard Visualization
The dashboard provides interactive visualization of match data. To customize:
- Edit `dashboard/Dashboard.py`
- Update `dashboard/Match_Daten.csv` with new data
- Restart the Streamlit server

## Model Training

### Training Data Structure
Organize training data in YOLO format:
```
training_ball/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### Example data.yaml
```yaml
path: training_ball
train: images/train
val: images/val

names:
  0: tennis_ball
```

### Training Commands
```bash
# Train ball detector
yolo detect train data=training_ball/data.yaml model=yolo11n.pt epochs=100

# Train racket detector
yolo detect train data=training_racket/data.yaml model=yolo11n.pt epochs=100

# Train court detector
yolo detect train data=training_court/data.yaml model=yolo11n.pt epochs=100
```

### Training Tips
- Start with pre-trained weights for faster convergence
- Use data augmentation for better generalization
- Monitor validation metrics to prevent overfitting
- Adjust confidence thresholds based on use case
- Consider ensemble models for improved accuracy

## Results

### Model Performance
Detailed performance metrics and visualizations can be found in:
- `try_ml/ml_test_tennis.ipynb` - Model evaluation notebook
- `runs/detect/train*/` - Training logs and metrics
- Dashboard - Real-time match statistics

### Example Outputs
- Bounding box detection on rackets, balls, and players
- Court line detection and homography transformation
- Player pose estimation and movement tracking
- Match statistics (rally length, shot speed, court coverage)

## Project Structure Recommendations

For better organization, consider:
- Adding a `docs/` folder for detailed documentation
- Creating a `scripts/` folder for automation scripts
- Adding `tests/` for unit and integration tests
- Including sample data in `data/samples/` for quick testing

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines
- Follow PEP 8 style guide for Python code
- Add docstrings to all functions and classes
- Update documentation for new features
- Include tests for new functionality
- Ensure all tests pass before submitting PR

## License

This project is licensed under the MIT License - see the LICENSE file for details.

**Project Link**: [https://github.com/yourusername/CDA2-3DA_TENNIS-CHALLENGE](https://github.com/yourusername/CDA2-3DA_TENNIS-CHALLENGE)

## Acknowledgments

- Ultralytics for the YOLO implementation
- Streamlit for the dashboard framework
- Tennis match footage providers
- Open source community contributors

---

**Note**: This project is for educational and research purposes. Ensure you have appropriate permissions for any tennis match footage used.
