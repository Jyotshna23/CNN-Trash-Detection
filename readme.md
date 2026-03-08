# CNN Trash Detection Website — Deep Learning Waste Classification

A CNN-based waste classification web application that identifies trash and recycling materials from images using Transfer Learning (MobileNetV2). Built with Python, TensorFlow, and Flask.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)
![Accuracy](https://img.shields.io/badge/Accuracy-74%25-brightgreen?style=flat-square)
![Status](https://img.shields.io/badge/Status-Live-success?style=flat-square)

---

## Overview

TrashAI uses a fine-tuned MobileNetV2 CNN trained on the TrashNet dataset to classify waste images into 6 categories and provide recycling guidance. Users upload an image and receive instant classification with disposal instructions.

---

## Features

- Image upload with drag-and-drop support
- CNN model inference with confidence scores
- Top 3 predictions with probability breakdown
- Recycling guidance, disposal tips, and category info
- Detection history dashboard with Chart.js visualizations
- SQLite storage for detection logs

---

## Waste Categories

| Category | Recyclable | Description |
|----------|-----------|-------------|
| Cardboard | Yes | Boxes, packaging |
| Glass | Yes | Bottles, jars |
| Metal | Yes | Cans, tins |
| Paper | Yes | Newspapers, magazines |
| Plastic | Yes | Bottles, containers |
| Trash | No | General waste |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | TensorFlow, Keras, MobileNetV2 |
| Training | Google Colab, TrashNet Dataset |
| Backend | Python, Flask, Flask-CORS |
| Database | SQLite |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Charts | Chart.js |

---

## ML Model — How It Works

1. **Base Model** — MobileNetV2 pretrained on ImageNet
2. **Fine-tuning** — Last 30 layers unfrozen and retrained
3. **Input** — 224x224 RGB image
4. **Output** — 6-class softmax probability distribution
5. **Dataset** — TrashNet (2,527 images, 6 classes)

---

## Project Structure

```
cnn-trash-detection/
├── app.py                  # Flask backend + API
├── requirements.txt
├── .gitignore
├── frontend/
│   ├── index.html          # Landing page
│   ├── detect.html         # Upload & detect
│   ├── dashboard.html      # Detection history
│   ├── style.css
│   └── main.js
└── model/
    ├── trash_model.h5      # Trained CNN model
    └── class_names.pkl     # Class labels
```

---

## Setup and Installation

```bash
git clone https://github.com/Jyotshna23/cnn-trash-detection.git
cd cnn-trash-detection
pip install -r requirements.txt
python app.py
# Open: http://localhost:5000
```

---

## Author

**Jyotshna Pogiri**
- GitHub: [Jyotshna23](https://github.com/Jyotshna23)
