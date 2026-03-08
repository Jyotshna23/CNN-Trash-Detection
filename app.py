# CNN Trash Detection Website - Flask Backend

import os
import json
import pickle
import datetime
import sqlite3
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(BASE_DIR, 'model')
FRONTEND_DIR = BASE_DIR
DB_PATH      = os.path.join(BASE_DIR, 'database', 'detections.db')

# Load Model
model       = None
class_names = None

def load_model():
    global model, class_names
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'trash_model.h5'))
        with open(os.path.join(MODEL_DIR, 'class_names.pkl'), 'rb') as f:
            class_names = pickle.load(f)
        print("Model loaded successfully.")
        print(f"Classes: {class_names}")
    except Exception as e:
        print(f"Model loading error: {e}")
        class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

load_model()

# Database Setup
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        predicted_class TEXT,
        confidence REAL,
        is_recyclable INTEGER,
        created_at TEXT
    )''')
    conn.commit()
    conn.close()

init_db()
print("Database initialized.")

# Recyclable classes
RECYCLABLE = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

CLASS_INFO = {
    'cardboard': {
        'recyclable': True,
        'description': 'Cardboard and paper boxes are highly recyclable.',
        'disposal': 'Flatten and place in recycling bin.',
        'color': '#8B6914',
        'tips': ['Remove tape and staples', 'Keep dry', 'Flatten boxes']
    },
    'glass': {
        'recyclable': True,
        'description': 'Glass bottles and jars can be recycled indefinitely.',
        'disposal': 'Rinse and place in glass recycling.',
        'color': '#2196F3',
        'tips': ['Rinse thoroughly', 'Remove lids', 'Separate by color if required']
    },
    'metal': {
        'recyclable': True,
        'description': 'Metal cans and aluminum are valuable recyclables.',
        'disposal': 'Rinse cans and place in recycling.',
        'color': '#9E9E9E',
        'tips': ['Rinse food residue', 'Crush cans to save space', 'Remove labels if possible']
    },
    'paper': {
        'recyclable': True,
        'description': 'Paper products like newspapers and magazines are recyclable.',
        'disposal': 'Keep dry and place in paper recycling.',
        'color': '#4CAF50',
        'tips': ['Keep dry', 'Remove plastic covers', 'Bundle newspapers together']
    },
    'plastic': {
        'recyclable': True,
        'description': 'Many plastic items are recyclable — check the number.',
        'disposal': 'Check recycling number (1-7) and sort accordingly.',
        'color': '#FF9800',
        'tips': ['Check plastic number', 'Rinse containers', 'Remove caps if different material']
    },
    'trash': {
        'recyclable': False,
        'description': 'This item is general waste and goes to landfill.',
        'disposal': 'Place in general waste bin.',
        'color': '#F44336',
        'tips': ['Consider reducing waste', 'Look for alternatives', 'Compost food waste']
    }
}

# Routes
@app.route('/')
def home():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/detect.html')
def detect_page():
    return send_from_directory(FRONTEND_DIR, 'detect.html')

@app.route('/result.html')
def result_page():
    return send_from_directory(FRONTEND_DIR, 'result.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)

# API - Predict
@app.route('/api/detect', methods=['POST'])
def detect():
    try:
        from PIL import Image
        import io

        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image uploaded'})

        file = request.files['image']
        img_bytes = file.read()

        # Preprocess image
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if model is not None:
            predictions = model.predict(img_array, verbose=0)[0]
            pred_idx    = int(np.argmax(predictions))
            confidence  = float(predictions[pred_idx]) * 100
            pred_class  = class_names[pred_idx]

            top3 = sorted(
                [(class_names[i], round(float(predictions[i]) * 100, 1)) for i in range(len(class_names))],
                key=lambda x: x[1], reverse=True
            )[:3]
        else:
            # Demo mode
            import random
            pred_class = random.choice(class_names)
            confidence = round(random.uniform(70, 95), 1)
            top3 = [(pred_class, confidence)]

        is_recyclable = pred_class in RECYCLABLE
        info = CLASS_INFO.get(pred_class, {})

        # Save to DB
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('INSERT INTO detections (predicted_class, confidence, is_recyclable, created_at) VALUES (?,?,?,?)',
                  (pred_class, confidence, 1 if is_recyclable else 0, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
        conn.close()

        return jsonify({
            'success': True,
            'predicted_class': pred_class,
            'confidence': round(confidence, 1),
            'is_recyclable': is_recyclable,
            'top3': top3,
            'info': info
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# API - Stats
@app.route('/api/stats', methods=['GET'])
def stats():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM detections')
        total = c.fetchone()[0]
        c.execute('SELECT COUNT(*) FROM detections WHERE is_recyclable=1')
        recyclable = c.fetchone()[0]
        c.execute('SELECT predicted_class, COUNT(*) as cnt FROM detections GROUP BY predicted_class ORDER BY cnt DESC LIMIT 6')
        breakdown = {row[0]: row[1] for row in c.fetchall()}
        conn.close()
        return jsonify({'total': total, 'recyclable': recyclable, 'breakdown': breakdown})
    except Exception as e:
        return jsonify({'total': 0, 'recyclable': 0, 'breakdown': {}})

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'running', 'model_loaded': model is not None})

if __name__ == '__main__':
    print("=" * 50)
    print("CNN Trash Detection Server Starting...")
    print("  Open: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
