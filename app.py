from flask import Flask, render_template, request, jsonify
import os
import cv2
import base64
from werkzeug.utils import secure_filename
from model_inference import load_models, process_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

# Создаем папку для загрузок
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Загружаем модели при старте
load_models()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

@app.route('/')
def index():
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])
def predict():
    """Обработка загруженного изображения."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            results = process_image(filepath)
            return jsonify(results)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            # Удаляем временный файл
            if os.path.exists(filepath):
                os.remove(filepath)
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/predict_test', methods=['POST'])
def predict_test():
    """Обработка тестового изображения по пути."""
    data = request.get_json()
    image_path = data.get('path')
    
    if not image_path or not os.path.exists(image_path):
        return jsonify({'error': 'Test image not found'}), 404
    
    try:
        results = process_image(image_path)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)