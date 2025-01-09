from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import base64

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploaded_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
MODEL_PATH = 'garbage_classification_model.h5'
model = load_model(MODEL_PATH)

# Map class indices to class labels
class_labels = {
    0: "Battery",
    1: "Biological",
    2: "Brown Glass",
    3: "Cardboard",
    4: "Cloth",
    5: "Green Glass",
    6: "Metal",
    7: "Paper",
    8: "Plastic",
    9: "Shoe",
    10: "White Glass"
}

# Function to preprocess images from file path
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to preprocess images from Base64
def preprocess_base64_image(img_data):
    img = Image.open(io.BytesIO(base64.b64decode(img_data.split(',')[1])))
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Upload and classify route
@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    # Save uploaded image
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Preprocess and predict
        img_array = preprocess_image(file_path)
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = class_labels[predicted_class_index]
        confidence = predictions[0][predicted_class_index] * 100  # Convert to percentage

        # Render result
        return render_template(
            'result.html',
            predicted_class=predicted_class_label,
            image_path=file_path,
            confidence=round(float(confidence), 2)
        )

# Capture from camera route
@app.route('/capture', methods=['POST'])
def capture():
    try:
        image_data = request.json['image']
        img_array = preprocess_base64_image(image_data)  # Preprocess Base64 image
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = class_labels[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index] * 100)  # Convert to percentage and ensure type `float`

        return jsonify({
            'predicted_class': predicted_class_label,
            'confidence': round(confidence, 2)  # Confidence as a rounded float
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    # Create the upload directory if not exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
