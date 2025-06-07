from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load your trained Keras model
model = load_model('CNN_best_model.keras')

# Define the tumor class names in the same order as the model was trained
class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# Folder to store uploaded images
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Open and preprocess the image
    image = Image.open(filepath).convert("RGB")  # Change to "L" if you trained on grayscale
    image = image.resize((224, 224))             # Resize to match training input
    img_array = np.array(image) / 255.0          # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Predict using the model
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]) * 100)
    predicted_class = class_names[predicted_index]

    # Debug info (optional)
    print("Predictions:", predictions[0])
    print("Predicted Index:", predicted_index)
    print("Predicted Class:", predicted_class)

    return render_template(
        'result.html',
        image_path=filepath,
        predicted_class=predicted_class,
        confidence=round(confidence, 2),
        class_probabilities=dict(zip(class_names, map(lambda x: round(x * 100, 2), predictions[0])))
    )


if __name__ == '__main__':
    app.run(debug=True)
