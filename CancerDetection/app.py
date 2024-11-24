from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model("oral_cancer_detection_model.h5")

# Define the folder to save uploaded images temporarily
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if the uploaded file is an image
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess the image
        img = load_img(filepath, target_size=(224, 224))  # Resize to model input size
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0) / 255.0  # Scale pixel values if required

        # Predict using the loaded model
        prediction = model.predict(img)
        class_label = 'Cancerous' if prediction[0][0] > 0.3 else 'Normal'

        return render_template('index.html', label=class_label, filepath=filepath)

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
