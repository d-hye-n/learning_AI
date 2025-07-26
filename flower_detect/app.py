import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Load the trained model ---
model = tf.keras.models.load_model('flower_model.h5')
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# --- Image Preprocessing ---
def preprocess_image(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    return img_array

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join('uploads', filename)
            if not os.path.exists('uploads'):
                os.makedirs('uploads')
            file.save(filepath)

            # Preprocess the image and predict
            processed_image = preprocess_image(filepath)
            predictions = model.predict(processed_image)
            score = tf.nn.softmax(predictions[0])
            predicted_class = class_names[np.argmax(score)]
            confidence = 100 * np.max(score)

            return render_template('index.html', filename=filename, prediction=predicted_class, confidence=f"{confidence:.2f}")
    return render_template('index.html', filename=None, prediction=None, confidence=None)

@app.route('/uploads/<filename>')
def send_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == '__main__':
    app.run(debug=True)