import os

import numpy as np
from flask import Flask, request, render_template
from keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model('model.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

labels = {0: 'Healthy', 1: 'Infected'}


def get_result(image_path):
    img = load_img(image_path, target_size=(225, 225))
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    return predictions


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        base_path = os.path.dirname(__file__)
        file_path = os.path.join(
            base_path, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        predictions = get_result(file_path)
        predicted_label = labels[np.argmax(predictions)]
        return str(predicted_label)
    return None


if __name__ == '__main__':
    app.run(debug=True)
