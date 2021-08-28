from flask import Flask, request
from flask import render_template, jsonify
import base64
import json
from PIL import Image
import numpy as np
from keras.models import load_model
import tensorflow as tf
import os


dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = dir_path + '/uploads'
STATIC_FOLDER = dir_path + '/static'

graph = tf.get_default_graph()
with graph.as_default():
   model = load_model(STATIC_FOLDER + '/' + 'mnist_cnn.h5')

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():

    if request.method == 'GET':
        return render_template('index.html')
    else:

        data = request.form['image']
        data = data.replace('data:image/png;base64,','')
        data = data.replace(' ', '+')
        imgdata = base64.b64decode(data)
        filename = UPLOAD_FOLDER + '/test.jpg'
        with open(filename, 'wb') as f:
            f.write(imgdata)

        result = predict(filename)
        predicted_number = np.asscalar(np.argmax(result, axis=1))
        accuracy = round(result[0][predicted_number] * 100, 2)

        return jsonify({'label':predicted_number},)



def predict(filename):
    img = Image.open(filename)
    img = img.resize((28, 28), Image.ANTIALIAS)
    img = np.asarray(img)
    data = img[:, :, 3] # select only visible channel

    data = np.expand_dims(data, axis=2)
    data = np.expand_dims(data, axis=0)

    with graph.as_default():
        predicted = model.predict(data)

    return predicted


if __name__ == '__main__':
    app.debug = True
    app.run(debug=True)
