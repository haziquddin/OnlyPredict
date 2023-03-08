import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from keras_preprocessing.image import load_img
from flask import Flask, render_template, request

custom_objects = {'KerasLayer': hub.KerasLayer}

app = Flask(__name__)

new_model = keras.models.load_model('./models/mdl_ver_01_TL.h5', custom_objects=custom_objects)


@app.route('/', methods=['GET'])
def hello():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    image_file = request.files['image_file']
    image_path = "./images/" + image_file.filename
    image_file.save(image_path)

    image = load_img(image_path)
    tf_image = np.array(image)
    resize = tf.image.resize(tf_image, (224, 224))
    yhat = new_model.predict(np.expand_dims(resize/255, 0))

    labels = ['Dermatitis', 'Eczema', 'Melanoma', 'Psoriasis']
    label = labels[np.argmax(yhat)]
    classification = label

    if (np.argmax(yhat) == 0):
        return render_template('Dermatitis.html', prediction=yhat)
    elif (np.argmax(yhat) == 1):
        return render_template('Eczema.html', prediction=yhat)
    elif (np.argmax(yhat) == 2):
        return render_template('Melanoma.html', prediction=yhat)
    else:
        return render_template('Psoriasis.html', prediction=yhat)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
