import os
from flask import Flask, redirect, render_template, request
from PIL import Image

import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.models import load_model



class_names = ['Pepper__bell___Bacterial_spot',
 'Pepper__bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']

model = load_model('resnet_model.h5')
print(model.summary())



def predict(image_path):
    image = Image.open(image_path)
    image = image.resize((256, 256))
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class


app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads/', filename)
        image.save(file_path)
        print(file_path)
        pred = predict(file_path)
        return render_template('submit.html' , title = pred, image_url = file_path )



if __name__ == '__main__':
    app.run(debug=True)
