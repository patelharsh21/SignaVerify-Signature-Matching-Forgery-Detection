

from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from io import BytesIO
from PIL import Image

app = Flask(__name__)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

model = tf.keras.models.load_model("savedModel/my_model_keras.keras", custom_objects={'euclidean_distance': euclidean_distance, "eucl_dist_output_shape":eucl_dist_output_shape})

def process(image1, image2):
    x = Image.open(BytesIO(image1)).resize((100, 100)).convert('RGB')
    x = image.img_to_array(x)
    x = tf.image.rgb_to_grayscale(x)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    y = Image.open(BytesIO(image2)).resize((100, 100)).convert('RGB')
    y = image.img_to_array(y)
    y = tf.image.rgb_to_grayscale(y)
    y = np.expand_dims(y, axis=0)
    y = y / 255.0

    y_pred = model.predict([x, y])
    y_pred = np.argmax(y_pred)
    return y_pred

@app.route('/', methods=['GET', 'POST'])
def home():
    note = ''
    if request.method == 'POST':
        if 'image1' not in request.files or 'image2' not in request.files:
            note = 'Please upload both images.'
        else:
            image1 = request.files['image1'].read()
            image2 = request.files['image2'].read()

            if image1 and image2:
                # Process the images
                res = process(image1, image2)
                if res == 1:
                    note = 'Forged Signature'
                else:
                    note = 'Real Signature'

    return render_template('home.html', note=note)

# if __name__ == '__main__':
#     app.run(debug=True)
