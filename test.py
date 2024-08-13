import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from io import BytesIO
from PIL import Image

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

# Load the model with custom objects
model = tf.keras.models.load_model(
    "savedModel/my_model_keras.keras",
    custom_objects={'euclidean_distance': euclidean_distance, "eucl_dist_output_shape": eucl_dist_output_shape}
)

def process(image1_path, image2_path):
    # Load and preprocess the first image
    x = Image.open(image1_path).resize((100, 100)).convert('RGB')
    x = image.img_to_array(x)
    x = tf.image.rgb_to_grayscale(x)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    # Load and preprocess the second image
    y = Image.open(image2_path).resize((100, 100)).convert('RGB')
    y = image.img_to_array(y)
    y = tf.image.rgb_to_grayscale(y)
    y = np.expand_dims(y, axis=0)
    y = y / 255.0

    # Predict using the model
    y_pred = model.predict([x, y])
    y_pred = np.argmax(y_pred)
    return y_pred

# Test the process function
image1_path = '/home/sahil/Desktop/SignaVerify-Signature-Matching-Forgery-Detection/signData/test/049/01_049.png'
image2_path = '/home/sahil/Desktop/SignaVerify-Signature-Matching-Forgery-Detection/signData/test/049/01_049.png'

result = process(image1_path, image2_path)
print("Output:", result)
print("Return type of output:", type(BytesIO(result)))
