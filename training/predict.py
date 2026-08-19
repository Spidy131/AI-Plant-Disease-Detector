import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from config import MODEL_PATH, IMAGE_SIZE

# Class names (same order as training)
CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Load model
model = tf.keras.models.load_model(MODEL_PATH)


def predict_image(image_path):
    img = image.load_img(image_path, target_size=IMAGE_SIZE)

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(predictions)

    confidence = float(predictions[0][predicted_index] * 100)
    return (
        CLASS_NAMES[predicted_index],
        confidence
    )