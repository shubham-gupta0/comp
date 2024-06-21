from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from io import BytesIO

app = Flask(__name__)

def preprocess_image(image, target_size=(128, 128)):
    try:
        image = Image.open(image)
        image = image.convert('RGB')  # Convert to RGB
        image = image.resize(target_size, Image.LANCZOS)
        image = img_to_array(image)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def load_mobilenet_model(model_path):
    return load_model(model_path)

@app.route('/compare', methods=['POST'])
def compare_images():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({"error": "Please provide both image1 and image2 files"}), 400

    image1 = request.files['image1'].read()
    image2 = request.files['image2'].read()

    image1 = preprocess_image(BytesIO(image1))
    image2 = preprocess_image(BytesIO(image2))

    if image1 is None or image2 is None:
        return jsonify({"error": "Error in preprocessing images"}), 400

    image1 = np.expand_dims(image1, axis=0)  # Add batch dimension
    image2 = np.expand_dims(image2, axis=0)  # Add batch dimension

    similarity = model.predict([image1, image2])

    # Assuming the model outputs a similarity score where higher values indicate greater similarity
    similarity_score = similarity[0][0]

    # Determine if images are of the same muzzle or not based on a threshold
    threshold = 0.5  # You may need to adjust this threshold based on your model's performance
    if similarity_score > threshold:
        result = "Same muzzle"
    else:
        result = "Different muzzles"

    return jsonify({"result": result, "similarity_score": similarity_score})

if __name__ == '__main__':
    # Define the path to the model
    model_path = "muzzle_model.h5"

    # Load the MobileNetV2 model
    model = load_mobilenet_model(model_path)

    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)
