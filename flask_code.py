import os
import numpy as np
from io import BytesIO
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
# Load the saved model
model = load_model(".\Trin_Model.h5")

# Define the allowed image extensions
ALLOWED_EXTENSIONS = {"png"}


# Function to check if the uploaded file has an allowed extension
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Define the route for the home page
@app.route("/")
def home():
    template_path = os.path.join(os.getcwd(), "templates", "index.html")
    if not os.path.exists(template_path):
        return jsonify({"error": "Template not found"})
    return render_template("index.html")


# Define the route for the prediction API
@app.route("/predict", methods=["POST"])
def predict():
    # Check if a file was uploaded
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    # Get the uploaded file
    file = request.files["file"]

    # load image
    img = load_img(BytesIO(file.read()), target_size=(180, 180))
    img = img_to_array(img)
    img = img / 255.0
    img = img.reshape(1, 180, 180, 3)

    predictions = model.predict(img)
    predicted_label = np.argmax(predictions)

    # Get the predicted label and probability
    # Define a dictionary mapping class index to class name
    class_dict = {
        1: "Atelectasis",
        2: "Cardiomegaly",
        3: "Consolidation",
        4: "Edema",
        5: "Effusion",
        6: "Emphysema",
        7: "Fibrosis",
        8: "Hernia",
        9: "Infiltration",
        10: "Mass",
        11: "No Finding",
        12: "Nodule",
        13: "Pleural_Thickening",
        14: "Pneumonia",
        15: "Pneumothorax",
    }

    predicted_label = class_dict.get(predicted_label)
    predicted_probability = str(predictions.max())

    # Return the predicted label and probability as JSON
    return jsonify({"label": predicted_label, "probability": predicted_probability})


if __name__ == "__main__":
    app.run()
