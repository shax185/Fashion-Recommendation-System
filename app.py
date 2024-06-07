from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import pandas as pd
import pickle
import tensorflow
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import numpy as np



app = Flask(__name__)

# Load data and models
embeddings = np.array(pickle.load(open('embeddings.pkl', 'rb')))
myntra = pd.read_csv('myntra.csv', index_col=0)
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


def feature_extraction(img_array, model):
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / np.linalg.norm(result)
    return normalized_result

def find_nearest_neighbors(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

def calculate_body_shape(bust, waist, hip):
    body_shape = ""

    # Ensure measurements are valid floats
    try:
        bust = float(bust)
        waist = float(waist)
        hip = float(hip)
    except ValueError:
        return "Error: Please enter valid numbers for measurements."

    # Calculate body shape based on ratios
    if waist * 1.25 <= bust and waist <= hip:
        body_shape = "Hourglass"
    elif hip * 1.05 > bust:
        body_shape = "Pear"
    elif hip * 1.05 < bust:
        body_shape = "Apple"
    else:
        high = max(bust, waist, hip)
        low = min(bust, waist, hip)
        difference = high - low
        if difference <= 5:
            body_shape = "Rectangle"

    return body_shape

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/bodyshapecalculator', methods=['GET', 'POST'])
def bodyshapecalculator():
    if request.method == 'POST':
        bust = float(request.form['bust'])
        waist = float(request.form['waist'])
        hip = float(request.form['hip'])

        # Calculate body shape
        body_shape = calculate_body_shape(bust, waist, hip)

        # Return the result as JSON
        return jsonify({'body_shape': body_shape})
    else:
        return render_template('bodyshapecalculator.html')

@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation():
    if request.method == 'POST':
        # Get the image file from the request
        image_file = request.files['image']
        # Process the image
        img = Image.open(io.BytesIO(image_file.read()))
        img = img.resize((224, 224))  # Resize image to match model input size
        img_array = np.asarray(img)  # Convert PIL Image to NumPy array
        # Extract features from the image
        features = feature_extraction(img_array, model)
        # Flatten the features array
        features_flat = features.flatten()
        # Retrieve indices of nearest neighbors
        indices = find_nearest_neighbors(features_flat, embeddings)
        # Flatten the indices array (if necessary)
        indices_flat = indices.flatten()
        # Get recommendations from the dataset
        recommendations = myntra.iloc[indices_flat]
        
        # Convert recommendations DataFrame to a list of dictionaries
        recommendation_list = recommendations.to_dict(orient='records')
        # Pass the recommendation list to the template
        return render_template('recommendation.html', recommendations=recommendation_list, myntra=myntra)
    
    # If it's a GET request, render the recommendation.html template without recommendations
    return render_template('recommendation.html', recommendations=None, myntra=myntra)


if __name__ == '__main__':
    app.run(debug=True)
