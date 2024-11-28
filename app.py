from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np

# Load the saved model pipeline
model = joblib.load('best_model_pipeline.pkl')

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.get_json()

        if not data:
            return jsonify({'error': 'Request body must be valid JSON.'}), 400

        # List of features actually used for prediction
        required_features = [
            'Age', 'Time Since Injury (days)', 'Glasgow Coma Scale (GCS)',
            'Midline Shift (mm)', 'Edema Volume (mL)', 'Lesion Volume (ML)'
        ]

        # Check for missing features
        missing_features = [feature for feature in required_features if feature not in data]
        if missing_features:
            return jsonify({'error': f'Missing features: {", ".join(missing_features)}'}), 400

        # Create a NumPy array with only the required features
        input_data = np.array([[float(data[feature]) for feature in required_features]])

        # Predict recovery time using the loaded pipeline
        prediction = model.predict(input_data)

        return jsonify({'predicted_recovery_time': round(float(prediction[0]), 2)})

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
