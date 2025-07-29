from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the saved model, scaler, and feature names
model = joblib.load('logistic_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')

@app.route('/')
def home():
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from form and convert to float
        input_values = [float(request.form[feature]) for feature in features]
        input_df = pd.DataFrame([input_values], columns=features)

        # Scale the input features
        input_scaled = scaler.transform(input_df)

        # Make prediction
        pred = model.predict(input_scaled)[0]
        result = "Heart Disease Detected" if pred == 1 else "No Heart Disease"

    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template('index.html', features=features, prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
