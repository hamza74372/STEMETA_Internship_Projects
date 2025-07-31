from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')  # Load your Random Forest model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form and convert to float
        features = [float(request.form.get(col)) for col in [
            'Year', 'Adult Mortality', 'infant deaths', 'Alcohol',
            'percentage expenditure', 'Hepatitis B', 'Measles', 'BMI',
            'under-five deaths', 'Polio', 'Total expenditure', 'Diphtheria',
            'HIV/AIDS', 'GDP', 'Population', 'thinness 1-19 years',
            'thinness 5-9 years', 'Income composition of resources', 'Schooling'
        ]]
        prediction = model.predict([features])[0]
        output = f"{prediction:.2f} years"
    except Exception as e:
        output = f"Error: {str(e)}"
    return render_template('index.html', prediction_text=output)

if __name__ == '__main__':
    app.run(debug=True)
