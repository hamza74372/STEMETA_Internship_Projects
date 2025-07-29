from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load('titanic_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            pclass = int(request.form['Pclass'])
            sex = 1 if request.form['Sex'] == 'female' else 0
            age = float(request.form['Age'])
            sibsp = int(request.form['SibSp'])
            parch = int(request.form['Parch'])
            fare = float(request.form['Fare'])
            embarked = request.form['Embarked']

            # One-hot encode Embarked
            embarked_q = 1 if embarked == 'Q' else 0
            embarked_s = 1 if embarked == 'S' else 0

            features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked_q, embarked_s]])

            pred = model.predict(features)[0]
            prediction = "Survived" if pred == 1 else "Did not survive"
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
