from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load saved model
model = joblib.load("insurance_rf_model.pkl")

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Get form data
        age = float(request.form["age"])
        sex = 1 if request.form["sex"] == "male" else 0
        bmi = float(request.form["bmi"])
        children = float(request.form["children"])
        smoker = 1 if request.form["smoker"] == "yes" else 0
        
        # Region one-hot encoding (assume regions: northeast, northwest, southeast, southwest)
        region = request.form["region"]
        region_northwest = 1 if region == "northwest" else 0
        region_southeast = 1 if region == "southeast" else 0
        region_southwest = 1 if region == "southwest" else 0

        # Feature order must match training
        features = np.array([[age, sex, bmi, children, smoker, region_northwest, region_southeast, region_southwest]])

        # Predict
        pred = model.predict(features)[0]
        pred = round(pred, 2)

        return render_template("index.html", prediction=pred)

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
