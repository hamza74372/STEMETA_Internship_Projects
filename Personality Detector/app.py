# app.py
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load your trained model and LabelEncoder (assume you saved them)
model = joblib.load('rf_personality_model.joblib')
le = joblib.load('label_encoder.joblib')

# Feature list grouped (example grouping)
feature_groups = {
    "Social Traits": [
        ("social_energy", "Energy in social situations"),
        ("alone_time_preference", "Preference to be alone"),
        ("talkativeness", "How talkative you are"),
        ("group_comfort", "Comfort in groups"),
        ("party_liking", "Enjoyment of parties"),
        ("listening_skill", "Ability to listen well"),
        ("online_social_usage", "Social media usage")
    ],
    "Emotional Traits": [
        ("empathy", "Ability to understand others"),
        ("deep_reflection", "Tendency to reflect deeply"),
        ("emotional_stability", "Emotional balance"),
        ("stress_handling", "Ability to handle stress"),
        ("friendliness", "Friendliness")
    ],
    "Personality & Lifestyle": [
        ("creativity", "Creativity level"),
        ("organization", "Organization skills"),
        ("planning", "Planning tendency"),
        ("routine_preference", "Preference for routine"),
        ("spontaneity", "Spontaneous nature"),
        ("adventurousness", "Willingness to take risks"),
        ("risk_taking", "Level of risk-taking"),
        ("reading_habit", "Interest in reading"),
        ("sports_interest", "Interest in sports"),
        ("travel_desire", "Desire to travel"),
        ("gadget_usage", "Usage of gadgets"),
        ("work_style_collaborative", "Preference for teamwork"),
        ("decision_speed", "Speed of decision-making"),
        ("leadership", "Leadership qualities"),
        ("public_speaking_comfort", "Comfort speaking publicly"),
        ("curiosity", "Curiosity level"),
        ("excitement_seeking", "Seeking excitement")
    ]
}


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Collect inputs from sliders
        input_data = []
        inputs_display = {}
        for group, features in feature_groups.items():
            for feature, desc in features:
                val = float(request.form.get(feature, 5))  # Default 5 if missing
                inputs_display[feature] = val
                input_data.append(val)

        # Predict
        X = np.array(input_data).reshape(1, -1)
        pred_encoded = model.predict(X)[0]
        pred_proba = model.predict_proba(X)[0]
        pred_label = le.inverse_transform([pred_encoded])[0]
        
        # Format probabilities nicely
        proba_dict = {le.inverse_transform([i])[0]: f"{prob*100:.2f}%" for i, prob in enumerate(pred_proba)}

        return render_template('index.html', 
                               feature_groups=feature_groups,
                               inputs=inputs_display,
                               prediction=pred_label,
                               probabilities=proba_dict)

    # GET method, just render form with defaults
    return render_template('index.html', feature_groups=feature_groups)

if __name__ == '__main__':
    app.run(debug=True)
