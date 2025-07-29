from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load your saved model and encoded feature names (list of strings)
model = joblib.load('random_forest_model.pkl')
model_features = joblib.load('feature_names.pkl')

# Define your original categorical features and their categories (replace or extend as needed)
categorical_features = {
    'BusinessTravel': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],
    'Department': ['Sales', 'Research & Development', 'Human Resources'],
    'Education': ['2', '3', '4', '5'],  # dropped first category because of drop_first=True (so '1' is baseline)
    'EducationField': ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'],
    'EnvironmentSatisfaction': ['2', '3', '4'],  # drop_first=True drops '1'
    'Gender': ['Male'],  # 'Female' dropped as baseline
    'JobInvolvement': ['2', '3', '4'],
    'JobRole': ['Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative',
                'Manager', 'Sales Representative', 'Research Director', 'Human Resources'],
    'JobSatisfaction': ['2', '3', '4'],
    'MaritalStatus': ['Married', 'Single'],  # 'Divorced' baseline
    'OverTime': ['Yes'],  # 'No' baseline
    'PerformanceRating': ['2', '3', '4'],
    'RelationshipSatisfaction': ['2', '3', '4'],
    'WorkLifeBalance': ['2', '3', '4']
}

# Numeric features you kept (replace or add more if needed)
numeric_features = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'HourlyRate', 
    'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 
    'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel',
    'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

# Build the form fields for rendering
form_fields = []

for nf in numeric_features:
    form_fields.append({'name': nf, 'type': 'number'})

for cat_feat, categories in categorical_features.items():
    form_fields.append({'name': cat_feat, 'type': 'select', 'options': categories})

def create_input_vector(form_data):
    input_dict = dict.fromkeys(model_features, 0)
    
    # Numeric features: directly from form, convert to float if possible
    for nf in numeric_features:
        val = form_data.get(nf)
        if val is not None and val != '':
            try:
                input_dict[nf] = float(val)
            except:
                input_dict[nf] = 0
        else:
            input_dict[nf] = 0
    
    # Categorical features: map to dummy columns
    for cat_feat, categories in categorical_features.items():
        selected = form_data.get(cat_feat)
        if selected:
            dummy_col = f"{cat_feat}_{selected}"
            if dummy_col in input_dict:
                input_dict[dummy_col] = 1
    
    # Return numpy array in correct column order
    input_vector = np.array([input_dict[feat] for feat in model_features]).reshape(1, -1)
    return input_vector

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', form_fields=form_fields, prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_vector = create_input_vector(request.form)
        pred = model.predict(input_vector)[0]
        prediction = 'Yes (Will Leave)' if pred == 1 else 'No (Will Stay)'
    except Exception as e:
        prediction = f"Error: {str(e)}"
    
    return render_template('index.html', form_fields=form_fields, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
