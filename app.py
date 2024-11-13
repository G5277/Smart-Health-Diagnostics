from flask import Flask, request, render_template, jsonify
import numpy as np

app = Flask(__name__)

import joblib
def load_meta_model():
    try:
        # Load the model using joblib
        model = joblib.load("stacked_meta_model.joblib")
        print("Model loaded successfully!")
        return model
    except FileNotFoundError:
        print("Error: Model file not found.")
    except joblib.externals.loky.backend.exceptions.UnpicklingError:
        print("Error: File could not be unpickled. It might be corrupted.")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")

from tensorflow.keras.models import load_model # type: ignore
def load_ann_model():
    try:
        # Load the Keras model
        model = load_model("ann_model.keras")
        print("ANN model loaded successfully!")
        return model
    except FileNotFoundError:
        print("Error: ANN model file not found.")
    except Exception as e:
        print(f"An error occurred while loading the ANN model: {e}")

import xgboost as xgb
def load_xgb_model():
    try:
        model = xgb.Booster()
        model.load_model("xgb_model.json")
        print("XGBoost model loaded successfully!")
        return model
    except FileNotFoundError:
        print("Error: XGBoost model file not found.")
    except xgb.core.XGBoostError:
        print("Error: Failed to load XGBoost model.")
    except Exception as e:
        print(f"An error occurred while loading the XGBoost model: {e}")

# Example of loading models
ann_model = load_ann_model()
xgb_model = load_xgb_model()
meta_model = load_meta_model()

@app.route('/', methods=['GET','POST'])
def home():
    prediction = 0
    if request.method == 'POST':
        # Collect input data from the form
        user_input = request.form
        # Collecting normal inputs
        age = int(user_input['age'])
        gender = 1 if user_input['gender'] == 'Male' else 0  # Example conversion
        bmi = float(user_input['bmi'])
        systolic_bp = float(user_input['systolic_bp'])
        diastolic_bp = float(user_input['diastolic_bp'])
        fasting_blood_sugar = float(user_input['fasting_blood_sugar'])

        # Collecting checkboxes for disease categories (1 if selected, 0 if not)
        # Update each category based on the new mapping
        heart_related = 0 if user_input['heart_related'] == 'None' else (1 if user_input['heart_related'] == 'FamIss' else 2)
        liver_related = 0 if user_input['liver_related'] == 'None' else (1 if user_input['liver_related'] == 'FamIss' else 2)
        kidney_related = 0 if user_input['kidney_related'] == 'None' else (1 if user_input['kidney_related'] == 'FamIss' else 2)
        lung_related = 0 if user_input['lung_related'] == 'None' else (1 if user_input['lung_related'] == 'FamIss' else 2)
        mental_health_related = 0 if user_input['mental_health_related'] == 'None' else (1 if user_input['mental_health_related'] == 'FamIss' else 2)

        # Collecting Yes/No inputs for smoking status and alcohol consumption
        smoking_status_Yes = 1 if user_input['smoking_status'] == 'Yes' else 0
        alcohol_consumption_Light = 1 if user_input['alcohol_consumption'] == 'Light' else 0
        alcohol_consumption_Moderate = 1 if user_input['alcohol_consumption'] == 'Moderate' else 0
        alcohol_consumption_None = 1 if user_input['alcohol_consumption'] == 'None' else 0


        # Collecting further inputs for physical activity, diet, and mental wellbeing
        physical_activity_Light	= 1 if user_input['physical_activity'] == 'Light' else 0
        physical_activity_Sedentary = 1 if user_input['physical_activity'] == 'Sedentary' else 0

        diet_Good = 1 if user_input['diet'] == 'Good' else 0
        diet_Poor = 1 if user_input['diet'] == 'Poor' else 0

        mental_wellbeing_Good = 1 if user_input['mental_wellbeing'] == 'Good' else 0
        mental_wellbeing_Moderate = 1 if user_input['mental_wellbeing'] == 'Moderate' else 0

        # Preparing feature array for model prediction
        features = np.array([
            age, gender, bmi, systolic_bp, diastolic_bp, fasting_blood_sugar,
            heart_related, liver_related, kidney_related, lung_related, mental_health_related,
            smoking_status_Yes, alcohol_consumption_Light,	alcohol_consumption_Moderate,
            alcohol_consumption_None, physical_activity_Light, physical_activity_Sedentary,
            diet_Good, diet_Poor, mental_wellbeing_Good, mental_wellbeing_Moderate
            ]).reshape(1, -1)


        # features = np.array([0.419594,0,0.741502,1.517149,0.197836,1.887030,2,0,0,0,0,True,True,False,False,False,True,True,False,False,True]).reshape(1, -1)

        # Make prediction using the trained model
        pred1 = ann_model.predict(features)
        print(pred1)
        xgb_dmatrix_n = xgb.DMatrix(features)  # Convert data to DMatrix for XGBoost
        pred2 = xgb_model.predict(xgb_dmatrix_n)
        stacked_predictions_new = np.column_stack((pred1, pred2))
        pred = meta_model.predict(stacked_predictions_new)
        prediction = pred[0]
        prediction = 5
        # Assuming the model returns a category

    # Return prediction result to a results page
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)