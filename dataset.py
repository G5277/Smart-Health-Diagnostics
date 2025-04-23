import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import save_model

# Generate synthetic dataset
def generate_dataset(n_samples=1000):
    np.random.seed(42)
    
    age = np.random.randint(18, 80, n_samples)
    gender = np.random.randint(0, 2, n_samples)
    bmi = np.random.uniform(15, 40, n_samples)
    systolic_bp = np.random.uniform(90, 180, n_samples)
    diastolic_bp = np.random.uniform(60, 120, n_samples)
    fasting_blood_sugar = np.random.uniform(70, 200, n_samples)

    heart_related = np.random.randint(0, 3, n_samples)
    liver_related = np.random.randint(0, 3, n_samples)
    kidney_related = np.random.randint(0, 3, n_samples)
    lung_related = np.random.randint(0, 3, n_samples)
    mental_health_related = np.random.randint(0, 3, n_samples)

    smoking_status_Yes = np.random.randint(0, 2, n_samples)
    alcohol_consumption_Light = np.random.randint(0, 2, n_samples)
    alcohol_consumption_Moderate = np.random.randint(0, 2, n_samples)
    alcohol_consumption_None = np.random.randint(0, 2, n_samples)

    physical_activity_Light = np.random.randint(0, 2, n_samples)
    physical_activity_Sedentary = np.random.randint(0, 2, n_samples)

    diet_Good = np.random.randint(0, 2, n_samples)
    diet_Poor = np.random.randint(0, 2, n_samples)

    mental_wellbeing_Good = np.random.randint(0, 2, n_samples)
    mental_wellbeing_Moderate = np.random.randint(0, 2, n_samples)

    X = np.column_stack([
        age, gender, bmi, systolic_bp, diastolic_bp, fasting_blood_sugar,
        heart_related, liver_related, kidney_related, lung_related, mental_health_related,
        smoking_status_Yes, alcohol_consumption_Light, alcohol_consumption_Moderate,
        alcohol_consumption_None, physical_activity_Light, physical_activity_Sedentary,
        diet_Good, diet_Poor, mental_wellbeing_Good, mental_wellbeing_Moderate
    ])

    # Dummy target: Let's assume a classification with 6 health risk levels
    y = np.random.randint(0, 6, n_samples)
    return X, y

# Train ANN
def train_ann(X_train, y_train, X_val, y_val):
    ann = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(6, activation='softmax')  # 6 classes
    ])
    ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    ann.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=1)
    return ann

# Train XGBoost
def train_xgb(X_train, y_train, X_val, y_val):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = {
        'objective': 'multi:softprob',
        'num_class': 6,
        'eval_metric': 'mlogloss'
    }
    xgb_model = xgb.train(params, dtrain, num_boost_round=50, evals=[(dval, 'eval')], early_stopping_rounds=10, verbose_eval=False)
    return xgb_model

# Train Meta Model (Stacking)
def train_meta_model(ann_model, xgb_model, X_val, y_val):
    pred1 = ann_model.predict(X_val)
    pred2 = xgb_model.predict(xgb.DMatrix(X_val))

    # Use predicted probabilities (for each class)
    stacked_features = np.hstack((pred1, pred2))
    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(stacked_features, y_val)
    return meta_model

# Main training pipeline
if __name__ == "__main__":
    X, y = generate_dataset(2000)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training ANN model...")
    ann_model = train_ann(X_train, y_train, X_val, y_val)
    save_model(ann_model, "ann_model.keras")
    print("ANN model saved.")

    print("Training XGBoost model...")
    xgb_model = train_xgb(X_train, y_train, X_val, y_val)
    xgb_model.save_model("xgb_model.json")
    print("XGBoost model saved.")

    print("Training meta model...")
    meta_model = train_meta_model(ann_model, xgb_model, X_val, y_val)
    joblib.dump(meta_model, "stacked_meta_model.joblib")
    print("Meta model saved.")
