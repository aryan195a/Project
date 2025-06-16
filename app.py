from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load the pre-trained components
model = joblib.load("best_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")  # e.g., StandardScaler, ColumnTransformer
le = joblib.load("label_encoder.pkl")

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    predicted_label = None

    if request.method == "POST":
        try:
            # 1. Collect input from form
            input_data = {
                "Age": float(request.form["Age"]),
                "ChestPain": float(request.form["ChestPain"]),
                "MaxHR": float(request.form["MaxHR"]),
                "Thal": float(request.form["Thal"]),
                "Pregnancies": float(request.form["Pregnancies"]),
                "Glucose": float(request.form["Glucose"]),
                "BloodPressure": float(request.form["BloodPressure"]),
                "Insulin": float(request.form["Insulin"]),
                "BMI": float(request.form["BMI"]),
                "DiabetesPedigreeFunction": float(request.form["DiabetesPedigreeFunction"])
            }

            # 2. Convert input into a DataFrame
            sample_input = pd.DataFrame([input_data])

            # 3. Apply the same transformation using the stored preprocessor
            sample_input_transformed = preprocessor.transform(sample_input)

            # 4. Predict the encoded label
            predicted_encoded_label = model.predict(sample_input_transformed)[0]

            # 5. Decode it to human-readable format
            predicted_label = le.inverse_transform([predicted_encoded_label])[0]

            prediction = f"Encoded: {predicted_encoded_label}, Health Status: {predicted_label}"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
