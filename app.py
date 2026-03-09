from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model pipeline
MODEL_PATH = "model.pkl"
if os.path.exists(MODEL_PATH):
    model = pickle.load(open(MODEL_PATH, "rb"))
else:
    model = None
    print("Warning: model.pkl not found. Please train the model first.")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/assessment")
def assessment():
    return render_template("assessment.html")

@app.route("/insights")
def insights():
    return render_template("insights.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        # Extract data from request
        data = request.json
        
        # Mapping for employment type
        emp_type_map = {
            "Salaried": 0,
            "Self-Employed": 1,
            "Freelancer": 2
        }
        
        annual_income = float(data.get("annual_income", 0))
        loan_amount = float(data.get("loan_amount", 0))
        credit_score = float(data.get("credit_score", 0))
        employment_type_str = data.get("employment_type", "Salaried")
        employment_type = emp_type_map.get(employment_type_str, 0)
        years_employment = float(data.get("years_employment", 0))
        existing_credit_lines = int(data.get("existing_credit_lines", 0))

        # Create feature array
        features = np.array([[
            annual_income,
            loan_amount,
            credit_score,
            employment_type,
            years_employment,
            existing_credit_lines
        ]])

        # Predict probability
        # The pipeline automatically applies the scaler
        probabilities = model.predict_proba(features)[0]
        
        # Assume class 1 is default/high-risk
        # Class 1 probability
        risk_probability = probabilities[1]

        # Determine risk category
        if risk_probability < 0.3:
            risk_level = "Low Risk"
        elif risk_probability <= 0.6:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        return jsonify({
            "probability": round(risk_probability * 100, 2),
            "risk_level": risk_level
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
