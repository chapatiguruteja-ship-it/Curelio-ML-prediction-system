from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("decision_tree_model.pkl")

# Load label encoders
encoders = joblib.load("label_encoders.pkl")

insurance_encoder = encoders["insurance"]
city_encoder = encoders["city"]
hospital_encoder = encoders["hospital"]
policy_encoder = encoders["policy"]
treatment_encoder = encoders["treatment"]
target_encoder = encoders["target"]

insurance_list = list(insurance_encoder.classes_)
city_list = list(city_encoder.classes_)
hospital_list = list(hospital_encoder.classes_)
policy_list = list(policy_encoder.classes_)
treatment_list = list(treatment_encoder.classes_)
@app.route("/")
def home():
    return render_template("index.html", insurance_list=insurance_list, city_list=city_list, hospital_list=hospital_list, policy_list=policy_list, treatment_list=treatment_list)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        Insurance_Provider = request.form["Insurance_Provider"].strip().lower()
        city = request.form["city"].strip().lower()
        hospital = request.form["hospital"].strip().lower()
        policy = request.form["policy"].strip().lower()
        treatment = request.form["treatment"].strip().lower()
        claim_amount = float(request.form["claim_amount"])

        insurance = insurance_encoder.transform([Insurance_Provider])[0]
        city = city_encoder.transform([city])[0]
        hospital = hospital_encoder.transform([hospital])[0]
        policy = policy_encoder.transform([policy])[0]
        treatment = treatment_encoder.transform([treatment])[0]

        input_data = pd.DataFrame(
            [[
                insurance,
                city,
                hospital,
                policy,
                treatment,
                claim_amount
            ]],
            columns=[
                "Insurance_Provider",
                "City",
                "Hospital_Name",
                "Policy_Type",
                "Treatment",
                "Claim_Amount"
            ]
        )

        prediction = model.predict(input_data)[0]
        confidence = round(max(model.predict_proba(input_data)[0]) * 100, 2)

        result = target_encoder.inverse_transform([prediction])[0]

        return render_template(
            "index.html",
            prediction=result,
            confidence=confidence
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction="Prediction Failed",
            confidence=0,
            error=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)