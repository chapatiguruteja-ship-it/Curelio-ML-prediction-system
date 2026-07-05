from flask import Flask, render_template, request
import joblib
import pandas as pd


app = Flask(__name__)


# Load model and encoders

model = joblib.load("decision_tree_model.pkl")
encoders = joblib.load("label_encoders.pkl")



@app.route("/")
def home():
    return render_template("index.html")



@app.route("/about")
def about():
    return render_template("about.html")



@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/predict", methods=["GET","POST"])
def predict():

    result = None


    # Dropdown values

    insurance_providers = encoders["insurance"].classes_
    cities = encoders["city"].classes_
    hospitals = encoders["hospital"].classes_
    policies = encoders["policy"].classes_
    treatments = encoders["treatment"].classes_



    if request.method == "POST":


        insurance = request.form["insurance"]
        city = request.form["city"]
        hospital = request.form["hospital"]
        policy = request.form["policy"]
        treatment = request.form["treatment"]
        amount = int(request.form["amount"])



        # Encoding

        insurance = encoders["insurance"].transform([insurance])[0]

        city = encoders["city"].transform([city])[0]

        hospital = encoders["hospital"].transform([hospital])[0]

        policy = encoders["policy"].transform([policy])[0]

        treatment = encoders["treatment"].transform([treatment])[0]



        input_data = [[
            insurance,
            city,
            hospital,
            policy,
            treatment,
            amount
        ]]


        prediction = model.predict(input_data)[0]


        if prediction == 1:
            result="eligible"

        else:
            result="not"



    return render_template(
        "prediction.html",

        result=result,

        insurance_providers=insurance_providers,

        cities=cities,

        hospitals=hospitals,

        policies=policies,

        treatments=treatments
    )








if __name__=="__main__":
    app.run()