from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("career_recommendation_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    skills = request.form["skills"]
    interest = request.form["interest"]
    personality = request.form["personality"]
    education = request.form["education"]

    input_df = pd.DataFrame([{
        "Skills": skills,
        "Interest": interest,
        "Personality": personality,
        "Education_Level": education
    }])

    predicted_career = model.predict(input_df)[0]

    return render_template("result.html", career=predicted_career)

if __name__ == "__main__":
    app.run(debug=True)
