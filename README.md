# Career Recommendation System

A machine learning web application that predicts the most suitable career path for a student based on their skills, interests, personality type, and education level.

## What It Does

The system takes four inputs from the user through a web form and uses a trained ML model to recommend one of 10 career options:

**Input fields:**
- **Skills** — e.g., Python, Android, Kotlin, Content Writing
- **Interest** — e.g., Web Development, Designing, Data Analysis
- **Personality** — Logical / Creative / Analytical / Social
- **Education Level** — Diploma / UG / PG

**Career predictions it can output:**
- Software Engineer
- Web Developer
- Data Scientist
- ML Engineer
- Mobile App Developer
- Android Developer
- UI/UX Designer
- Cloud Engineer
- DevOps Engineer
- Digital Marketer

## How It Works

1. `main.py` — trains a **Random Forest Classifier** on a dataset of 4,500 student records using a scikit-learn Pipeline
2. Text fields (Skills, Interest) are processed using **TF-IDF Vectorization**
3. Categorical fields (Personality, Education Level) are encoded using **One-Hot Encoding**
4. The trained model is saved as `career_recommendation_model.pkl` using joblib
5. `app.py` — a **Flask web app** that loads the trained model and serves predictions through a browser interface

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python |
| Web Framework | Flask |
| ML Model | Random Forest Classifier (scikit-learn) |
| Text Processing | TF-IDF Vectorizer |
| Data Handling | Pandas |
| Model Saving | Joblib |
| Frontend | HTML, CSS (Jinja2 templates) |
| Dataset | 4,500 records, 10 career categories |

## Project Structure

```
career_suggestion/
├── app.py                          # Flask web application
├── main.py                         # Model training script
├── career_dataset_final.csv        # Training dataset (4500 records)
├── career_recommendation_model.pkl # Saved trained model
└── templates/
    ├── index.html                  # Input form page
    └── result.html                 # Prediction result page
```

## How to Run

### Prerequisites
Make sure Python is installed along with the required libraries:

```bash
pip install flask scikit-learn pandas joblib
```

### Step 1 — Train the model (optional, model already included)

```bash
python main.py
```

This trains the Random Forest model on the dataset and saves it as `career_recommendation_model.pkl`.

### Step 2 — Run the web app

```bash
python app.py
```

### Step 3 — Open in browser

Go to: `http://127.0.0.1:5000`

Fill in your skills, interest, personality, and education level — click **Predict Career** to see your recommended career path.

## Project Purpose

Built as a college assignment to explore how machine learning can be applied to real-world decision-making problems. The project demonstrates end-to-end ML development — from data preprocessing and model training to deploying predictions through a web interface using Flask.

---

*Developed by Husanpreet Singh,Jashan,Harshdeep | B.Tech CSE, GNDEC
