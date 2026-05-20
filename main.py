import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder  
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
csv_path = Path(__file__).resolve().parent.parent/"career_dataset_final.csv"
if not csv_path.exists():
    print(f"ERROR: CSV file not found at {csv_path}")
    print("Please make sure 'career_dataset_final.csv' is present in the project root directory.")
    exit(1)
df = pd.read_csv(csv_path)
X=df[['Skills','Interest','Personality','Education_Level']]
Y=df['Recommended_Career']
X_train,X_test,y_train,y_test=train_test_split(X, Y, test_size=0.2, random_state=42)

preprocessor=ColumnTransformer(
    transformers=[
        ('Skills_tfidf', TfidfVectorizer(), 'Skills'),
        ('Interest_tfidf', TfidfVectorizer(), 'Interest'),
        ('Personality', OneHotEncoder(handle_unknown='ignore'), ['Personality']),
        ('Education_Level', OneHotEncoder(handle_unknown='ignore'), ['Education_Level'])
    ],
    remainder='drop',
    sparse_threshold=0
)
model=Pipeline(
    steps=[
        ('preprocessor',preprocessor),
        ("classifier",RandomForestClassifier(n_estimators=200,random_state=42))
    ]
)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
user_input={
    "Skills": input("Enter skills: "),
    "Interest": input("interest: "),
    "Personality": input("personality: "),
    "Education_Level": input("Enter education: "),
}
user_df=pd.DataFrame([user_input])
prediction=model.predict(user_df)[0]
print("career suggest to you:", prediction)
import joblib
joblib.dump(model, "career_recommendation_model.pkl")

