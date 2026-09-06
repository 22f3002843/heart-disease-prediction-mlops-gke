import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data.csv')
if 'sno' in df.columns:
    df = df.drop(columns=['sno'])

if df['target'].dtype == object:
    df['target'] = df['target'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})

if 'gender' in df.columns and df['gender'].dtype == object:
    df['gender'] = df['gender'].astype(str).str.strip().str.lower().map({'male': 1, 'female': 0})

X = df.drop(columns=['target'])
y = df['target']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, 'model.joblib')
print("Saved model.joblib successfully")
