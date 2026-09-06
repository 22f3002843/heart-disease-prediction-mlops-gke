import requests
import json
import time
import subprocess
import pandas as pd
import numpy as np

# Retrieve External IP of Service
get_ip_cmd = "kubectl get svc heart-disease-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}'"
ext_ip = subprocess.check_output(get_ip_cmd, shell=True).decode('utf-8').strip().strip("'")

if not ext_ip:
    print("Warning: External IP not detected, defaulting to localhost:8080")
    api_url = "http://127.0.0.1:8080/predict"
else:
    api_url = f"http://{ext_ip}/predict"

print(f"Targeting API Endpoint: {api_url}")

# Load training data as distribution blueprint
df = pd.read_csv('data.csv')
for col in ['sno', 'target']:
    if col in df.columns:
        df = df.drop(columns=[col])

if df['gender'].dtype == object:
    df['gender'] = df['gender'].astype(str).str.strip().str.lower().map({'male': 1, 'female': 0})

np.random.seed(42)
n_samples = 100

random_data = {}
for col in df.columns:
    if df[col].dtype in [np.int64, int]:
        random_data[col] = np.random.randint(int(df[col].min()), int(df[col].max()) + 1, size=n_samples)
    else:
        random_data[col] = np.round(np.random.uniform(float(df[col].min()), float(df[col].max()), size=n_samples), 2)

random_df = pd.DataFrame(random_data)
random_df.to_csv("random_100_samples.csv", index=False)
print("Created random_100_samples.csv")

results = []
print("Sending 100 per-sample prediction requests...")
for idx, row in random_df.iterrows():
    payload = row.to_dict()
    for k in ['gender', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
        if k in payload:
            payload[k] = int(payload[k])
            
    res = requests.post(api_url, json=payload)
    if res.status_code == 200:
        pred_res = res.json()
        results.append({
            "sample_index": idx,
            "inputs": payload,
            "prediction": pred_res.get("prediction"),
            "probability": pred_res.get("probability"),
            "status": "SUCCESS"
        })
    else:
        results.append({
            "sample_index": idx,
            "status": f"FAILED_{res.status_code}"
        })
    time.sleep(0.05)

with open("per_sample_prediction_logs.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved per_sample_prediction_logs.json successfully.")
