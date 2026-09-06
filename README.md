<div align="center">

# ❤️ Heart Disease Prediction — MLOps Production Deployment on GCP

> **IITM BS — MLOps OPPE-2 | May 2026**

[![IITM BS - MLOps OPPE-2](https://img.shields.io/badge/IITM%20BS-MLOps%20OPPE--2-blue?style=for-the-badge&logo=google-cloud&logoColor=white)](https://study.iitm.ac.in)
[![May 2026](https://img.shields.io/badge/Session-May%202026-brightgreen?style=for-the-badge)](#)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)](https://kubernetes.io)
[![GKE](https://img.shields.io/badge/Google%20Cloud-GKE-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)](https://cloud.google.com/kubernetes-engine)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/features/actions)

<br/>

<p align="center">
  A production-oriented Machine Learning Operations (MLOps) deployment of a <strong>Heart Disease Prediction Model</strong> using <strong>Google Cloud Platform (GCP)</strong>, <strong>Docker</strong>, <strong>FastAPI</strong>, <strong>Kubernetes/GKE</strong>, <strong>GitHub Actions</strong>, <strong>SHAP</strong>, <strong>Fairlearn</strong>, <strong>Cloud Logging</strong>, high-concurrency stress testing with <strong>wrk</strong>, and <strong>input drift detection</strong>.
</p>

</div>

---

## 📑 Table of Contents

- [📌 Project Overview](#-project-overview)
- [🎯 OPPE-2 Deliverables](#-oppe-2-deliverables)
- [🏗️ System Architecture](#️-system-architecture)
- [📂 Repository Structure](#-repository-structure)
- [📁 File-by-File Explanation](#-file-by-file-explanation)
  - [☁️ .github/workflows/deploy.yml](#️-githubworkflowsdeployyml)
  - [☸️ k8s/deployment.yaml](#️-k8sdeploymentyaml)
  - [📈 k8s/hpa.yaml](#-k8shpayaml)
  - [🐳 Dockerfile](#-dockerfile)
  - [🚫 .dockerignore](#-dockerignore)
  - [🚫 .gitignore](#-gitignore)
  - [🌐 app.py](#-apppy)
  - [🧠 train_and_save_model.py](#-train_and_save_modelpy)
  - [📊 data.csv](#-datacsv)
  - [🔍 evaluate_responsible_ai.py](#-evaluate_responsible_aipy)
  - [🔬 SHAP Explainability](#-shap-explainability)
  - [⚖️ Fairlearn Fairness Analysis](#️-fairlearn_metrics_reporttxt)
  - [🧪 run_sample_predictions.py](#-run_sample_predictionspy)
  - [📄 random_100_samples.csv](#-random_100_samplescsv)
  - [📝 per_sample_prediction_logs.json](#-per_sample_prediction_logsjson)
  - [☁️ gcp_cloud_logging_proof.json](#️-gcp_cloud_logging_proofjson)
  - [🚀 run_stress_test.py](#-run_stress_testpy)
  - [🦾 post_payload.lua](#-post_payloadlua)
  - [📈 stress_test_report.txt](#-stress_test_reporttxt)
  - [📉 evaluate_input_drift.py](#-evaluate_input_driftpy)
  - [📋 drift_detection_report.txt](#-drift_detection_reporttxt)
- [🔄 End-to-End MLOps Workflow](#-end-to-end-mlops-workflow)
- [🔐 CI/CD Architecture](#-cicd-architecture)
- [☸️ Kubernetes Scalability](#️-kubernetes-scalability)
- [📊 Observability](#-observability)
- [🚀 Performance Testing](#-performance-testing)
- [📉 Input Drift Monitoring](#-input-drift-monitoring)
- [🛡️ Production-Oriented Design Considerations](#️-production-oriented-design-considerations)
- [🧪 Reproducibility](#-reproducibility)
- [📦 Technology Stack](#-technology-stack)
- [📑 Deliverable Evidence Map](#-deliverable-evidence-map)
- [🗂️ Complete Artifact Summary](#️-complete-artifact-summary)
- [🧭 Project Execution Flow](#-project-execution-flow)
- [🏁 Final Outcome](#-final-outcome)
- [👨‍💻 Project Metadata](#-project-metadata)

---

## 📌 Project Overview

This project converts a heart disease prediction model from a traditional machine-learning workflow into a production-oriented, API-served system.

The model predicts whether a patient is likely to have heart disease using **13 clinical input features**.

The complete solution addresses the major requirements of the OPPE-2 assignment:

- 🔍 **Model Explainability**
- ⚖️ **Fairness Evaluation**
- 🐳 **Dockerized Model Serving**
- ☸️ **Kubernetes Deployment**
- ☁️ **Google Kubernetes Engine (GKE)**
- 📈 **Horizontal Pod Autoscaling**
- 🔄 **CI/CD using GitHub Actions**
- 🔐 **GitHub Actions authentication using Workload Identity Federation**
- 📊 **Per-sample prediction logging**
- 📝 **Google Cloud Logging**
- 🚀 **High-concurrency performance testing**
- 📉 **Input data drift detection**
- 🧩 **Maintainable project structure**
- 🛡️ **Production-oriented deployment practices**

---

## 🎯 OPPE-2 Deliverables

| Deliverable | Requirement | Implementation |
| :---: | :---| :---|
| **D1** | Private Git Repository | GitHub private repository |
| **D2** | Model Explainability | SHAP |
| **D3** | Fairness Testing | Fairlearn using `age` |
| **D4** | Dockerized API on GKE | FastAPI + Docker + Kubernetes + GKE |
| **D5** | 100-row Prediction + Observability | FastAPI + structured logging + Cloud Logging |
| **D6** | Stress Testing | `wrk` with 2,100 concurrent connections |
| **D7** | Input Drift Detection | KS Test + Chi-Square + PSI |

---

## 🏗️ System Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                   SYSTEM ARCHITECTURE PIPELINE                                                   │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

 [ 1. DATA & MODELING ]
 ┌───────────────┐        ┌──────────────────────┐        ┌──────────────────────┐        ┌────────────────┐
 │ Heart Disease │        │    Model Training    │        │    Random Forest     │        │ Model Artifact │
 │    Dataset    │ ─────► │ train_and_save_model │ ─────► │    Classification    │ ─────► │  model.joblib  │
 │   data.csv    │        └──────────────────────┘        └──────────┬───────────┘        └───────┬────────┘
 └───────────────┘                                                   │                            │
                                              ┌──────────────────────┴──────────────────────┐     │
                                              ▼                                             ▼     │
                                       ┌──────────────┐                              ┌──────────┐ │
                                       │     SHAP     │                              │Fairlearn │ │
                                       │Explainability│                              │ Fairness │ │
                                       └──────────────┘                              └──────────┘ │
                                                                                                  │
 [ 2. SERVING & CLOUD DEPLOYMENT ]                                                                │
 ┌───────────────────────┐         ┌─────────────────────────┐         ┌──────────────────────────┴─┐
 │          GKE          │         │    Docker Container     │         │          FastAPI           │
 │ Kubernetes Deployment │ ◄────── │          Image          │ ◄────── │       Prediction API       │
 │   [HPA: 1 → 3 Pods]   │         └─────────────────────────┘         │          (app.py)          │
 └──────────┬────────────┘                                             └────────────────────────────┘
            │
            ├─────────────────────────────────────────┬────────────────────────────────────────┐
            ▼                                         ▼                                        ▼
 [ 3. TESTING, LOGGING & DRIFT ]                      │                                        │
 ┌──────────────────────┐                             │                                        │
 │ 100 Sample Predict.  │                             ▼                                        ▼
 │ (run_sample_predict) │                      ┌──────────────┐                         ┌──────────────┐
 └──────────┬───────────┘                      │     wrk      │                         │Cloud Logging │
            │                                  │ Stress Test  │                         │Observability │
            ▼                                  │(2100 Conns.) │                         └──────────────┘
 ┌──────────────────────┐                      └──────────────┘
 │      Random 100      │
 │     Sample Data      │ ─────► ┌───────────────────────┐
 │(random_100_samples)  │        │ Input Drift Detection │
 └──────────────────────┘        │(KS + Chi-Square + PSI)│
                                 └───────────────────────┘

 [ CI/CD AUTOMATION PIPELINE ]
 ┌──────────────────────┐         ┌──────────────────────┐         ┌──────────────────────┐         ┌──────────────────────┐
 │    GitHub Actions    │ ──────► │     Docker Build     │ ──────► │   Push to Artifact   │ ──────► │    Deploy to GKE     │
 │    CI/CD Pipeline    │         │      (Container)     │         │       Registry       │         │  (Rollout & Verify)  │
 └──────────────────────┘         └──────────────────────┘         └──────────────────────┘         └──────────────────────┘
```

---

## 📂 Repository Structure

```text
.
├── .github/
│   └── workflows/
│       └── deploy.yml
│
├── k8s/
│   ├── deployment.yaml
│   └── hpa.yaml
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
│
├── data.csv
├── model.joblib
│
├── app.py
├── train_and_save_model.py
│
├── evaluate_responsible_ai.py
├── shap_feature_importance.png
├── shap_least_impact_summary.txt
├── fairlearn_metrics_report.txt
│
├── run_sample_predictions.py
├── random_100_samples.csv
├── per_sample_prediction_logs.json
├── gcp_cloud_logging_proof.json
│
├── run_stress_test.py
├── post_payload.lua
├── stress_test_report.txt
│
├── evaluate_input_drift.py
└── drift_detection_report.txt
```

## 📁 File-by-File Explanation

### ☁️ `.github/workflows/deploy.yml`

#### Purpose
This file defines the GitHub Actions CI/CD pipeline.

Whenever changes are pushed to the `main` branch, the workflow:
- Checks out the repository.
- Authenticates to Google Cloud.
- Uses Workload Identity Federation (WIF).
- Configures Docker authentication.
- Builds the Docker image.
- Pushes the image to Google Artifact Registry.
- Obtains GKE cluster credentials.
- Deploys the updated application to Kubernetes.
- Applies the HPA configuration.
- Waits for the Kubernetes rollout to complete.

#### OPPE Relevance
**Deliverable 4 — Dockerized API Deployment + CI/CD**

The workflow uses:
```text
[GitHub] ──► [GitHub Actions] ──► [Google Cloud Authentication (WIF)] ──► [Artifact Registry] ──► [GKE] ──► [Kubernetes Deployment]
```

---

### ☸️ `k8s/deployment.yaml`

#### Purpose
Defines the Kubernetes Deployment for the heart disease API.

It specifies:
- Application name
- Container image
- Container port
- CPU requests
- Memory requests
- CPU limits
- Memory limits
- Readiness probe
- Liveness probe
- Initial replica count

The API container listens on:
```text
8080
```

The Kubernetes deployment uses health checks against:
```text
/
```

This allows Kubernetes to determine whether the API is ready and healthy.

#### OPPE Relevance
**Deliverable 4 — Kubernetes Deployment**

---

### 📈 `k8s/hpa.yaml`

#### Purpose
Defines the Horizontal Pod Autoscaler (HPA).

The configured scaling policy is:
- **Minimum Pods:** `1`
- **Maximum Pods:** `3`
- **CPU Target:** `50%`

Therefore, Kubernetes can automatically scale the API based on CPU utilization while respecting the assignment's maximum of 3 pods.

```text
[CPU Load (Target: 50%)] ──► [Kubernetes HPA Controller] ──► [Autoscale: Pod 1 ── Pod 2 ── Pod 3 (Max: 3 Pods)]
```

#### OPPE Relevance
**Deliverable 4 — Scalability / Autoscaling**

---

### 🐳 `Dockerfile`

#### Purpose
Defines the container image used to package the prediction API.

The Docker image:
- Uses Python 3.10
- Creates `/app` as the working directory
- Installs Python dependencies
- Copies the API and model artifact
- Exposes port `8080`
- Starts the FastAPI application using Uvicorn

Application startup:
```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

#### OPPE Relevance
**Deliverable 4 — Dockerized API**

---

### 🚫 `.dockerignore`

#### Purpose
Prevents unnecessary files from being copied into the Docker build context.

Examples include:
- `__pycache__`
- `*.pyc`
- `.git`

This keeps the Docker build smaller and cleaner.

---

### 🚫 `.gitignore`

#### Purpose
Defines files that should not be committed to Git.

It is used to prevent temporary files, generated files, and other non-required artifacts from being tracked.

---

### 🌐 `app.py`

#### Purpose
This is the main FastAPI prediction server.

It loads the trained machine-learning model and exposes the prediction API.

#### API Endpoints

##### Health Check
- **Endpoint:** `GET /`

Example response:
```json
{
  "status": "healthy",
  "service": "heart-disease-gke"
}
```

##### Prediction
- **Endpoint:** `POST /predict`

The endpoint accepts the patient's clinical features and returns:
- Prediction
- Prediction label
- Probability
- Status

Example:
```json
{
  "prediction": 1,
  "prediction_label": "yes",
  "probability": 0.87,
  "status": "success"
}
```

#### Structured Logging
Each prediction request generates a structured log containing:
- Timestamp
- Input features
- Prediction
- Prediction label
- Probability

This allows the API to integrate with Google Cloud Logging.

#### OPPE Relevance
- **Deliverable 4** — API Deployment
- **Deliverable 5** — Prediction Logging & Observability

---

### 🧠 `train_and_save_model.py`

#### Purpose
Trains the Random Forest classification model using:
```text
data.csv
```

The script:
- Loads the dataset.
- Removes the `sno` identifier.
- Encodes the target.
- Encodes gender.
- Separates features and target.
- Trains a Random Forest classifier.
- Saves the trained model.

The resulting model artifact is:
```text
model.joblib
```

#### OPPE Relevance
Supports the model-serving component of:
- **Deliverable 4**

---

### 📊 `data.csv`

#### Purpose
This is the primary heart disease dataset used as the training/reference dataset.

The dataset contains the clinical attributes used by the prediction model.

The features include:
- `age`
- `gender`
- `cp`
- `trestbps`
- `chol`
- `fbs`
- `restecg`
- `thalach`
- `exang`
- `oldpeak`
- `slope`
- `ca`
- `thal`

The prediction target is:
```text
target
```

The `sno` column is treated as an identifier and excluded from model features.

---

### 🔍 `evaluate_responsible_ai.py`

#### Purpose
This script performs the Responsible AI evaluation for:
- SHAP explainability
- Fairlearn fairness analysis

The model is evaluated using a train/test split.

---

### 🔬 SHAP Explainability

#### `shap_feature_importance.png`
##### Purpose
Visualizes the feature importance obtained from SHAP.

The ranking is based on the average magnitude of SHAP values.

A larger mean absolute SHAP value indicates a larger contribution to the model's output.

#### `shap_least_impact_summary.txt`
##### Purpose
Contains the generated explanation identifying the features with the least impact on heart disease prediction.

The report ranks features from:
```text
Lowest Impact → Highest Impact
```

It also provides a plain-English interpretation of the least impactful features.

##### OPPE Relevance
**Deliverable 2 — Model Explainability**

---

### ⚖️ `fairlearn_metrics_report.txt`

#### Purpose
Contains the Fairlearn fairness evaluation results.

The sensitive attribute is:
```text
age
```

Age is evaluated using two groups:
- Senior (`>=60`)
- Younger (`<60`)

The report contains group-level metrics such as:
- Accuracy
- Selection Rate
- True Positive Rate
- False Positive Rate
- Difference between groups
- Ratio between groups

This allows the model's performance to be compared across age groups.

#### OPPE Relevance
**Deliverable 3 — Fairness Testing**

#### ⚖️ Fairness Evaluation Concept

The fairness pipeline is:
```text
                    ┌──► [Age < 60]  ──┐
[Test Dataset] ──► [Model Predictions] ──► [Fairlearn] ─┤                  ├──► [Compare Metrics] ──► [Fairness Report]
                                                        └──► [Age >= 60] ──┘
```

---

### 🧪 `run_sample_predictions.py`

#### Purpose
Generates a 100-row random inference dataset and sends the records individually to the deployed API.

The script:
- Loads the training feature ranges.
- Generates 100 random samples.
- Saves them to: `random_100_samples.csv`
- Sends each sample individually to: `/predict`
- Collects the prediction results.
- Saves the prediction evidence.

#### OPPE Relevance
**Deliverable 5 — Per-Sample Prediction**

---

### 📄 `random_100_samples.csv`

#### Purpose
Contains the generated 100-row inference dataset.

This dataset is used for:
- Per-sample prediction
- Prediction logging
- Stress testing
- Input drift detection

Therefore, it acts as a common test dataset across multiple deliverables.

---

### 📝 `per_sample_prediction_logs.json`

#### Purpose
Stores the prediction results for the 100 generated samples.

Each record contains information such as:
```json
{
  "sample_index": 0,
  "inputs": {
    "age": 54,
    "gender": 1,
    "cp": 2
  },
  "prediction": 1,
  "probability": 0.82,
  "status": "SUCCESS"
}
```

This provides local evidence of the per-sample prediction process.

#### OPPE Relevance
**Deliverable 5**

---

### ☁️ `gcp_cloud_logging_proof.json`

#### Purpose
Contains exported Google Cloud Logging evidence for prediction requests.

The API emits structured prediction logs containing:
- `timestamp`
- `input_features`
- `prediction`
- `prediction_label`
- `probability`

The exported JSON provides evidence that prediction events were recorded in the GCP logging environment.

#### OPPE Relevance
**Deliverable 5 — Observability**

---

### 🚀 `run_stress_test.py`

#### Purpose
Automates the high-concurrency performance test.

The script:
- Obtains the GKE service external IP.
- Reads the 100-row prediction dataset.
- Generates the wrk payload script.
- Executes the benchmark.
- Captures the raw benchmark output.
- Generates the stress-test report.

The workload is configured for:
- **Threads:** `12`
- **Connections:** `2100`
- **Duration:** `30 seconds`

This satisfies the requirement for more than 2,000 concurrent connections.

#### OPPE Relevance
**Deliverable 6 — Performance Monitoring & Stress Testing**

---

### 🦾 `post_payload.lua`

#### Purpose
Lua scripting configuration for wrk.

wrk normally performs GET requests. Since the model API requires:
```text
POST /predict
```

this Lua script:
- Sets the HTTP method to `POST`.
- Sets `Content-Type: application/json`.
- Loads prediction payloads.
- Randomly selects one of the available payloads.
- Sends it to `/predict`.

This enables realistic load testing of the prediction endpoint.

---

### 📈 `stress_test_report.txt`

#### Purpose
Stores the results and analysis of the high-concurrency wrk benchmark.

The report covers:
- Throughput
- Number of requests
- Latency
- Latency percentiles
- Socket errors
- Read errors
- Timeouts
- High-concurrency behavior

The benchmark uses:
```text
2100 concurrent connections
```
which is greater than the required 2,000 connections.

#### OPPE Relevance
**Deliverable 6**

---

### 📉 `evaluate_input_drift.py`

#### Purpose
Detects whether incoming prediction data has a different distribution from the reference training data.

The comparison is:
```text
data.csv  VS  random_100_samples.csv
```

The script uses different statistical methods depending on the feature type:

#### Continuous Features
Uses:
- **Kolmogorov-Smirnov (KS) Test**

#### Categorical Features
Uses:
- **Chi-Square Test**

#### Additional Metric
The script also calculates:
- **Population Stability Index (PSI)**

The drift decision uses a statistical significance threshold of:
```text
p-value < 0.05
```

#### OPPE Relevance
**Deliverable 7 — Input Drift Detection**

---

### 📋 `drift_detection_report.txt`

#### Purpose
Contains the generated feature-by-feature drift analysis.

The report includes:
- Feature name
- Statistical test
- Test statistic
- P-value
- PSI
- Drift detected/not detected
- Overall summary

The reference dataset is:
```text
data.csv
```

The incoming/generated inference dataset is:
```text
random_100_samples.csv
```

---

## 🔄 End-to-End MLOps Workflow

The project can be viewed as the following lifecycle:

```text
┌──────────────┐     ┌────────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│   Dataset    │────►│ Model Training │────►│    Model     │────►│   FastAPI    │────►│    Docker    │────►│Artifact Registry│
│   data.csv   │     │ (Random Forest)│     │   Artifact   │     │     API      │     │  Container   │     └────────┬─────────┘
└──────────────┘     └───────┬────────┘     └──────────────┘     └──────────────┘     └──────────────┘              │
                             │                                                                                      ▼
              ┌──────────────┴──────────────┐                                                              ┌────────────────┐
              ▼                             ▼                                                              │  GKE Cluster   │
       ┌──────────────┐              ┌──────────────┐                                                      │[HPA: 1→3 Pods] │
       │     SHAP     │              │  Fairlearn   │                                                      └───────┬────────┘
       │Explainability│              │   Fairness   │                                                              │
       └──────────────┘              └──────────────┘                                                              │
                                                                 ┌─────────────────────────────────────────────────┴─────────────────┐
                                                                 ▼                                 ▼                                 ▼
                                                          ┌──────────────┐                  ┌──────────────┐                  ┌──────────────┐
                                                          │ Predictions  │                  │ wrk Testing  │                  │Cloud Logging │
                                                          │ (100 Rows)   │                  │(Stress Test) │                  │ Prediction   │
                                                          └──────┬───────┘                  └──────────────┘                  │    Logs      │
                                                                 │                                                            └──────────────┘
                                                                 ▼
                                                          ┌──────────────┐
                                                          │ Input Drift  │
                                                          │  Detection   │
                                                          └──────────────┘
```

---

## 🔐 CI/CD Architecture

The deployment pipeline uses GitHub Actions with Workload Identity Federation (WIF) rather than storing a long-lived Google Cloud service-account JSON key.

```text
[Developer] ──(git push)──► [GitHub Repository] ──► [GitHub Actions] ──(OIDC / WIF)──► [Google Service Account]
                                                                                                │
                                ┌───────────────────────────────────────────────────────────────┴───────────────────────────────┐
                                ▼                                                                                               ▼
                       [Artifact Registry] ──► [GKE / Kubernetes Deployment] ────────────────────────────────────► [Running API Service]
```

### CI/CD Stages

```text
[1. Checkout] ──► [2. Authenticate (WIF)] ──► [3. Configure GCP] ──► [4. Configure Registry] ──► [5. Docker Build]
                                                                                                       │
[10. Verify Rollout] ◄── [9. Apply HPA] ◄── [8. Deploy K8s Manifest] ◄── [7. Get GKE Credentials] ◄── [6. Docker Push]
```

---

## ☸️ Kubernetes Scalability

The application uses Kubernetes Horizontal Pod Autoscaling.

### Configuration

| Parameter | Value |
| :---| :---|
| **Minimum replicas** | `1` |
| **Maximum replicas** | `3` |
| **CPU target** | `50%` |
| **Container port** | `8080` |

Therefore, the application can scale horizontally while respecting the assignment requirement:
```text
Maximum 3 pods
```

---

## 📊 Observability

The API generates structured logs for every prediction request.

Each prediction event contains:
- Timestamp
- Input Features
- Prediction
- Prediction Label
- Probability

Example conceptual log:
```json
{
  "severity": "INFO",
  "message": "Prediction executed",
  "timestamp": "2026-09-06T00:00:00Z",
  "input_features": {},
  "prediction": 1,
  "prediction_label": "yes",
  "probability": 0.87
}
```

These logs can be inspected using Google Cloud Logging.

---

## 🚀 Performance Testing

The deployed API is tested under high concurrency using:
```text
wrk
```

### Workload

| Parameter | Configuration |
| :---| :---|
| **Threads** | `12` |
| **Connections** | `2100` |
| **Duration** | `30 seconds` |
| **Endpoint** | `POST /predict` |

### Metrics Evaluated
- Requests/sec
- Total requests
- Average latency
- Latency distribution
- 50th percentile
- 75th percentile
- 90th percentile
- 99th percentile
- Socket errors
- Read errors
- Timeouts

The purpose is to evaluate how the deployed prediction service behaves under heavy concurrent traffic.

---

## 📉 Input Drift Monitoring

Input drift compares:
- **Reference:** `data.csv`
- **Incoming:** `random_100_samples.csv`

### Statistical Tests

| Feature Type | Test |
| :---| :---|
| **Continuous** | Kolmogorov-Smirnov |
| **Categorical** | Chi-Square |
| **Stability** | Population Stability Index |

### Drift Criterion
```text
p-value < 0.05
```

A detected distribution shift can indicate that incoming data no longer resembles the data used to train the model.

This can trigger further actions such as:
```text
[Data validation] ──► [Drift investigation] ──► [Model monitoring] ──► [Potential retraining]
```

---

## 🛡️ Production-Oriented Design Considerations

The deployment incorporates several practices intended to make the system more production-oriented:

- **Containerization**  
  The model-serving application is packaged inside a Docker container.

- **Health Checks**  
  Kubernetes readiness and liveness probes are configured for the API.

- **Autoscaling**  
  The API can scale between `1 → 3 pods` based on CPU utilization.

- **Structured Logging**  
  Prediction events are emitted in a structured format suitable for centralized logging.

- **CI/CD**  
  Application changes can be automatically built and deployed through GitHub Actions.

- **Explainability**  
  SHAP is used to understand feature contributions to model predictions.

- **Fairness**  
  Fairlearn is used to evaluate model behavior across age groups.

- **Drift Detection**  
  Statistical tests are used to identify changes in incoming feature distributions.

- **Performance Testing**  
  The deployed endpoint is tested under high-concurrency traffic.

---

## 🧪 Reproducibility

The major components can be reproduced using the scripts in this repository:

```bash
# 1. Train model
python3 train_and_save_model.py

# 2. Responsible AI evaluation
python3 evaluate_responsible_ai.py

# 3. Generate 100 predictions
python3 run_sample_predictions.py

# 4. Stress testing
python3 run_stress_test.py

# 5. Drift detection
python3 evaluate_input_drift.py
```

---

## 📦 Technology Stack

| Category | Technology |
| :---| :---|
| **Programming Language** | Python |
| **ML Framework** | scikit-learn |
| **Model** | Random Forest Classifier |
| **Explainability** | SHAP |
| **Fairness** | Fairlearn |
| **API** | FastAPI |
| **API Server** | Uvicorn |
| **Containerization** | Docker |
| **Container Registry** | Google Artifact Registry |
| **Orchestration** | Kubernetes |
| **Cloud Kubernetes** | Google Kubernetes Engine (GKE) |
| **Autoscaling** | Kubernetes HPA |
| **CI/CD** | GitHub Actions |
| **Authentication** | Workload Identity Federation |
| **Logging** | Google Cloud Logging |
| **Stress Testing** | wrk |
| **Drift Detection** | SciPy statistical tests + PSI |

---

## 📑 Deliverable Evidence Map

### Deliverable 1 — Private Repository
- **Repository:** `<IITM_BS_ROLL_NUMBER>_IITMBS_MLOPS_OPPE2_MAY_2026`
- The repository contains the complete implementation and supporting artifacts.

---

### Deliverable 2 — Model Explainability
- **Implementation:** `evaluate_responsible_ai.py`
- **Evidence:**
  - `shap_feature_importance.png`
  - `shap_least_impact_summary.txt`
- **Method:** SHAP
- The report identifies the features having the lowest average contribution to model predictions.

---

### Deliverable 3 — Fairness Testing
- **Implementation:** `evaluate_responsible_ai.py`
- **Evidence:** `fairlearn_metrics_report.txt`
- **Method:** Fairlearn
- **Sensitive Attribute:** `age`
- Age is evaluated by comparing the defined age groups.

---

### Deliverable 4 — Dockerized API Deployment on GKE
- **Main files:**
  - `app.py`
  - `train_and_save_model.py`
  - `Dockerfile`
  - `requirements.txt`
  - `k8s/deployment.yaml`
  - `k8s/hpa.yaml`
  - `.github/workflows/deploy.yml`
- **Infrastructure:**
  - Docker
  - Artifact Registry
  - GKE
  - Kubernetes
  - HPA
  - GitHub Actions
  - WIF
- **Autoscaling:** Maximum pods = `3`

---

### Deliverable 5 — Prediction Logging & Observability
- **Main files:**
  - `run_sample_predictions.py`
  - `random_100_samples.csv`
  - `per_sample_prediction_logs.json`
  - `gcp_cloud_logging_proof.json`
- **Process:**
```text
[Generate 100 rows] ──► [Send each row individually] ──► [POST /predict] ──► [Generate prediction] ──► [Structured log] ──► [Cloud Logging]
```

---

### Deliverable 6 — Performance Monitoring
- **Main files:**
  - `run_stress_test.py`
  - `post_payload.lua`
  - `stress_test_report.txt`
- **Test:**
  - `2100` concurrent connections
  - `12` threads
  - `30` seconds
- **Metrics:**
  - Throughput
  - Latency
  - Percentiles
  - Errors
  - Timeouts

---

### Deliverable 7 — Input Drift Detection
- **Main files:**
  - `evaluate_input_drift.py`
  - `drift_detection_report.txt`
- **Comparison:**
```text
Training Dataset             Inference Dataset
   data.csv         VS     random_100_samples.csv
```
- **Methods:**
  - Kolmogorov-Smirnov Test
  - Chi-Square Test
  - Population Stability Index

---

## 🗂️ Complete Artifact Summary

| File | Role |
| :---| :---|
| `data.csv` | Original heart disease dataset |
| `train_and_save_model.py` | Trains and saves the Random Forest model |
| `app.py` | FastAPI prediction service |
| `Dockerfile` | Builds the API container |
| `requirements.txt` | Python dependencies |
| `.dockerignore` | Docker build exclusions |
| `.gitignore` | Git tracking exclusions |
| `k8s/deployment.yaml` | Kubernetes API deployment |
| `k8s/hpa.yaml` | Kubernetes autoscaling configuration |
| `.github/workflows/deploy.yml` | GitHub Actions CI/CD |
| `evaluate_responsible_ai.py` | SHAP + Fairlearn evaluation |
| `shap_feature_importance.png` | SHAP feature importance visualization |
| `shap_least_impact_summary.txt` | Least-impact feature explanation |
| `fairlearn_metrics_report.txt` | Fairness evaluation evidence |
| `run_sample_predictions.py` | Generates and sends 100 predictions |
| `random_100_samples.csv` | 100-row inference dataset |
| `per_sample_prediction_logs.json` | Prediction results |
| `gcp_cloud_logging_proof.json` | Cloud Logging evidence |
| `run_stress_test.py` | Automates wrk stress testing |
| `post_payload.lua` | POST payload generator for wrk |
| `stress_test_report.txt` | Stress-test results and analysis |
| `evaluate_input_drift.py` | Statistical input drift analysis |
| `drift_detection_report.txt` | Drift analysis evidence |
| `README.md` | Project documentation |

---

## 🧭 Project Execution Flow

The complete implementation follows this sequence:

```text
[PHASE 1: MODEL & RESPONSIBLE AI]
[1. Prepare Dataset] ──► [2. Train ML Model] ──► [3. SHAP Explainability] ──► [4. Fairlearn Fairness]
                                                                                       │
[PHASE 2: API SERVING & GKE DEPLOYMENT]                                                │
[9. Configure HPA] ◄── [8. Deploy to GKE] ◄── [7. Push to Registry] ◄── [6. Containerize (Docker)] ◄── [5. Build FastAPI]
       │
       ▼
[PHASE 3: CI/CD & SECURITY]
[10. Configure GitHub Actions CI/CD] ──► [11. Authenticate GitHub → GCP (WIF)]
                                                            │
[PHASE 4: INFERENCE, LOGGING, TESTING & DRIFT]              │
[12. Generate 100 Samples] ◄────────────────────────────────┘
       │
       └──► [13. Per-Sample Predictions] ──► [14. Cloud Logging Evidence] ──► [15. 2100-Conn wrk Test]
                                                                                       │
[18. Detect Input Drift] ◄── [17. Compare Training vs Inference] ◄── [16. Analyze Performance] ◄──┘
```

---

## 🏁 Final Outcome

This project demonstrates an end-to-end MLOps workflow for deploying a machine-learning model into a cloud-based production-oriented environment.

The final system combines:

```text
Machine Learning + Responsible AI + API Serving + Containerization + Kubernetes + Cloud Infrastructure + Autoscaling + CI/CD + Observability + Performance Testing + Drift Detection
```

The result is a scalable and observable heart disease prediction service deployed on Google Kubernetes Engine, with supporting Responsible AI, performance, logging, and data-drift evaluation artifacts.

---

## 👨‍💻 Project Metadata

- **Program:** IITM BS — MLOps OPPE-2
- **Domain:** Healthcare / Machine Learning Operations
- **Deployment Platform:** Google Cloud Platform (GCP)
- **Container Orchestration:** Google Kubernetes Engine (GKE)
- **CI/CD:** GitHub Actions + Workload Identity Federation (WIF)
- **Model:** Random Forest Classifier
