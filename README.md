# 💳 Credit Card Fraud Detection  
**Business-Oriented Machine Learning Project**

![Python](https://img.shields.io/badge/python-3.13-blue)  
![Status](https://img.shields.io/badge/status-in%20progress-yellow)

---

## 🧭 Executive Summary

This repository contains a **production-inspired credit card fraud detection system** built as part of a **final-year BSc Computer Science thesis**.

The project is designed with a **business lens**:

- Reduce **financial losses from fraudulent transactions**
- Protect **customer trust and brand reputation**
- Provide **actionable, explainable insights** to fraud analysts and risk teams
- Demonstrate a **reproducible ML pipeline** suitable for real-world deployment (API + dashboard)

The focus is not only on building models, but on **understanding the business trade-offs** between:
- Catching as many fraudulent transactions as possible (**recall / sensitivity**)
- Keeping false alarms at a manageable level (**precision**, customer experience, operational cost)

---

## 🏦 Business Problem & Objectives

Credit card fraud, even at low percentage levels, can generate **disproportionately high financial impact** due to chargebacks, investigation costs and customer churn.

### Core Business Question

> *Can we detect fraudulent transactions early and accurately enough to support a bank’s fraud detection team, while keeping false positives at an operationally acceptable level?*

### Business Objectives

1. **Early Detection**  
   Build models that can flag suspicious transactions in (near) real-time.

2. **High Recall on Fraud Cases (Class = 1)**  
   Minimize **false negatives** – fraudulent transactions that pass undetected.

3. **Controlled False Positives**  
   Avoid overwhelming analysts and irritating customers with unnecessary alerts.

4. **Explainability & Trust**  
   Provide interpretable outputs (feature importance, example rules, SHAP-style explanations) so that **risk teams can trust and audit** the model.

5. **Reproducibility & Extensibility**  
   Organize the project so it can be:
   - Extended by another data scientist
   - Integrated into a **REST API** and a **web interface**
   - Used as a blueprint in a corporate environment

---

## 📊 Dataset

- **Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- **Transactions:** 284,807  
- **Fraud cases:** 492  
- **Fraud rate:** ~0.17% (highly imbalanced)

The dataset is **anonymized** and includes:
- PCA-transformed features: `V1`–`V28`
- `Amount` and `Time`
- `Class` (0 = legitimate, 1 = fraud)

> 🔐 The raw dataset is stored locally in `data/data_raw/` and is **excluded from Git** for good practice around data handling.

---

## 🛠️ Technology Stack

**Core Language & Environment**
- Python 3.13.0
- Virtual environment (`.venv`) and `requirements.txt` for reproducibility

**Data & Analysis**
- `NumPy`, `Pandas` – data manipulation and numerical computing  
- `Matplotlib`, `Seaborn` – visualizations & dashboards (EDA, model performance)  
- `Jupyter Notebooks` – experimentation and exploratory analysis  

**Machine Learning**
- `scikit-learn` – models, metrics, cross-validation  
  - Logistic Regression (baseline)
  - Decision Trees
  - Random Forest
  - Gradient Boosting (e.g. XGBoost / similar, where applicable)
- Planned: model explainability with **SHAP / LIME-style** techniques

**Integration & Deployment (Planned)**
- REST API with **FastAPI** or **Flask**
- Simple **web UI** (e.g. Streamlit / Flask templates)
- Containerization with **Docker**
- Optional cloud deployment (e.g. AWS / Azure / Heroku)

---

## 🔬 Methodology (From Data to Business Value)

### 1. Data Preprocessing & Quality

- Check for **missing or invalid values**
- Scale `Amount` and potentially `Time`
- Train–test split with **class imbalance in mind**
- Optionally create **time-based splits** to mimic real-world streaming data

### 2. Handling Class Imbalance

Because fraud is rare, classical accuracy is misleading. Strategies include:

- Baseline: **no resampling** (to understand the raw difficulty)
- **SMOTE** (Synthetic Minority Oversampling Technique)
- **Random undersampling** of the majority class
- Optionally compare with **class-weighted models**

> Business angle: different resampling strategies simulate different **operational policies** (e.g. “more aggressive” vs “more conservative” fraud detection).

### 3. Model Training

Initial and planned models:

- **Logistic Regression**  
  - Baseline model, interpretable coefficients  
  - Good benchmark for more complex models

- **Decision Tree Classifier**  
  - Non-linear, rule-based model  
  - Easy to interpret and explain to non-technical stakeholders

- **Random Forest / Gradient Boosting**  
  - Ensemble techniques for higher performance  
  - Often used in production fraud systems  
  - Trade-off: harder to interpret vs. better metrics

- (Optional, advanced) **Neural Networks** for comparison on structured/tabular data.

### 4. Model Evaluation & Business Metrics

Standard ML metrics:
- Accuracy, Precision, Recall, F1-score
- ROC-AUC, PR-AUC
- Confusion Matrix

Business-oriented focus:
- **Recall on Class = 1** (How many fraudulent cases do we catch?)
- **Precision on Class = 1** (How many of our alerts are truly fraud?)
- **False Positive Rate** (How many legitimate customers are disturbed?)
- Threshold analysis: find operating points that match **business risk appetite**.

> The final report will discuss **scenarios**, such as:
> - “High-security mode” → higher recall, lower precision  
> - “Customer-friendly mode” → fewer false positives, slightly lower recall  

---

## 📈 Results & Insights (Work in Progress)

This section is updated as experiments progress.

Planned deliverables:
- Comparative performance table: **Logistic Regression vs Decision Tree vs Ensembles**
- ROC / PR curves saved under `reports/figures/`
- Example **decision paths / rules** from the tree model
- Business interpretation:
  - Which features contribute most to fraud detection?
  - How would changing the decision threshold affect:
    - Number of flagged transactions per day
    - Estimated avoided fraud
    - Number of extra customer verifications

---

## 📂 Project Structure (high-level)

```bash
cc-fraud-detection/
├── certificates/      # Certifications & badges related to the project
├── data/              # Datasets (raw & processed) - raw data not tracked in git
├── models/            # Saved trained models (*.joblib)
├── notebooks/         # Jupyter notebooks for research & experimentation
├── reports/           # Thesis-oriented reports & documentation
│   └── figures/       # Plots & diagrams (organized per week: week2/, …, week10/)
├── src/               # Production-like Python scripts (ML pipeline)
├── .gitignore         # Git ignore rules (data_raw, .venv, models, etc.)
├── README.md          # Project overview (you are here)
└── requirements.txt   # Python dependencies

```
---

## 📁 Project Folders & Roles

- **`notebooks/` → Research & experimentation**  
  Jupyter Notebooks που χρησιμοποιούνται για:
  - Πειραματισμό με διαφορετικά μοντέλα και hyperparameters  
  - EDA (Exploratory Data Analysis) και οπτικοποιήσεις  
  - Γρήγορο δοκιμαστικό κώδικα πριν μπει σε παραγωγική μορφή στο `src/`  

- **`src/` → Production-like code (reusable functions, scripts)**  
  Περιέχει πιο “καθαρό” και επαναχρησιμοποιήσιμο κώδικα:
  - Σενάρια φόρτωσης δεδομένων, preprocessing και training  
  - Ορισμοί pipelines και βοηθητικών συναρτήσεων  
  - Κώδικα που μπορεί να χρησιμοποιηθεί αργότερα σε API ή σε deployment  

- **`reports/` → Material that connects directly to the thesis chapters**  
  Ο φάκελος που “κουμπώνει” με την πτυχιακή:
  - Markdown αρχεία ανά εβδομάδα / ενότητα (EDA, Modeling, Evaluation κ.λπ.)  
  - Περιγραφές πειραμάτων, συμπεράσματα, business ερμηνείες  
  - Αναφορές σε figures και πίνακες που αποθηκεύονται στο `reports/figures/`  

---

## 📌 Project Management & Workflow

The project is managed using a **Kanban-style workflow** to reflect a real team environment.

- **Trello Board (Kanban):**  
  Columns: `To Do` / `In Progress` / `Done`

Each card corresponds to concrete tasks such as:

- **EDA & Data Understanding**  
  - Explore distributions, correlations, data quality  
  - Identify issues relevant to fraud detection

- **Model baselines & tuning**  
  - Train and compare baseline models (e.g. Logistic Regression, Decision Trees)  
  - Perform basic hyperparameter tuning

- **Metrics & business interpretation**  
  - Evaluate models with fraud-relevant metrics (recall, precision, ROC/PR curves)  
  - Translate results into **business impact** (false positives, missed fraud, workload on analysts)

- **Documentation & thesis sections**  
  - Update markdown reports in `reports/`  
  - Align code experiments with thesis chapters

- **Deployment & explainability**  
  - Prepare models for serving (API)  
  - Add interpretability artifacts (feature importance, explanations)

This structure makes the project **transparent**, **trackable**, and closer to **industry practices**.

---

## 🚀 Deployment Roadmap (Planned)

### 1. Model Packaging

- Persist the **best-performing model** (e.g., `joblib` or `pickle`)  
- Store **preprocessing steps** (scalers, encoders) together with the model  
- Ensure reproducibility of the full inference pipeline

### 2. REST API

Expose the model through a simple REST API, e.g. with **FastAPI** or **Flask**:

- Endpoint: `POST /predict`  
  - Input: transaction features (JSON payload)  
  - Output:
    - Fraud probability (e.g., score between 0 and 1)  
    - Recommended action: **flag / review / approve**

This mirrors how banks integrate ML models into existing systems (core banking, fraud-monitoring tools, dashboards).

### 3. Web Interface / Dashboard

Build a lightweight **web UI** for business users and analysts:

- Manual transaction scoring (paste or upload data)  
- View:
  - Distributions and trends in alerts  
  - Model performance snapshots (precision–recall, confusion matrix)  
- Implementation options:
  - **Streamlit** for fast prototyping  
  - Or **Flask + HTML templates** for more control

### 4. Containers & Cloud

- Create a **Dockerfile** to containerize:
  - Model  
  - API  
  - Any preprocessing logic  
- Optional cloud deployment to platforms such as:
  - **AWS** (e.g. ECS, Fargate, EC2)  
  - **Azure** (App Service, Container Apps)  
  - **Heroku** or similar for demo purposes  

This shows how the solution can be **scaled and integrated** in a real-world environment.

---

## 🎓 Certifications & Skills Acquired

To align the project with **industry expectations**, a series of certifications and courses are being completed in parallel:

| Certification / Course                                      | Provider                                  | Month Completed | Linked Report Section        |
|-------------------------------------------------------------|-------------------------------------------|-----------------|------------------------------|
| Supervised Machine Learning: Regression & Classification    | Coursera (Stanford / DeepLearning.AI)     | Month 1         | Intro & Background           |
| Data Science with Python                                   | Great Learning Academy                    | Month 1         | Intro & Background           |
| Python, Pandas, Data Visualization (Micro-courses)         | Kaggle Learn                              | Month 1–2       | Data & Methodology           |
| Machine Learning with Python (IBM Digital Badge)           | IBM Cognitive Class                       | Month 2         | Data & Methodology           |
| Intermediate Machine Learning                              | Kaggle Learn                              | Month 3         | Experiments & Modeling       |
| Feature Engineering                                        | Kaggle Learn                              | Month 3         | Experiments & Modeling       |
| ML Explainability (SHAP, LIME)                             | Kaggle Learn                              | Month 3–4       | Results & Explainability     |
| Docker Essentials                                          | IBM                                       | Month 5         | Deployment                   |
| AWS Cloud Practitioner Essentials                          | AWS Training                              | Month 5         | Deployment                   |
| Google Cloud Skill Badges (ML on GCP, Responsible AI)      | Google Cloud                              | Month 5         | Deployment                   |
| Deploy Web App with Containers                             | Microsoft Learn                           | Month 5         | Deployment                   |

These certifications are referenced in the written thesis as **evidence of structured learning and professional development**, and show a clear path from **core ML skills** to **MLOps and cloud deployment**.

---

## 👤 Author

**Lazaros Voulistiotis**  
🎓 BSc Computer Science (Final Year)  
🚀 Aspiring Machine Learning Engineer
