# 💳 Credit Card Fraud Detection  

![Python](https://img.shields.io/badge/python-3.13-blue)  
![Status](https://img.shields.io/badge/status-in%20progress-yellow)  

---

## 📌 Project Overview  
A **real-world inspired system** for detecting fraudulent credit card transactions using **Python** and **Machine Learning**.  

This project simulates a **production-ready ML pipeline**:  
- Data preprocessing & feature engineering  
- Training and comparing multiple ML models  
- Evaluating trade-offs between false positives and false negatives  
- Future deployment as a REST API & web app  

The goal is to deliver something **practical for companies** and **reproducible for research**.

---

## 📊 Dataset  
- Source: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- **284,807 transactions**, of which **492 are fraudulent (0.17%)**  
- Highly **imbalanced dataset** → handled via resampling & anomaly detection techniques  

⚠️ Dataset stored in `data/data_raw/` (ignored by Git).

---

## 🛠️ Technologies & Libraries  
- **Python 3.13.0**  
- **Jupyter Notebooks** → experimentation  
- **NumPy** → numerical computing  
- **Pandas** → data handling  
- **Matplotlib** & **Seaborn** → visualization  
- **Scikit-learn** → ML models & evaluation  
- **ipykernel** → Jupyter integration  

---

## 🔬 Methods  
1. **Data Preprocessing**  
   - Handle missing values  
   - Normalize numerical features  
   - Manage class imbalance (SMOTE / undersampling)  

2. **Exploratory Data Analysis (EDA)**  
   - Fraud vs. non-fraud distribution  
   - Correlations, PCA visualization  

3. **Model Training**  
   - Logistic Regression  
   - Decision Trees  
   - Random Forest  
   - Gradient Boosting  
   - Neural Networks (optional, advanced)  

4. **Model Evaluation**  
   - Accuracy, Precision, Recall, F1-score  
   - ROC-AUC & PR-AUC  
   - Confusion Matrix  
   - Cross-validation  

---

## 📈 Results  
- Comparative results of models (tables + plots)  
- ROC & PR curves for visualization  
- Discussion of trade-offs (false positives vs. false negatives)  

📌 *This section will be continuously updated as experiments progress.*

---

## ✅ Project Structure  
```bash
cc-fraud-detection/
├── data/
│   └── data_raw/        # raw dataset (ignored in git)
├── notebooks/           # Jupyter notebooks (EDA, modeling, results)
├── reports/
│   └── figures/         # plots, visualizations
├── src/                 # Python source code
├── logs/                # training logs (ignored in git)
├── models/              # saved models (ignored in git)
├── requirements.txt     # Python dependencies
├── .gitignore
└── README.md
```
---

## 📌 Project Management

Trello Board → [Kanban (To Do / In Progress / Done)](https://trello.com/b/c9be9zsO/my-trello-board)

## 🚀 Deployment (Planned)

- Serve best model via REST API (FastAPI / Flask)
- Simple UI (Streamlit / Flask)
- Optional cloud deployment: Heroku / AWS / Azure

## 🎓 Certifications & Skills Acquired

To support this project, I completed industry-recognized certifications, ensuring both theoretical grounding and practical skills:

| Certification / Course | Provider | Month Completed | Linked Report Section |
|-------------------------|----------|-----------------|------------------------|
| Supervised Machine Learning: Regression & Classification | Coursera (Stanford / DeepLearning.AI) | Month 1 | Intro & Background |
| Data Science with Python | Great Learning Academy | Month 1 | Intro & Background |
| Python, Pandas, Data Visualization (Micro-courses) | Kaggle Learn | Month 1–2 | Data & Methodology |
| Machine Learning with Python (IBM Digital Badge) | IBM Cognitive Class | Month 2 | Data & Methodology |
| Intermediate Machine Learning | Kaggle Learn | Month 3 | Experiments & Modeling |
| Feature Engineering | Kaggle Learn | Month 3 | Experiments & Modeling |
| ML Explainability (SHAP, LIME) | Kaggle Learn | Month 3–4 | Results & Explainability |
| Docker Essentials | IBM | Month 5 | Deployment |
| AWS Cloud Practitioner Essentials | AWS Training | Month 5 | Deployment |
| Google Cloud Skill Badges (ML on GCP, Responsible AI) | Google Cloud | Month 5 | Deployment |
| Deploy Web App with Containers | Microsoft Learn | Month 5 | Deployment |


## 👤 Author

Lazaros Voulistiotis
🎓 BSc Computer Science (Final Year) | Aspiring Machine Learning Engineer