# Credit Card Fraud Detection

## 📌 Project Overview
This project focuses on building a **Credit Card Fraud Detection system** using Python and Machine Learning.  
It explores real-world financial transaction data, applies **data preprocessing**, **feature engineering**, and trains multiple ML algorithms to classify fraudulent vs. legitimate transactions.  

The goal is to simulate a production-like workflow that is **practical for companies** and **reproducible for research**.

---

## 📊 Dataset
- Source: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- Contains **284,807 transactions**, of which **492 are fraudulent (0.17%)**.  
- Highly **imbalanced dataset** → requires special handling (resampling / anomaly detection).

⚠️ Note: Raw dataset is stored in `data/data_raw/` (ignored by Git).

---

## 🛠️ Technologies & Libraries
- **Python 3.13.0**
- **Jupyter Notebooks** → interactive experimentation
- **NumPy** → numerical computing with arrays and vectors
- **Pandas** → dataset handling (CSV, Excel, transformations)
- **Matplotlib** → data visualization
- **Seaborn** → statistical plots
- **Scikit-learn** → machine learning algorithms (logistic regression, decision trees, random forests, etc.)
- **ipykernel** → Jupyter integration with virtual environments

---

## 🔬 Methods
1. **Data Preprocessing**
   - Handle missing values
   - Normalize numerical features
   - Handle class imbalance (SMOTE / undersampling)
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
   - ROC-AUC
   - Confusion Matrix
   - Cross-validation

---

## 📈 Results
- Will include comparison of models (tables + plots).
- ROC curves for performance visualization.
- Discussion of trade-offs (false positives vs. false negatives).

---

## ✅ Project Structure
cc-fraud-detection/ <br>
│── data/ <br>
│ └── data_raw/ # raw dataset (ignored in git) <br>
│── notebooks/ # Jupyter notebooks (EDA, modeling, results) <br>
│── reports/ <br>
│ └── figures/ # plots, visualizations <br>
│── src/ # Python source code <br>
│── logs/ # logs during training (ignored in git) <br>
│── models/ # saved models (ignored in git) <br>
│── requirements.txt # Python dependencies <br>
│── .gitignore <br>
│── README.md <br>


---

## 📌 Project Management
- **Trello Board**: [(https://trello.com/invite/b/68a414f9be0ed9b54d51fbb9/ATTIcda162497a0904de2dc75948b466334e365904DC/my-trello-board)]  
  Kanban style → *To Do / In Progress / Done* for tasks.

---

## 🚀 Deployment (Future Work)
- Wrap best model into a **REST API** (FastAPI or Flask).  
- Build a simple **UI** (Streamlit / Flask web app).  
- (Optional) Deploy to **Heroku / AWS / Azure** for cloud readiness.

---

## 👤 Author
**Lazaros Voulistiotis**  
- BSc Computer Science Student (Final Year)  
- Aspiring Machine Learning Engineer  
- GitHub: [https://github.com/LazarosVoulistiotis]  
- LinkedIn: [https://www.linkedin.com/in/lazaros-voulistiotis/]  
