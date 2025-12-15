# Credit Card Fraud Detection using Machine Learning 💳🚨

An end-to-end Machine Learning project that detects fraudulent credit card transactions using an imbalanced dataset. The system uses XGBoost with SMOTE and is deployed as an interactive Streamlit web application.

## 📌 Overview

Credit card fraud detection is a real-world classification problem where fraudulent transactions are extremely rare. This project focuses on handling class imbalance, selecting the right evaluation metrics (like Precision-Recall AUC), and deploying a usable ML system.

## 🚀 Features

- Handles highly imbalanced data
- Uses SMOTE (Synthetic Minority Over-sampling Technique) for oversampling
- Model comparison:
  - Logistic Regression
  - Random Forest
  - XGBoost (final model)
- Evaluation using:
  - ROC-AUC
  - Precision-Recall AUC
- Interactive Streamlit dashboard
- Supports:
  - Batch CSV prediction
  - Single transaction prediction
  - Adjustable fraud detection threshold

## 🧠 Machine Learning Pipeline

- Data preprocessing and feature scaling
- Stratified train-test split
- Imbalanced learning using SMOTE
- Model training and comparison
- Final model selection using PR-AUC (Precision-Recall AUC)
- Deployment using Streamlit

## 📊 Model Performance

| Metric | Score | Note |
|--------|-------|------|
| ROC-AUC | ~0.98 | Excellent discrimination |
| PR-AUC | ~0.86 | Better measure for imbalanced data |
| Recall | High | Minimizes missed fraud cases (False Negatives) |

## 🛠️ Tech Stack

- Python
- Scikit-learn
- XGBoost
- Imbalanced-learn
- Pandas, NumPy
- Streamlit

## 🖥️ Run the Application Locally

### 1️⃣ Clone the repository

```bash
git clone <YOUR_REPO_URL>
cd credit-card-fraud-detection
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Streamlit app

```bash
streamlit run app.py
```

## 📂 Project Structure

```
credit-card-fraud-detection/
│
├── app.py                   # Streamlit dashboard and prediction logic
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── model/                   # Directory to store the trained model (e.g., final_model.pkl)
├── data/                    # Directory for sample dataset (e.g., creditcard.csv)
└── .gitignore               # Ignored files
```

## 🎯 Key Highlights

- Real-world fraud detection problem
- Imbalanced classification handling
- Production-style ML pipeline
- Deployment-ready web application
- Resume and interview ready project

## 👤 Author

**Harshitha Konakanchi**

---

⭐ If you find this project useful, feel free to star the repository!
