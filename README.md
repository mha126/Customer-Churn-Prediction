# 📊 Customer Churn Classifier (Tabular ML Project)

This project trains a **classic machine learning model** on a **tabular customer churn dataset**.

It predicts whether a customer is likely to **churn (leave)** based on:

- Tenure (months subscribed)
- Monthly charges
- Number of support tickets
- Whether they use the mobile app
- Contract type (month-to-month, one-year, two-year)
- Payment method (credit card, bank transfer, PayPal)

The dataset is **synthetic but realistic**, generated to reflect typical churn behavior, and stored in `data/customer_churn.csv`.

## 🔧 Tech stack

- Python
- pandas, numpy
- scikit-learn
- RandomForestClassifier
- Train/test split, preprocessing pipelines, evaluation reports

## 📁 Project structure

```text
tabular-ml-churn/
├── train.py
├── predict.py
├── requirements.txt
└── data/
    └── customer_churn.csv
```

## 🚀 Getting started

```bash
pip install -r requirements.txt
python train.py
```

This will:

- Load `data/customer_churn.csv`
- Split into train/test
- Build a preprocessing + model pipeline
- Train a RandomForest classifier
- Print classification metrics
- Save the trained model to `churn_model.joblib`

## 🔍 Making a prediction

After training:

```bash
python predict.py 5 80 4 0 month-to-month credit_card
```

Example output:

```text
Churn probability: 0.78
Predicted churn label: 1 (1 = churn, 0 = stay)
```

## 🧠 CV description

You can describe this project on your CV as:

> Developed a customer churn prediction model on tabular data using scikit-learn. Built a full pipeline with preprocessing, feature encoding, train/test split, and a RandomForest classifier, and exposed a simple CLI for inference on new customer profiles.
