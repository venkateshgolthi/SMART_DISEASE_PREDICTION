# Smart Disease Prediction System Using Ensemble Model

AI-powered disease prediction from patient symptoms, built with scikit-learn and Streamlit.

> **Disclaimer:** This is an academic machine learning project and is not a substitute for professional medical diagnosis. Predictions may be incorrect. Consult a qualified healthcare professional for medical advice.

## Overview
The system predicts one of **10 diseases** (Allergy, Cold, COVID-19, Dengue, Flu, Gastritis, Malaria, Migraine, Pneumonia, Typhoid) from **9 binary symptoms**. Five classifiers are compared with Voting and Stacking ensembles, and the Stacking Ensemble is deployed in a web application with secure login and prediction history.

## Features
- User registration and login (salted scrypt password hashing, SQLite storage)
- Dashboard with prediction statistics and model evaluation charts
- Symptom-based disease prediction with the Stacking Ensemble
- Side-by-side predictions from individual models
- Per-user prediction history with search, filters and CSV export
- Downloadable PDF / CSV prediction reports

## Dataset
`data/Disease_raw_v2.csv` is a **synthetic (simulated) dataset** generated from the symptom patterns of a reference file. It is **not real patient data**.

| Item | Value |
|---|---|
| Rows | 1,500 (150 per disease) |
| Attributes | 10 (9 binary symptoms + target `Disease`) |
| Symptoms | Fever, Cough, Fatigue, Headache, Sore Throat, Body Pain, Nausea, Vomiting, Runny Nose |
| Missing values | about 5% of symptom cells |
| Split | stratified 80/20 (1,200 train / 300 test) |

## Models and results (test set, 300 records)
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Decision Tree | 77.33% | 79.83% | 77.33% | 76.67% | 95.96% |
| Random Forest | 78.00% | 79.91% | 78.00% | 77.49% | 97.07% |
| SVM | 77.67% | 80.29% | 77.67% | 77.22% | 97.30% |
| Naive Bayes | 79.67% | 81.46% | 79.67% | 79.28% | 97.84% |
| KNN | 77.67% | 80.27% | 77.67% | 76.80% | 95.98% |
| Voting (Soft) | 78.67% | 80.60% | 78.67% | 78.36% | 97.76% |
| **Stacking Ensemble** | 78.00% | 79.64% | 78.00% | 77.76% | 97.61% |

Results apply to the synthetic dataset only and are not clinical performance.

## Screenshots
| Login | Dashboard |
|---|---|
| ![Login](screenshots/login.png) | ![Dashboard](screenshots/dashboard.png) |

| Prediction form | Prediction result |
|---|---|
| ![Form](screenshots/predict_form.png) | ![Result](screenshots/result.png) |

## Tech stack
Python 3.10+, Pandas, NumPy, scikit-learn, joblib, Matplotlib, Seaborn, Plotly, Streamlit, SQLite, ReportLab.

## Installation and usage
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python3 -m pip install -r requirements.txt
python3 train_models.py        # optional: models/ is already included
python3 -m streamlit run app.py
```
Open the Local URL shown in the terminal (usually http://localhost:8501), register an account, then log in.

## Project structure
```
app.py               Streamlit user interface
auth.py              registration, login, password hashing
database.py          SQLite: users and prediction history
preprocessing.py     feature order, imputer, input validation
prediction.py        loads saved models and predicts
report.py            PDF / CSV reports
train_models.py      trains, evaluates and saves the models
assets/              CSS theme
data/                dataset
models/              saved models and metadata
reports/             evaluation metrics (metrics.json)
```

## Authors
- G. Venkatesh (24B11AI127)
- D. Anji Babu (24B11AI093)
- P. Vinay (24B11AI351)
- V. Anil (24B11AI453)

B.Tech, Artificial Intelligence and Machine Learning, Aditya University, Surampalem.

## License
Released under the MIT License (see `LICENSE`).
