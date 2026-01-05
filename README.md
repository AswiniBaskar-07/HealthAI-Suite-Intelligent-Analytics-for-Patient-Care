**🏥 HealthAI Suite – Intelligent Analytics for Patient Care**
**📌 Project Overview**

HealthAI Suite is an end-to-end machine learning application focused on patient risk stratification using structured healthcare data.
The project demonstrates the complete ML lifecycle — from data preprocessing and model training to deployment using Streamlit and FastAPI.

**⚠️ Scope Clarification (Important):**
The original HealthAI Suite problem statement includes multiple optional modules (CNN, NLP, Chatbot, Translator, etc.).
This submission focuses exclusively on the Risk Stratification (Classification) module, which satisfies the mandatory project requirements.
Other modules are considered optional extensions and are not implemented in this version.

**🎯 Problem Statement**

Predict whether a hospital patient is at high risk or low risk based on demographic details, vitals, lab values, medical history, and hospitalization details.

**This helps:**

Early identification of high-risk patients

Better clinical decision support

Improved hospital resource planning

🧠 Implemented Module
✅ Risk Stratification (Classification)

Binary classification: High Risk vs Low Risk

**Input:** Tabular healthcare data

**Output**: Risk label + probability score

**📂 Project Structure**
HealthAI-Suite-Intelligent-Analytics-for-Patient-Care/
│
├── Healthcare.ipynb              # Data preprocessing, EDA, model training
├── Healthcare.py                 # Streamlit dashboard (UI)
├── api.py                        # FastAPI backend for predictions
├── HealthCare_Data_Enhanced.csv  # Synthetic healthcare dataset
├── final_healthcare_risk_model.pkl  # Trained ML model
├── feature_names.pkl             # Model feature order
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation

**📊 Dataset Description**

Type: Synthetic tabular healthcare dataset

**Records:** ~10,000 patients

**Source:** Simulated data created using realistic medical assumptions

**Privacy:** No real patient data used (fully anonymized & synthetic)

**Key Features**

**Demographics**: age, gender, blood type

**Vitals:** BP, heart rate, oxygen saturation, temperature

Medical conditions: diabetes, heart disease

Hospital data: ICU, length of stay, treatment cost

Target variable: outcome_label (0 = Low Risk, 1 = High Risk)

🔍 Exploratory Data Analysis (EDA)

Performed in Healthcare.ipynb:

Missing value handling

Feature distributions

Correlation analysis

Categorical encoding

Feature scaling using pipelines

**🤖 Model Details**

**Task:** Binary Classification

Model Type: Machine Learning classifier (via Scikit-learn pipeline)

Preprocessing:

Numerical scaling

Categorical encoding

Evaluation Metrics:

Accuracy

ROC-AUC

Confusion Matrix

The trained model is saved as:

final_healthcare_risk_model.pkl

**🚀 How to Run the Project**
1️⃣ Install Dependencies
pip install -r requirements.txt

**2️⃣ Run the Jupyter Notebook (Optional – Training)**
jupyter notebook Healthcare.ipynb


This step is optional if you are using the pre-trained model.

**3️⃣** Run Streamlit Dashboard****
streamlit run Healthcare.py


**Features:**

Doctor login (demo credentials)

Auto-fill patient data from CSV

Manual data entry

Risk prediction with probability

Patient image upload (optional)

4️⃣ Run FastAPI Backend
uvicorn api:app --reload


**API available at:**

http://127.0.0.1:8000


**Swagger docs:**

http://127.0.0.1:8000/docs

**🔗 API Endpoint**
POST /predict

**Input: **JSON patient data
**Output:** Risk label + probability

**Example response:**

{
  "risk": "HIGH",
  "probability": 0.87
}

**⚖️ Ethics & Compliance**

No real patient data used

Dataset is fully synthetic

No PII included

Designed for educational purposes only

Complies with ethical AI guidelines

**⚠️ Limitations**

Uses synthetic data (not real hospital records)

Only one ML module implemented

Model performance may not generalize to real-world clinical data

No deep learning or NLP modules included

**🔮 Future Enhancements**

Regression (Length of Stay prediction)

Patient clustering

Association rule mining

Deep learning models (CNN, LSTM)

NLP with BioBERT

Healthcare chatbot

Docker deployment

MLflow experiment tracking

**🛠 Tech Stack**

Python

Pandas, NumPy

Scikit-learn

Streamlit

FastAPI

Joblib

Matplotlib / Seaborn

**👨‍⚕️ Demo Login (Streamlit)**
Doctor ID: admin
Password: admin123

**📌 Final Note**

This project demonstrates end-to-end ML system design, focusing on quality, deployment, and clarity, rather than implementing every optional module.

## 👤 Author

Aswini Baskar


