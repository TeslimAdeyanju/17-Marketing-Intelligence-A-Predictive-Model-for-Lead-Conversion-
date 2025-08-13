
# 17-Marketing-Intelligence-A-Predictive-Model-for-Lead-Conversion

## 🎯 Objective

This project aims to build a classification model that predicts whether a lead will convert into a customer. The goal is to help marketing teams prioritize high-value prospects and improve conversion efficiency by implementing a simple and interpretable lead scoring system.

---

## 📦 Dataset

- **Source**: [Kaggle - Leads Dataset](https://www.kaggle.com/ashydv/leads-dataset)
- **Target Variable**: `Converted` (1 = converted, 0 = not converted)
- **Observation**: Each row represents a lead with attributes such as:
  - `Lead Source`
  - `TotalVisits`
  - `Page Views Per Visit`
  - `Last Activity`
  - `Lead Quality`, etc.

---

## 🧼 Step 1: Data Cleaning & Preparation

- Removed duplicates and irrelevant columns
- Treated missing values using:
  - Mode for categorical columns
  - Median for numerical columns
- Encoded categorical features using **One-Hot Encoding**
- Scaled numerical features with **MinMaxScaler**
- Handled class imbalance with **SMOTE (Synthetic Minority Over-sampling Technique)**

---

## 📊 Step 2: Exploratory Data Analysis (EDA)

- Checked class distribution (conversion vs. non-conversion)
- Visualized relationships between key features and conversion
- Key insights:
  - Leads from `Reference` and `Google` showed higher conversion rates
  - More time spent on the website and more visits increased the likelihood of conversion

---

## 🧠 Step 3: Model Building

- **Train/Validation/Test Split**: 60% / 20% / 20%
- **Model Used**: Logistic Regression (simple, interpretable baseline)
- Training pipeline:
  - Fit the model on training data
  - Evaluated on validation set
  - Final performance checked on test set

---

## 📈 Step 4: Model Evaluation

- **Confusion Matrix**
- **Metrics**:
  - Accuracy: `XX.XX%`
  - Precision: `XX.XX%`
  - Recall: `XX.XX%`
  - F1 Score: `XX.XX%`
  - ROC AUC Score: `XX.XX`

> The model demonstrated solid performance, effectively identifying high-conversion leads while minimizing false positives.

---

## 🔍 Step 5: Feature Importance

- Examined model coefficients to interpret feature impact
- Most influential features:
  - `Total Time Spent on Website`
  - `Lead Source`
  - `Page Views Per Visit`
  - `Last Activity`
  - `Lead Origin`

---

## 🧮 Step 6: Lead Scoring System

- Leads were scored using model probabilities:
  - **High Potential**: score > 0.70
  - **Medium Potential**: score between 0.40 and 0.70
  - **Low Potential**: score < 0.40

This scoring helps marketing teams prioritize outreach for high-likelihood leads.

---

## 🖥️ Step 7: Application (Optional)

- Model exported using `pickle`
- Built a **Streamlit app** to:
  - Upload new lead data
  - Predict conversion probabilities
  - Display lead category based on score

---

## 📝 Conclusion

This project shows how machine learning can transform raw marketing data into actionable insights. A lead scoring model enables data-driven decisions, optimizes campaign targeting, and enhances customer acquisition strategies.

---

## 💻 Tools Used

- Python
  - `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
- Handling imbalance: `imblearn` (SMOTE)
- Model: `LogisticRegression`
- App: `Streamlit` (optional)
- Version Control: Git & GitHub

---

## 📂 Folder Structure (Example)

