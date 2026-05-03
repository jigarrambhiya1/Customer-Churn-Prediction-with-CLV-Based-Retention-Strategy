
# Customer Churn Prediction with CLV-Based Retention Strategy

> **Predicting telecom customer churn using XGBoost with SHAP explainability and Customer Lifetime Value prioritization for targeted retention campaigns**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📊 Project Overview

This project builds a machine learning model to predict which telecom customers are at risk of churning, then translates those predictions into actionable retention strategies using:

- **XGBoost classification** with SMOTE for class imbalance (AUC: 0.83)
- **SHAP explainability** to identify top churn drivers (contract type, tenure)
- **CLV-based prioritization matrix** to focus retention efforts on high-value at-risk customers

**Business Impact:** Identified 212 high-priority customers representing 61% ($507K) of total revenue at risk.

---

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| **Model** | XGBoost |
| **AUC-ROC** | 0.8318 |
| **Precision (Churn)** | 0.52 |
| **Recall (Churn)** | 0.71 |
| **Top Churn Driver** | Tenure (SHAP importance: 0.70) |
| **High-Priority Customers** | 212 (15% of test set) |
| **Revenue at Risk (Top 15%)** | $507,535 |

---

## 🗂️ Dataset

**Source:** [Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

- **Size:** 7,043 customers, 21 features
- **Target:** Binary (Churn: Yes/No)
- **Churn Rate:** 26.54% (class imbalance handled with SMOTE)
- **Features:** Demographics, account info, services subscribed, billing details

---

## 🛠️ Tech Stack

**Languages & Libraries:**
- Python 3.8+
- Pandas, NumPy (data processing)
- Scikit-learn (preprocessing, baseline models)
- XGBoost (final model)
- SHAP (explainability)
- Matplotlib, Seaborn (visualization)
- Imbalanced-learn (SMOTE)

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/jigarrambhiya1/churn-prediction-clv.git
cd churn-prediction-clv
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Script
```bash
python churn_prediction_shap_clv.py
```

**Runtime:** ~2-5 minutes (downloads dataset, trains 3 models, generates 8 visualizations)

---

## 📁 Project Structure

```
churn-prediction-clv/
├── churn_prediction_shap_clv.py    # Main script
├── requirements.txt                 # Dependencies
├── README.md                        # This file
└── outputs/                         # Generated files (created on run)
    ├── churn_predictions_with_clv.csv
    ├── shap_feature_importance.csv
    ├── priority_matrix.csv
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── precision_recall_curve.png
    ├── shap_summary.png
    ├── shap_waterfall_high_risk.png
    └── clv_churn_priority_matrix.png
```

---

## 🔍 Methodology

### 1. Data Preprocessing
- Converted `TotalCharges` from object to numeric (handled 11 nulls)
- One-hot encoded 15 categorical features (19 → 30 features)
- Standardized numerical features (tenure, MonthlyCharges, TotalCharges)

### 2. Handling Class Imbalance
- Applied **SMOTE** (Synthetic Minority Oversampling) on training set
- Before: 4,139 No Churn | 1,495 Churn
- After: 4,139 No Churn | 4,139 Churn

### 3. Model Training & Selection
Compared three models on AUC-ROC:

| Model | AUC-ROC |
|-------|---------|
| Logistic Regression | 0.8210 |
| Random Forest | 0.8162 |
| **XGBoost** ✅ | **0.8318** |

**Why XGBoost?**
- Best AUC-ROC performance
- Handles non-linear relationships well
- Native support for missing values
- Works well with SHAP for explainability

### 4. Explainability with SHAP
Used SHAP (SHapley Additive exPlanations) to identify churn drivers:

**Top 5 Churn Drivers:**
1. **Tenure** (0.70) — New customers churn more
2. **Fiber Optic Internet** (0.69) — Service quality issues?
3. **Electronic Check Payment** (0.51) — Friction in payment method
4. **Monthly Charges** (0.49) — Price sensitivity
5. **Two-Year Contract** (0.44) — Contract flexibility matters

### 5. CLV-Based Prioritization
- Estimated **Customer Lifetime Value (CLV)** = Monthly Charges × Tenure
- Created 2×2 matrix: Churn Risk (model probability) × CLV
- **Top 15%** flagged as high-priority (212 customers, $507K at risk)

---

## 📈 Visualizations

### Confusion Matrix
![Confusion Matrix](outputs/confusion_matrix.png)

### ROC Curve
![ROC Curve](outputs/roc_curve.png)

### SHAP Summary Plot
![SHAP Summary](outputs/shap_summary.png)

### CLV × Churn Risk Priority Matrix
![Priority Matrix](outputs/clv_churn_priority_matrix.png)

---

## 💡 Business Recommendations

### 🚨 High Risk + High Value (212 customers, $507K at risk)
- **Immediate intervention:** Dedicated account manager
- **Loyalty discount:** 15-20% off for 6-month commitment
- **Priority support:** Dedicated hotline for service issues
- **Contract upgrade offers:** Incentivize move to annual contracts

### ⚠️ High Risk + Medium/Low Value
- **Automated win-back email sequence**
- **Survey + $10 service credit** to understand pain points
- **Highlight new features/services** they may not know about

### 🟡 Medium Risk + High Value
- **Proactive check-in call** from customer success team
- **Early access to new products/services**
- **Loyalty rewards program enrollment**

### ✅ Low Risk
- **Standard retention marketing** (newsletters, product updates)
- **Upsell opportunities** (premium tiers, add-ons)

---

## 📊 Model Performance Details

### Classification Report
```
              precision    recall  f1-score   support

    No Churn       0.88      0.77      0.82      1035
       Churn       0.52      0.71      0.60       374

    accuracy                           0.75      1409
   macro avg       0.70      0.74      0.71      1409
weighted avg       0.79      0.75      0.76      1409
```

**Interpretation:**
- **71% recall on churn** — catches most at-risk customers
- **52% precision** — some false positives, but retention campaigns are low-cost
- **Trade-off:** Optimized for recall (don't miss churners) over precision

---

## 🎓 Key Learnings

1. **Class imbalance matters** — SMOTE improved minority class recall by 12%
2. **Explainability builds trust** — SHAP made model actionable for business stakeholders
3. **CLV prioritization = ROI** — Not all churn is equal; focus on high-value customers first
4. **Contract flexibility is critical** — Month-to-month customers churn 3x more than annual contracts

---

## 🔮 Future Enhancements

- [ ] **Hyperparameter tuning** (GridSearchCV for XGBoost params)
- [ ] **Feature engineering** — Interaction terms (tenure × contract type)
- [ ] **Survival analysis** — Time-to-churn modeling with Cox regression
- [ ] **Real-time scoring** — Deploy model via Flask API for live churn predictions
- [ ] **A/B test retention strategies** — Measure impact of interventions on actual churn

---

## 📝 How to Use This Project

### For Portfolio/Resume:
- Highlight **SHAP explainability** (rare for student projects)
- Emphasize **CLV prioritization** (shows business thinking)
- Mention **61% of revenue at risk identified** (concrete impact)

### For Interviews:
**Q: "Walk me through your churn project."**

**A:** "I built an XGBoost model to predict telecom customer churn with 83% AUC. The key value wasn't just the model—it was making it actionable. I used SHAP to find that tenure and contract type were the top churn drivers, meaning new customers on flexible contracts were highest risk. Then I prioritized retention using a CLV matrix, focusing on the 15% of customers representing 61% of revenue at risk—that's $507K we could save with targeted campaigns like loyalty discounts or contract upgrades."

**Q: "Why XGBoost over Random Forest?"**

**A:** "XGBoost had 1.5% higher AUC and better precision-recall balance. It also integrates well with SHAP for explainability, which was critical for translating model outputs into marketing actions."

**Q: "How would you deploy this in production?"**

**A:** "Monthly batch scoring to identify at-risk customers, integrated with the CRM to trigger automated retention workflows. I'd also A/B test interventions—control group vs. discount offers—to measure actual churn reduction and ROI."

---

## 🤝 Contributing

This is a personal portfolio project, but suggestions are welcome! Open an issue or submit a pull request.

---

## 📄 License

MIT License - feel free to use for learning/portfolio purposes.

---

## 👤 Author

**Jigar Rambhiya**
- GitHub: [@jigarrambhiya1](https://github.com/jigarrambhiya1)
- LinkedIn: [jigarrambhiya](https://www.linkedin.com/in/jigarrambhiya)
- Email: jigarrambhiya1@gmail.com

---

## 🙏 Acknowledgments

- Dataset: [IBM/Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- SHAP Library: [slundberg/shap](https://github.com/slundberg/shap)
- Inspiration: Marketing Mix Modeling & CLV frameworks from academic research

---

**⭐ If you found this project useful, please star the repository!**

