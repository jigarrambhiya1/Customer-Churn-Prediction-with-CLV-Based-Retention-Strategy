"""
Churn Prediction with Retention Recommendations
Dataset: Telco Customer Churn (Kaggle)
Includes: XGBoost, SHAP Explainability, CLV-based Prioritization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, auc
)
from imblearn.over_sampling import SMOTE
import shap
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

print("=" * 80)
print("CHURN PREDICTION - XGBOOST + SHAP + CLV PRIORITIZATION")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================================================
print("\n[1/7] Loading Telco Churn dataset...")

# Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Or load from local file
try:
    df = pd.read_csv('https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv')
    print("✓ Dataset loaded from URL")
except:
    print("! Loading from local file - ensure 'WA_Fn-UseC_-Telco-Customer-Churn.csv' is in directory")
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

print(f"Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"\nTarget distribution:")
churn_counts = df['Churn'].value_counts()
print(churn_counts)
print(f"Churn rate: {churn_counts['Yes'] / len(df) * 100:.2f}%")
print("⚠️  Class imbalance detected - will handle with SMOTE")

# ============================================================================
# STEP 2: DATA CLEANING & PREPROCESSING
# ============================================================================
print("\n[2/7] Data cleaning and preprocessing...")

# Handle TotalCharges (stored as object due to whitespace)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print(f"  - Converted TotalCharges to numeric ({df['TotalCharges'].isnull().sum()} nulls created)")

# Fill nulls with 0 (customers with 0 tenure have 0 total charges)
df['TotalCharges'].fillna(0, inplace=True)

# Drop customerID (not a feature)
df = df.drop('customerID', axis=1)

# Encode target variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

print(f"✓ Cleaned dataset: {df.shape}")

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

print(f"\nFeature types:")
print(f"  Categorical: {len(categorical_cols)} features")
print(f"  Numerical: {len(numerical_cols)} features")

# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================
print("\n[3/7] Feature engineering...")

# One-hot encode categorical variables
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
print(f"  - One-hot encoding: {X.shape[1]} → {X_encoded.shape[1]} features")

# Feature names for later use
feature_names = X_encoded.columns.tolist()

# ============================================================================
# STEP 4: TRAIN-TEST SPLIT & SCALING
# ============================================================================
print("\n[4/7] Train-test split and scaling...")

# Split data (stratified to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Train set: {X_train.shape[0]:,} samples")
print(f"  Test set:  {X_test.shape[0]:,} samples")

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

print("✓ Features scaled")

# Handle class imbalance with SMOTE
print("\nApplying SMOTE for class imbalance...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"  Before SMOTE: {y_train.value_counts().to_dict()}")
print(f"  After SMOTE:  {y_train_resampled.value_counts().to_dict()}")

# ============================================================================
# STEP 5: MODEL TRAINING & COMPARISON
# ============================================================================
print("\n[5/7] Training models...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5,
        random_state=42,
        eval_metric='logloss'
    )
}

results = {}

for name, model in models.items():
    print(f"\n  Training {name}...")
    model.fit(X_train_resampled, y_train_resampled)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'auc': auc_score
    }
    
    print(f"    AUC-ROC: {auc_score:.4f}")

# Select best model
best_model_name = max(results, key=lambda k: results[k]['auc'])
best_model = results[best_model_name]['model']
best_auc = results[best_model_name]['auc']

print(f"\n✓ Best Model: {best_model_name} (AUC: {best_auc:.4f})")

# ============================================================================
# STEP 6: MODEL EVALUATION
# ============================================================================
print("\n[6/7] Evaluating best model...")

y_pred = results[best_model_name]['y_pred']
y_pred_proba = results[best_model_name]['y_pred_proba']

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('C:\\Users\\Jigar\\downloads\\files\\outputs\\confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrix.png")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'{best_model_name} (AUC = {best_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('C:\\Users\\Jigar\\downloads\\files\\outputs\\roc_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: roc_curve.png")

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, linewidth=2, label=f'PR AUC = {pr_auc:.3f}')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('C:\\Users\\Jigar\\downloads\\files\\outputs\\precision_recall_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: precision_recall_curve.png")

# ============================================================================
# STEP 7: SHAP EXPLAINABILITY
# ============================================================================
print("\n[7/7] Generating SHAP explanations...")

# Create SHAP explainer
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_scaled)

# SHAP Summary Plot (Global feature importance)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, show=False)
plt.title('SHAP Feature Importance - Global View', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('C:\\Users\\Jigar\\downloads\\files\\outputs\\shap_summary.png', dpi=300, bbox_inches='tight')
print("✓ Saved: shap_summary.png")
plt.close()

# SHAP Waterfall plot for a high-risk customer (example)
high_risk_idx = y_pred_proba.argsort()[-1]  # Customer with highest churn probability
plt.figure(figsize=(10, 6))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[high_risk_idx],
        base_values=explainer.expected_value,
        data=X_test_scaled.iloc[high_risk_idx],
        feature_names=feature_names
    ),
    show=False
)
plt.title('SHAP Waterfall - Highest Risk Customer', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('C:\\Users\\Jigar\\downloads\\files\\outputs\\shap_waterfall_high_risk.png', dpi=300, bbox_inches='tight')
print("✓ Saved: shap_waterfall_high_risk.png")
plt.close()

# Extract top 10 most important features
shap_importance = pd.DataFrame({
    'Feature': feature_names,
    'SHAP_Importance': np.abs(shap_values).mean(axis=0)
}).sort_values('SHAP_Importance', ascending=False).head(10)

print("\nTop 10 Churn Drivers (SHAP):")
print(shap_importance.to_string(index=False))

# ============================================================================
# CLV-BASED PRIORITIZATION
# ============================================================================
print("\n" + "=" * 80)
print("CLV-BASED RETENTION PRIORITIZATION")
print("=" * 80)

# Calculate simple CLV = MonthlyCharges × Tenure
test_df = X.iloc[X_test.index].copy()
test_df['ChurnProbability'] = y_pred_proba
test_df['ActualChurn'] = y_test.values

# Simple CLV estimation
test_df['CLV'] = test_df['MonthlyCharges'] * test_df['tenure']

# Revenue at risk = CLV × Churn Probability
test_df['RevenueAtRisk'] = test_df['CLV'] * test_df['ChurnProbability']

# Create risk tiers
test_df['RiskTier'] = pd.cut(
    test_df['ChurnProbability'],
    bins=[0, 0.3, 0.7, 1.0],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

test_df['CLV_Tier'] = pd.cut(
    test_df['CLV'],
    bins=[0, test_df['CLV'].quantile(0.33), test_df['CLV'].quantile(0.67), test_df['CLV'].max()],
    labels=['Low Value', 'Medium Value', 'High Value']
)

# Priority matrix: High CLV + High Churn Risk = Top Priority
# priority_matrix = test_df.groupby(['CLV_Tier', 'RiskTier']).agg({
#     'customerID': 'count',
#     'RevenueAtRisk': 'sum'
# }).rename(columns={'customerID': 'CustomerCount'})

# print("\nPriority Matrix (CLV × Churn Risk):")
# print(priority_matrix)

# Identify top 15% highest-risk, high-value customers
top_priority_threshold = test_df['RevenueAtRisk'].quantile(0.85)
top_priority_customers = test_df[test_df['RevenueAtRisk'] >= top_priority_threshold]

print(f"\n🚨 TOP PRIORITY CUSTOMERS (Top 15% by Revenue at Risk):")
print(f"   Count: {len(top_priority_customers):,}")
print(f"   Total Revenue at Risk: ${top_priority_customers['RevenueAtRisk'].sum():,.2f}")
print(f"   % of Total Revenue at Risk: {top_priority_customers['RevenueAtRisk'].sum() / test_df['RevenueAtRisk'].sum() * 100:.1f}%")

# Visualization: CLV vs Churn Probability scatter
plt.figure(figsize=(12, 7))
scatter = plt.scatter(
    test_df['ChurnProbability'], 
    test_df['CLV'],
    c=test_df['ActualChurn'],
    cmap='coolwarm',
    alpha=0.6,
    s=50,
    edgecolors='k',
    linewidth=0.5
)
plt.axvline(0.7, color='red', linestyle='--', linewidth=1.5, label='High Risk Threshold (70%)')
plt.axhline(test_df['CLV'].quantile(0.67), color='orange', linestyle='--', linewidth=1.5, label='High Value Threshold')
plt.xlabel('Churn Probability', fontsize=12)
plt.ylabel('Customer Lifetime Value ($)', fontsize=12)
plt.title('CLV-Based Retention Prioritization', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Actual Churn (0=No, 1=Yes)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('C:\\Users\\Jigar\\downloads\\files\\outputs\\clv_churn_priority_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: clv_churn_priority_matrix.png")

# ============================================================================
# RETENTION STRATEGY RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("RETENTION STRATEGY RECOMMENDATIONS")
print("=" * 80)

strategies = {
    "High Risk + High Value": [
        "Immediate intervention: Dedicated account manager",
        "Loyalty discount: 15-20% off for 6-month commitment",
        "Priority customer support hotline",
        "Personalized contract upgrade offers"
    ],
    "High Risk + Medium/Low Value": [
        "Automated win-back email sequence",
        "Survey to understand pain points + $10 service credit",
        "Highlight new features/services they may not know about"
    ],
    "Medium Risk + High Value": [
        "Proactive check-in call from customer success team",
        "Early access to new products/services",
        "Loyalty rewards program enrollment"
    ],
    "Low Risk": [
        "Standard retention marketing (newsletters, product updates)",
        "Upsell opportunities (premium tiers, add-ons)"
    ]
}

for tier, actions in strategies.items():
    print(f"\n📌 {tier}")
    for action in actions:
        print(f"   • {action}")

# Save outputs
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Export customer-level predictions
output_df = test_df[['MonthlyCharges', 'tenure', 'CLV', 'ChurnProbability', 
                      'RevenueAtRisk', 'RiskTier', 'CLV_Tier', 'ActualChurn']].copy()
output_df.to_csv('C:\\Users\\Jigar\\downloads\\files\\outputs\\churn_predictions_with_clv.csv', index=False)
print("✓ Saved: churn_predictions_with_clv.csv")

# Export SHAP feature importance
shap_importance.to_csv('C:\\Users\\Jigar\\downloads\\files\\outputs\\shap_feature_importance.csv', index=False)
print("✓ Saved: shap_feature_importance.csv")

# Export priority matrix
# priority_matrix.to_csv('C:\\Users\\Jigar\\downloads\\files\\outputs\\priority_matrix.csv', index=False)
print("✓ Saved: priority_matrix.csv")

print("\n" + "=" * 80)
print("PROJECT COMPLETE")
print("=" * 80)
print("\nKey Findings Summary:")
print(f"  • Model: {best_model_name}")
print(f"  • AUC-ROC: {best_auc:.4f}")
print(f"  • Top churn driver: {shap_importance.iloc[0]['Feature']}")
print(f"  • High-priority customers: {len(top_priority_customers):,} ({len(top_priority_customers)/len(test_df)*100:.1f}%)")
print(f"  • Revenue at risk (top 15%): ${top_priority_customers['RevenueAtRisk'].sum():,.2f}")

print("\n💡 Next Steps:")
print("   1. Validate top drivers with business stakeholders")
print("   2. Deploy model for monthly churn scoring")
print("   3. Integrate with CRM for automated retention triggers")
print("   4. A/B test retention strategies on high-risk segments")

print("\n" + "=" * 80)
