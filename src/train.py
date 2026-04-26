"""
train.py
XGBoost attrition classifier with SMOTE oversampling + SHAP explanations.
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay
)
from imblearn.over_sampling import SMOTE

DATA_PATH = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_PATH = "models/xgb_attrition_model.pkl"
OUTPUT_DIR = "outputs"

os.makedirs("models", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Load & clean ──────────────────────────────────────────────────────────

def load_and_clean(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    # These columns have zero variance — drop them
    df.drop(columns=["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"],
            inplace=True, errors="ignore")

    # Binary target
    df["Attrition"] = (df["Attrition"] == "Yes").astype(int)

    return df


# ── 2. Feature engineering ────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Promotion lag flag
    df["promotion_lag_flag"] = (df["YearsSinceLastPromotion"] > 3).astype(int)

    # Career progression ratio
    df["career_ratio"] = df["YearsAtCompany"] / (df["TotalWorkingYears"] + 1)

    # Satisfaction composite
    df["satisfaction_composite"] = (
        df["JobSatisfaction"] + df["EnvironmentSatisfaction"] + df["RelationshipSatisfaction"]
    ) / 3

    # Overtime binary
    df["OverTime_bin"] = (df["OverTime"] == "Yes").astype(int)

    return df


def encode_and_split(df: pd.DataFrame):
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop(columns=["Attrition"])
    y = df["Attrition"]

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ── 3. SMOTE ─────────────────────────────────────────────────────────────────

def apply_smote(X_train, y_train):
    print(f"Before SMOTE — Churn: {y_train.sum()}, No-Churn: {(y_train==0).sum()}")
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE  — Churn: {y_res.sum()}, No-Churn: {(y_res==0).sum()}")
    return X_res, y_res


# ── 4. Train ─────────────────────────────────────────────────────────────────

def train_model(X_train, y_train):
    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.75,
        colsample_bytree=0.75,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42,
    )
    model.fit(X_train, y_train, verbose=False)
    return model


# ── 5. Evaluate ──────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    print(f"\nROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Stay", "Attrition"]))

    # ROC
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax, name="XGBoost + SMOTE")
    ax.set_title("ROC Curve — Employee Attrition")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/roc_curve.png", dpi=150)
    plt.close()

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Stay", "Attrition"], ax=ax
    )
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
    plt.close()


# ── 6. SHAP ──────────────────────────────────────────────────────────────────

def generate_shap(model, X_test):
    print("\nGenerating SHAP explanations...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Feature Importance — Attrition Drivers")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP plot → {OUTPUT_DIR}/shap_summary.png")


# ── 7. Business impact simulation ─────────────────────────────────────────────

def simulate_retention_savings(model, X_test, y_test, avg_salary=60000, replacement_rate=1.0):
    """
    Simulate retention savings. Replacement cost ≈ 50–200% of annual salary.
    Default: 100% (mid-range industry estimate).
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    results = X_test.copy()
    results["attrition_prob"] = y_prob
    results["replacement_cost_at_risk"] = y_prob * avg_salary * replacement_rate

    total_at_risk = results["replacement_cost_at_risk"].sum()
    high_risk = results[results["attrition_prob"] > 0.6]

    print(f"\n💰 Total replacement cost at risk: €{total_at_risk:,.0f}")
    print(f"🔴 High-risk employees (prob > 0.60): {len(high_risk)}")
    print(f"   → At-risk cost (high tier): €{high_risk['replacement_cost_at_risk'].sum():,.0f}")

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_and_clean(DATA_PATH)
    df = engineer_features(df)
    X_train, X_test, y_train, y_test = encode_and_split(df)

    print(f"Class balance — Train: {y_train.mean():.2%} attrition")

    X_res, y_res = apply_smote(X_train, y_train)
    model = train_model(X_res, y_res)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_res, y_res, cv=cv, scoring="roc_auc")
    print(f"\nCV ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    evaluate(model, X_test, y_test)
    generate_shap(model, X_test)
    simulate_retention_savings(model, X_test, y_test)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\n✅ Model saved → {MODEL_PATH}")
