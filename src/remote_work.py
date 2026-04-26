"""
remote_work.py
Overlay attrition risk scores onto remote/hybrid work patterns.
Inspired by BMW 4-country workforce analysis.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = "models/xgb_attrition_model.pkl"
DATA_PATH = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
OUTPUT_DIR = "outputs"


def assign_work_mode(df: pd.DataFrame) -> pd.DataFrame:
    """
    IBM dataset doesn't have a WorkMode column.
    We simulate it based on department + job role patterns
    (inspired by real BMW remote-work study across DE, PL, HU, IN).
    """
    np.random.seed(42)
    n = len(df)

    # Simulate work mode distribution by department
    dept_remote_probability = {
        "Research & Development": 0.45,
        "Sales": 0.15,
        "Human Resources": 0.30,
    }

    work_modes = []
    for _, row in df.iterrows():
        p_remote = dept_remote_probability.get(row.get("Department_orig", ""), 0.25)
        p_hybrid = 0.40
        p_office = 1 - p_remote - p_hybrid
        mode = np.random.choice(
            ["Fully Remote", "Hybrid", "In-Office"],
            p=[p_remote, p_hybrid, p_office]
        )
        work_modes.append(mode)

    df["WorkMode"] = work_modes
    return df


def assign_country(df: pd.DataFrame) -> pd.DataFrame:
    """Simulate 4-country distribution (BMW study: DE, PL, HU, IN)."""
    np.random.seed(99)
    countries = np.random.choice(
        ["Germany", "Poland", "Hungary", "India"],
        size=len(df),
        p=[0.40, 0.25, 0.20, 0.15],
    )
    df["Country"] = countries
    return df


def load_model_and_data():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    df_raw = pd.read_csv(DATA_PATH)
    df_raw["Department_orig"] = df_raw["Department"]
    df_raw = assign_work_mode(df_raw)
    df_raw = assign_country(df_raw)

    # Encode for model (keep originals for analysis)
    df = df_raw.copy()
    df.drop(columns=["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber",
                      "Department_orig"], inplace=True, errors="ignore")
    df["Attrition"] = (df["Attrition"] == "Yes").astype(int)

    from train import engineer_features
    df = engineer_features(df)

    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        if col not in ["WorkMode", "Country"]:
            df[col] = le.fit_transform(df[col])

    X = df.drop(columns=["Attrition", "WorkMode", "Country"])
    return model, X, df_raw, df


def analyse_workmode_risk(model, X, df_raw, df_encoded):
    probs = model.predict_proba(X)[:, 1]
    df_raw = df_raw.copy()
    df_raw["attrition_risk"] = probs

    # ── Chart 1: Avg attrition risk by work mode ──────────────────────────────
    wm_avg = df_raw.groupby("WorkMode")["attrition_risk"].mean().sort_values(ascending=False)
    print("\n📊 Average Attrition Risk by Work Mode:")
    print(wm_avg.round(3))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Bar chart
    wm_avg.plot(kind="bar", ax=axes[0], color=["#e74c3c", "#f39c12", "#2ecc71"])
    axes[0].set_title("Avg Attrition Risk by Work Mode")
    axes[0].set_ylabel("Mean Churn Probability")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=15)

    # Box plot
    df_raw.boxplot(column="attrition_risk", by="WorkMode", ax=axes[1])
    axes[1].set_title("Risk Distribution by Work Mode")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Attrition Probability")
    plt.sca(axes[1])
    plt.xticks(rotation=15)

    # Country heatmap
    pivot = df_raw.pivot_table(
        values="attrition_risk", index="Country", columns="WorkMode", aggfunc="mean"
    )
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn_r", ax=axes[2])
    axes[2].set_title("Attrition Risk Heatmap\n(Country × Work Mode)")

    plt.suptitle("Remote Work & Attrition Risk Analysis — 4-Country Study", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/attrition_by_workmode.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Chart saved → {OUTPUT_DIR}/attrition_by_workmode.png")

    # Key finding
    remote_risk = df_raw[df_raw["WorkMode"] == "Fully Remote"]["attrition_risk"].mean()
    hybrid_risk = df_raw[df_raw["WorkMode"] == "Hybrid"]["attrition_risk"].mean()
    ratio = remote_risk / hybrid_risk
    print(f"\n🔍 Key Finding: Fully Remote employees show {ratio:.1f}× higher predicted "
          f"attrition risk vs Hybrid peers")
    print(f"   Remote avg: {remote_risk:.3f} | Hybrid avg: {hybrid_risk:.3f}")


if __name__ == "__main__":
    model, X, df_raw, df_enc = load_model_and_data()
    analyse_workmode_risk(model, X, df_raw, df_enc)
