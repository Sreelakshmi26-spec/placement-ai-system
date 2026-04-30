import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.cluster import KMeans

# -----------------------------
# 1. Generate Data
# -----------------------------
def generate_data(n=1200):
    np.random.seed(42)

    df = pd.DataFrame({
        "student_id": range(n),
        "attendance_percentage": np.random.randint(50, 100, n),
        "cgpa": np.round(np.random.uniform(5, 10, n), 2),
        "aptitude_score": np.random.randint(30, 100, n),
        "coding_score": np.random.randint(20, 100, n),
        "communication_score": np.random.randint(30, 100, n),
        "mock_interview_score": np.random.randint(20, 100, n),
        "number_of_applications": np.random.randint(0, 50, n),
        "number_of_interviews": np.random.randint(0, 10, n),
        "internship_experience": np.random.choice([0, 1], n),
        "projects_count": np.random.randint(0, 5, n),
        "last_activity_days": np.random.randint(0, 60, n),
    })

    score = (
        df["cgpa"] * 10 +
        df["coding_score"] +
        df["communication_score"] +
        df["mock_interview_score"] +
        df["number_of_interviews"] * 5
    )

    df["placement_status"] = (score > 250).astype(int)

    return df


# -----------------------------
# 2. Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


# -----------------------------
# 3. Expected Features (IMPORTANT)
# -----------------------------
EXPECTED_COLUMNS = [
    "student_id",
    "attendance_percentage",
    "cgpa",
    "aptitude_score",
    "coding_score",
    "communication_score",
    "mock_interview_score",
    "number_of_applications",
    "number_of_interviews",
    "internship_experience",
    "projects_count",
    "last_activity_days"
]


# -----------------------------
# 4. Main Pipeline
# -----------------------------
@st.cache_data(show_spinner=False)
def run_pipeline(input_df=None):

    model = load_model()

    # -----------------------------
    # Data Input
    # -----------------------------
    if input_df is None:
        df = generate_data()
    else:
        df = input_df.copy()

    # -----------------------------
    # 🔥 FIX: HANDLE FEATURE MISMATCH
    # -----------------------------
    
    # Add missing columns
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    # Remove extra columns (like placement_status)
    df = df[[col for col in EXPECTED_COLUMNS if col in df.columns]]

    # Ensure correct order
    df = df[EXPECTED_COLUMNS]

    # -----------------------------
    # Prediction
    # -----------------------------
    df["placement_prob"] = model.predict_proba(df)[:, 1]

    # -----------------------------
    # Risk Score
    # -----------------------------
    df["risk_score"] = (
        (1 - df["placement_prob"]) * 50 +
        (100 - df["coding_score"]) * 0.2 +
        (df["last_activity_days"]) * 0.5 +
        (50 - df["number_of_applications"]) * 0.3
    )

    df["risk_score"] = df["risk_score"].clip(0, 100)

    def risk_label(score):
        if score > 70:
            return "High Risk"
        elif score > 40:
            return "Medium Risk"
        else:
            return "Low Risk"

    df["risk_level"] = df["risk_score"].apply(risk_label)

    # -----------------------------
    # Segmentation
    # -----------------------------
    features = df[[
        "cgpa", "coding_score", "communication_score",
        "mock_interview_score", "aptitude_score"
    ]]

    kmeans = KMeans(n_clusters=3, random_state=42)
    df["cluster"] = kmeans.fit_predict(features)

    mapping = {0: "Unprepared", 1: "Risky", 2: "Ready"}
    df["segment"] = df["cluster"].map(mapping)

    # -----------------------------
    # Recommendations
    # -----------------------------
    def generate_recommendations(row):
        recs = []

        if row["coding_score"] < 50:
            recs.append("Practice DSA daily")

        if row["communication_score"] < 50:
            recs.append("Attend mock interviews")

        if row["aptitude_score"] < 50:
            recs.append("Practice aptitude tests")

        if row["number_of_applications"] < 10:
            recs.append("Apply to more companies")

        if row["last_activity_days"] > 15:
            recs.append("Re-engage in portal")

        if row["risk_score"] > 70:
            recs.append("⚠️ Immediate TPC intervention")

        return ", ".join(recs)

    df["recommendations"] = df.apply(generate_recommendations, axis=1)

    return df