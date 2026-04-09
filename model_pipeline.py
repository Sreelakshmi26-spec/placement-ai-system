import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# -----------------------------
# 1. Generate Synthetic Data
# -----------------------------
def generate_data(n=1200):
    np.random.seed(42)

    data = pd.DataFrame({
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

    # Target logic (realistic)
    score = (
        data["cgpa"] * 10 +
        data["coding_score"] +
        data["communication_score"] +
        data["mock_interview_score"] +
        data["number_of_interviews"] * 5
    )

    data["placement_status"] = (score > 250).astype(int)

    return data

# -----------------------------
# 2. Train Placement Model
# -----------------------------
def train_model(df):
    X = df.drop(["placement_status", "student_id"], axis=1)
    y = df["placement_status"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler

# -----------------------------
# 3. Clustering
# -----------------------------
def cluster_students(df):
    features = df[[
        "cgpa", "coding_score", "communication_score",
        "mock_interview_score", "aptitude_score"
    ]]

    kmeans = KMeans(n_clusters=3, random_state=42)
    df["cluster"] = kmeans.fit_predict(features)

    # Map clusters
    mapping = {0: "Unprepared", 1: "Risky", 2: "Ready"}
    df["segment"] = df["cluster"].map(mapping)

    return df, kmeans

# -----------------------------
# 4. Risk Score
# -----------------------------
def calculate_risk(df, model, scaler):
    X = df.drop(["placement_status", "student_id", "cluster", "segment"], axis=1)
    X_scaled = scaler.transform(X)

    df["placement_prob"] = model.predict_proba(X_scaled)[:, 1]

    # Risk formula
    df["risk_score"] = (
        (1 - df["placement_prob"]) * 50 +
        (100 - df["coding_score"]) * 0.2 +
        (df["last_activity_days"]) * 0.5 +
        (50 - df["number_of_applications"]) * 0.3
    )

    df["risk_score"] = df["risk_score"].clip(0, 100)

    # ✅ ADD THIS (INSIDE FUNCTION ONLY)
    def risk_label(score):
        if score > 70:
            return "High Risk"
        elif score > 40:
            return "Medium Risk"
        else:
            return "Low Risk"

    df["risk_level"] = df["risk_score"].apply(risk_label)

    return df
# -----------------------------
# 5. Recommendation Engine
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
        recs.append("Re-engage in placement portal")

    if row["risk_score"] > 70:
        recs.append("⚠️ High Risk - Immediate TPC intervention")

    return ", ".join(recs)

def apply_recommendations(df):
    df["recommendations"] = df.apply(generate_recommendations, axis=1)
    return df


# -----------------------------
# RUN PIPELINE
# -----------------------------
def run_pipeline():
    df = generate_data()

    model, scaler = train_model(df)
    df, _ = cluster_students(df)
    df = calculate_risk(df, model, scaler)
    df = apply_recommendations(df)

    return df