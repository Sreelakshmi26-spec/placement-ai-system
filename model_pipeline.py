import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from xgboost import XGBClassifier

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
# 2. Train Model
# -----------------------------
def train_model(df):
    X = df.drop(["placement_status", "student_id"], axis=1)
    y = df["placement_status"]

    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    return model


# -----------------------------
# 3. Clustering
# -----------------------------
def cluster_students(df):
    features = df[[
        "cgpa",
        "coding_score",
        "communication_score",
        "mock_interview_score",
        "aptitude_score"
    ]]

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(features)

    mapping = {
        0: "Unprepared",
        1: "At Risk",
        2: "Ready"
    }

    df["segment"] = df["cluster"].map(mapping)

    return df, kmeans


# -----------------------------
# 4. Risk Calculation
# -----------------------------
def calculate_risk(df, model):
    X = df.drop(
        ["placement_status", "student_id", "cluster", "segment"],
        axis=1
    )

    df["placement_prob"] = model.predict_proba(X)[:, 1]

    df["risk_score"] = (
        (1 - df["placement_prob"]) * 50 +
        (100 - df["coding_score"]) * 0.2 +
        df["last_activity_days"] * 0.5 +
        (50 - df["number_of_applications"]) * 0.3
    )

    df["risk_score"] = df["risk_score"].clip(0, 100)

    def label(x):
        if x > 70:
            return "High Risk"
        elif x > 40:
            return "Medium Risk"
        else:
            return "Low Risk"

    df["risk_level"] = df["risk_score"].apply(label)

    return df


# -----------------------------
# 5. Recommendations
# -----------------------------
def generate_recommendations(row):
    recs = []

    if row["coding_score"] < 50:
        recs.append("Practice DSA daily")

    if row["communication_score"] < 50:
        recs.append("Improve communication skills")

    if row["aptitude_score"] < 50:
        recs.append("Practice aptitude tests")

    if row["number_of_applications"] < 10:
        recs.append("Apply to more companies")

    if row["last_activity_days"] > 15:
        recs.append("Re-engage with placement portal")

    if row["risk_score"] > 70:
        recs.append("⚠️ Immediate TPC intervention needed")

    return ", ".join(recs)


def apply_recommendations(df):
    df["recommendations"] = df.apply(generate_recommendations, axis=1)
    return df


# -----------------------------
# 6. Full Pipeline
# -----------------------------
def run_pipeline():
    df = generate_data()

    model = train_model(df)

    df, _ = cluster_students(df)

    df = calculate_risk(df, model)

    df = apply_recommendations(df)

    return df