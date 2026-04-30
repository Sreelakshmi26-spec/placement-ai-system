import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# -----------------------------
# Generate Data (same as before)
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
# Train Model
# -----------------------------
df = generate_data()

X = df.drop(["placement_status"], axis=1)
y = df["placement_status"]

model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    eval_metric="logloss"
)

model.fit(X, y)

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved as model.pkl")