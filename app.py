import streamlit as st
import pandas as pd
import plotly.express as px
from model_pipeline import run_pipeline

st.set_page_config(layout="wide")

st.title("🎯 AI Early Warning System for Placements")

uploaded_file = st.file_uploader("Upload Student CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Custom dataset loaded!")
else:
    df = run_pipeline()

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")

risk_level_filter = st.sidebar.multiselect(
    "Risk Level",
    df["risk_level"].unique(),
    default=df["risk_level"].unique()
)

risk_filter = st.sidebar.slider("Risk Score", 0, 100, (0, 100))
segment_filter = st.sidebar.multiselect(
    "Segment", df["segment"].unique(), default=df["segment"].unique()
)

filtered_df = df[
    (df["risk_score"].between(risk_filter[0], risk_filter[1])) &
    (df["segment"].isin(segment_filter)) &
    (df["risk_level"].isin(risk_level_filter))
]

# -----------------------------
# KPIs
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Students", len(df))
col2.metric("High Risk Students", len(df[df["risk_score"] > 70]))
col3.metric("Avg Placement Probability", round(df["placement_prob"].mean(), 2))

# -----------------------------
# Charts
# -----------------------------
st.subheader("📊 Risk Distribution")
fig1 = px.histogram(df, x="risk_score", nbins=20)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("📊 Placement Probability")
fig2 = px.histogram(df, x="placement_prob")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("📊 Student Segments")
fig3 = px.pie(df, names="segment")
st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# At-Risk Students
# -----------------------------
st.subheader("🚨 Top At-Risk Students")

top_risk = df.sort_values(by="risk_score", ascending=False).head(20)

st.dataframe(top_risk[[
    "student_id",
    "risk_score",
    "placement_prob",
    "segment",
    "recommendations"
]])

# -----------------------------
# Alerts Section
# -----------------------------
st.subheader("🚨 High Risk Alerts")

high_risk_students = df[df["risk_score"] > 80]

for _, row in high_risk_students.head(5).iterrows():
    st.warning(f"Student {row['student_id']} needs immediate attention!")

# -----------------------------
# Student View
# -----------------------------
# -----------------------------
# Weak Areas Analysis
# -----------------------------
st.subheader("📉 Most Common Weak Areas")

weak_areas = {
    "Coding": (df["coding_score"] < 50).sum(),
    "Communication": (df["communication_score"] < 50).sum(),
    "Aptitude": (df["aptitude_score"] < 50).sum()
}

st.bar_chart(weak_areas)

st.subheader("🎓 Student Lookup")

student_id = st.number_input("Enter Student ID", min_value=0, max_value=1200, step=1)

student = df[df["student_id"] == student_id]

if not student.empty:
    st.metric("Risk Score", round(student["risk_score"].values[0], 2))
    st.metric("Placement Probability", round(student["placement_prob"].values[0], 2))

    st.write("### Segment")
    st.write(student["segment"].values[0])

    st.write("### Recommendations")
    st.write(student["recommendations"].values[0])

    # -----------------------------
# Download Report
# -----------------------------
st.subheader("📥 Download Student Report")

csv = df.to_csv(index=False)

st.download_button(
    label="Download Full Report",
    data=csv,
    file_name="student_risk_report.csv",
    mime="text/csv"
)