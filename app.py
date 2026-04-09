import streamlit as st
import pandas as pd
import plotly.express as px
import json
import smtplib

from model_pipeline import run_pipeline
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Placement AI System",
    page_icon="🎯",
    layout="wide"
)

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "role" not in st.session_state:
    st.session_state.role = None

if "student_id" not in st.session_state:
    st.session_state.student_id = None


# ---------------- LOAD USERS ----------------
def load_users():
    with open("users.json", "r") as f:
        return json.load(f)


# ---------------- LOGIN PAGE ----------------
def login_page():
    st.markdown("<h1 style='text-align:center;'>🎓 Placement AI System</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        <div style="
            background: rgba(255,255,255,0.06);
            padding: 30px;
            border-radius: 20px;
            backdrop-filter: blur(15px);
            border: 1px solid rgba(255,255,255,0.1);
        ">
        """, unsafe_allow_html=True)

        role = st.selectbox("Login As", ["TPC", "Student"])
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        users = load_users()

        if st.button("Login"):
            username = username.strip()
            password = password.strip()

            if username in users and users[username]["password"] == password:

                if users[username]["role"] == role:

                    st.session_state.logged_in = True
                    st.session_state.role = role
                    st.session_state.student_id = username

                    st.rerun()
                else:
                    st.error("Role mismatch!")

            else:
                st.error("Invalid credentials")


# ---------------- PDF REPORT ----------------
def generate_pdf(df):
    pdf = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()

    content = [Paragraph("Placement Report", styles["Title"])]
    pdf.build(content)


# ---------------- EMAIL ALERT ----------------
def send_email(student_id):
    print(f"Alert sent for student {student_id}")


# ---------------- INTERVENTION ----------------
def generate_intervention(score):
    if score > 80:
        return "Immediate mentor + mock interviews"
    elif score > 50:
        return "Weekly coding + aptitude training"
    else:
        return "Soft skills + consistency plan"


# ---------------- ADMIN DASHBOARD ----------------
def show_admin_dashboard(df):
    st.title("📊 Admin Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Students", len(df))
    col2.metric("High Risk", len(df[df["risk_score"] > 80]))
    col3.metric("Avg Probability", round(df["placement_prob"].mean(), 2))

    st.subheader("All Students")
    st.dataframe(df)


# ---------------- STUDENT DASHBOARD ----------------
def show_student_dashboard(df):
    st.title("🎓 My Dashboard")

    sid = st.session_state.student_id
    student = df[df["student_id"] == int(sid)]

    if not student.empty:
        st.metric("Risk Score", float(student["risk_score"].values[0]))
        st.metric("Placement Probability", float(student["placement_prob"].values[0]))

        st.write("Segment:", student["segment"].values[0])
        st.write("Recommendations:", student["recommendations"].values[0])


# ---------------- MAIN APP ----------------
def main_page():

    df = run_pipeline()
    df["intervention"] = df["risk_score"].apply(generate_intervention)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Students", "Analytics"])

    # ---------------- DASHBOARD ----------------
    if page == "Dashboard":

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Students", len(df))
        col2.metric("High Risk", len(df[df["risk_score"] > 80]))
        col3.metric("Avg Prob", f"{df['placement_prob'].mean():.2f}")

        st.plotly_chart(px.histogram(df, x="risk_score"), use_container_width=True)
        st.plotly_chart(px.pie(df, names="segment"), use_container_width=True)

        if st.session_state.role == "TPC":
            show_admin_dashboard(df)
        else:
            show_student_dashboard(df)

    # ---------------- STUDENTS ----------------
    elif page == "Students":

        st.subheader("Top Risk Students")
        st.dataframe(df.sort_values("risk_score", ascending=False).head(20))

        st.subheader("Student Lookup")
        sid = st.number_input("Enter Student ID", step=1)

        student = df[df["student_id"] == sid]

        if not student.empty:
            st.success("Student Found")

            st.metric("Risk Score", float(student["risk_score"].values[0]))
            st.metric("Placement Prob", float(student["placement_prob"].values[0]))
            st.write(student["segment"].values[0])
            st.write(student["recommendations"].values[0])

    # ---------------- ANALYTICS ----------------
    elif page == "Analytics":

        st.subheader("Weak Areas")
        weak = {
            "Coding": (df["coding_score"] < 50).sum(),
            "Communication": (df["communication_score"] < 50).sum(),
            "Aptitude": (df["aptitude_score"] < 50).sum()
        }

        st.bar_chart(weak)

        st.download_button(
            "Download Report",
            df.to_csv(index=False),
            "report.csv"
        )


# ---------------- APP ROUTER ----------------
if not st.session_state.logged_in:
    login_page()
else:
    main_page()