import streamlit as st
import pandas as pd
import plotly.express as px
import json
import requests

from model_pipeline import run_pipeline
from streamlit_lottie import st_lottie

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Placement AI System",
    page_icon="🎯",
    layout="wide"
)

# ---------------- LOAD USERS ----------------
def load_users():
    try:
        with open("users.json", "r") as f:
            return json.load(f)
    except:
        return {}

# ---------------- LOTTIE ----------------
@st.cache_data
def load_lottie(url):
    return requests.get(url).json()

# ---------------- GLOBAL CSS (INSANE UI) ----------------
st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background: radial-gradient(circle at top, #020617, #0f172a, #1e293b);
    color: white;
}

/* TITLE GLOW */
h1 {
    background: linear-gradient(90deg,#38bdf8,#818cf8,#22d3ee);
    -webkit-background-clip: text;
    color: transparent;
    text-align: center;
}

/* GLASS CARD */
.card {
    background: rgba(255,255,255,0.06);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255,255,255,0.1);
    transition: 0.3s;
}

/* HOVER EFFECT */
.card:hover {
    transform: translateY(-5px) scale(1.02);
    box-shadow: 0px 10px 40px rgba(0,0,0,0.6);
}

/* METRICS */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    border-radius: 15px;
    padding: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(90deg,#4f46e5,#06b6d4);
    border-radius: 12px;
    padding: 10px;
    font-weight: bold;
    color: white;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0px 5px 20px rgba(79,70,229,0.5);
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #020617;
}

/* SCROLLBAR */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: #4f46e5;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "role" not in st.session_state:
    st.session_state.role = None

if "student_id" not in st.session_state:
    st.session_state.student_id = None

if "data" not in st.session_state:
    st.session_state.data = None

# ---------------- LOGIN PAGE ----------------
def login_page():

    st.markdown("<h1>🎓 Placement AI System</h1>", unsafe_allow_html=True)

    lottie = load_lottie("https://assets5.lottiefiles.com/packages/lf20_jcikwtux.json")
    st_lottie(lottie, height=220)

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.markdown("""
<h3 style='text-align:center;'>🚀 Smarter Placements Start Here</h3>

<p style='text-align:center; font-size:14px; color:#cbd5f5;'>
AI-powered system to predict placement success, identify at-risk students,
and enable early interventions for better outcomes.
</p>

<hr style='border: 0.5px solid rgba(255,255,255,0.1); margin:15px 0;'>
""", unsafe_allow_html=True)

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        role = st.selectbox("Login As", ["TPC (Admin)", "Student"])

        #users = load_users()

        if st.button("Login"):
            username = username.strip()
            password = password.strip()
            
            users = load_users()
            #st.write("DEBUG USERS:", users)

            if username in users:

                user_data = users[username]

                if str(user_data["password"]).strip() == password:

                    st.session_state.logged_in = True
                    st.session_state.role = user_data["role"]

            # set student id only if student
                    if user_data["role"] == "Student":
                        st.session_state.student_id = int(username)

                    st.success("Login successful!")
                    st.rerun()

                else:
                    st.error("Wrong password")

            else:
                st.error("User not found")

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- INTERVENTION ----------------
def generate_intervention(score):
    if score > 80:
        return "🚨 Immediate mentor + daily mock interviews"
    elif score > 50:
        return "📚 Weekly coding + aptitude training"
    else:
        return "✅ Maintain consistency + improve soft skills"

# ---------------- STUDENT CARDS ----------------
def student_cards(df):
    st.subheader("🎯 Top Students (Interactive Cards)")

    cols = st.columns(4)

    for i, row in df.head(8).iterrows():
        with cols[i % 4]:
            with st.container():
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.write(f"**ID:** {row['student_id']}")
                st.write(f"Risk: {round(row['risk_score'],2)}")
                st.write(f"Segment: {row['segment']}")
                st.write(f"💡 {row['intervention']}")
                st.markdown("</div>", unsafe_allow_html=True)

# ---------------- MAIN PAGE ----------------
def main_page():

    # LOAD DATA ONCE
    if st.session_state.data is None:

        uploaded_file = st.file_uploader("Upload Dataset (optional)", type=["csv"])

        if uploaded_file:
            df_input = pd.read_csv(uploaded_file)
            with st.spinner("🚀 AI analyzing uploaded data..."):
                st.session_state.data = run_pipeline(df_input)
        else:
            with st.spinner("🚀 AI analyzing student data..."):
                st.session_state.data = run_pipeline()

    df = st.session_state.data

    df["intervention"] = df["risk_score"].apply(generate_intervention)

    # SIDEBAR
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Students", "Analytics"])

    # ---------------- DASHBOARD ----------------
    if page == "Dashboard":

        st.markdown("<h2>📊 Dashboard Overview</h2>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("🎓 Total Students", len(df))
        col2.metric("🚨 High Risk", len(df[df["risk_score"] > 80]))
        col3.metric("📈 Avg Probability", f"{df['placement_prob'].mean():.2f}")

        st.plotly_chart(px.histogram(df, x="risk_score"), use_container_width=True)
        st.plotly_chart(px.pie(df, names="segment"), use_container_width=True)

        student_cards(df.sort_values("risk_score", ascending=False))

    # ---------------- STUDENTS ----------------
    elif page == "Students":

        st.subheader("🚨 Top Risk Students")
        st.dataframe(df.sort_values("risk_score", ascending=False).head(20))

        st.subheader("🔍 Student Lookup")
        sid = st.number_input("Enter Student ID", step=1)

        student = df[df["student_id"] == sid]

        if not student.empty:
            student = student.iloc[0]

            st.success("Student Found")

            st.metric("Risk Score", float(student["risk_score"]))
            st.metric("Placement Prob", float(student["placement_prob"]))

            st.write("Segment:", student["segment"])
            st.write("Recommendations:", student["recommendations"])
            st.write("AI Plan:", student["intervention"])

    # ---------------- ANALYTICS ----------------
    elif page == "Analytics":

        st.subheader("📉 Weak Areas")

        weak = {
            "Coding": (df["coding_score"] < 50).sum(),
            "Communication": (df["communication_score"] < 50).sum(),
            "Aptitude": (df["aptitude_score"] < 50).sum()
        }

        st.bar_chart(weak)

        st.subheader("📥 Download Report")

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "report.csv"
        )

# ---------------- ROUTER ----------------
if not st.session_state.logged_in:
    login_page()
else:
    main_page()