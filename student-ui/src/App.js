import { useState } from "react";

export default function App() {
  const [loggedIn, setLoggedIn] = useState(false);

  return (
    <div style={styles.container}>
      {!loggedIn ? (
        <Login onLogin={() => setLoggedIn(true)} />
      ) : (
        <Dashboard />
      )}
    </div>
  );
}

/* ---------------- LOGIN PAGE ---------------- */
function Login({ onLogin }) {
  return (
    <div style={styles.loginPage}>
      <div style={styles.loginBox}>
        <h2>Student Login</h2>

        <input placeholder="Email" style={styles.input} />
        <input type="password" placeholder="Password" style={styles.input} />

        <button style={styles.button} onClick={onLogin}>
          Login
        </button>
      </div>
    </div>
  );
}

/* ---------------- DASHBOARD ---------------- */
function Dashboard() {
  return (
    <div style={styles.dashboard}>
      <h1>Student Dashboard</h1>

      <div style={styles.cardContainer}>
        <div style={styles.card}>Total Students: 1200</div>
        <div style={styles.card}>At Risk: 120</div>
        <div style={styles.card}>Placed: 860</div>
      </div>
    </div>
  );
}

/* ---------------- STYLES ---------------- */
const styles = {
  container: {
    fontFamily: "Arial",
  },

  loginPage: {
    height: "100vh",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    background: "linear-gradient(to right, #141e30, #243b55)",
    color: "white",
  },

  loginBox: {
    padding: 30,
    borderRadius: 10,
    background: "rgba(255,255,255,0.1)",
    textAlign: "center",
    width: 300,
  },

  input: {
    width: "100%",
    padding: 10,
    margin: "10px 0",
    borderRadius: 5,
    border: "none",
  },

  button: {
    width: "100%",
    padding: 10,
    background: "#6c5ce7",
    color: "white",
    border: "none",
    borderRadius: 5,
    cursor: "pointer",
  },

  dashboard: {
    padding: 20,
    background: "#0f172a",
    height: "100vh",
    color: "white",
  },

  cardContainer: {
    display: "flex",
    gap: 20,
    marginTop: 20,
  },

  card: {
    padding: 20,
    background: "rgba(255,255,255,0.1)",
    borderRadius: 10,
  },
};