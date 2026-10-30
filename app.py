"""Smart Disease Prediction System - AI-Powered Disease Prediction Using Ensemble Learning
Run:  streamlit run app.py
"""
import html
import inspect
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import auth
import database
import prediction
import preprocessing
import report

BASE = Path(__file__).parent
DISCLAIMER = report.DISCLAIMER

PAGE_DASH = "🏠 Dashboard"
PAGE_PRED = "🩺 Predict Disease"
PAGE_HIST = "📋 Prediction History"
PAGE_ABOUT = "ℹ️ About Project"
PAGE_LOGOUT = "🚪 Logout"
PAGES = [PAGE_DASH, PAGE_PRED, PAGE_HIST, PAGE_ABOUT, PAGE_LOGOUT]

st.set_page_config(page_title="Smart Disease Prediction System", page_icon="🩺", layout="wide")


# ----------------------------- helpers -----------------------------
def stretch(fn):
    """Full-width kwarg that works on both older and newer Streamlit versions."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if "use_container_width" in params:
        return {"use_container_width": True}
    if "width" in params:
        return {"width": "stretch"}
    return {}


def load_css():
    css = BASE / "assets" / "style.css"
    if css.exists():
        st.markdown(f"<style>{css.read_text()}</style>", unsafe_allow_html=True)


def hero():
    st.markdown(
        """<div class="app-hero"><h1>🩺 Smart Disease Prediction System</h1>
        <p>AI-Powered Disease Prediction Using Ensemble Learning</p></div>""",
        unsafe_allow_html=True)


def disclaimer():
    st.markdown(f'<div class="disclaimer">⚠️ <b>Disclaimer:</b> {html.escape(DISCLAIMER)}</div>',
                unsafe_allow_html=True)


def metric_card(label, value):
    st.markdown(f'<div class="metric-card"><div class="label">{html.escape(label)}</div>'
                f'<div class="value">{html.escape(str(value))}</div></div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading trained models...")
def get_artifacts():
    return prediction.load_artifacts()


def goto(page):
    st.session_state["nav"] = page


def history_df(user_id) -> pd.DataFrame:
    rows = database.get_predictions(user_id)
    if not rows:
        return pd.DataFrame(columns=["Date", "Patient", "Age", "Weight (kg)", "BMI",
                                     "Symptoms", "Predicted Disease", "Model Used"])
    return pd.DataFrame([{
        "Date": r["created_at"], "Patient": r["patient_ref"], "Age": r["age"],
        "Weight (kg)": r["weight_kg"], "BMI": r["bmi"],
        "Symptoms": ", ".join(preprocessing.pretty(s) for s in r["symptoms"]),
        "Predicted Disease": r["predicted_disease"], "Model Used": r["model_used"],
    } for r in rows])


# --------------------------- authentication ---------------------------
def auth_view():
    hero()
    tab_login, tab_reg = st.tabs(["🔐 Login", "📝 Register"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email")
            pw = st.text_input("Password", type="password")
            go = st.form_submit_button("Login")
        if go:
            now = time.time()
            locked_until = st.session_state.get("lock_until", 0)
            if now < locked_until:
                st.error(f"Too many failed attempts. Try again in {int(locked_until - now) + 1} seconds.")
            elif not email or not pw:
                st.error("Please enter both email and password.")
            else:
                user = auth.authenticate(email, pw)
                if user:
                    st.session_state.pop("fail_count", None)
                    st.session_state.pop("lock_until", None)
                    st.session_state["user"] = user
                    st.session_state["nav"] = PAGE_DASH
                    st.rerun()
                else:
                    fails = st.session_state.get("fail_count", 0) + 1
                    st.session_state["fail_count"] = fails
                    if fails >= 5:
                        st.session_state["lock_until"] = now + 60
                        st.session_state["fail_count"] = 0
                    st.error("Invalid email or password.")

    with tab_reg:
        with st.form("register_form"):
            name = st.text_input("Full name")
            email_r = st.text_input("Email address")
            pw1 = st.text_input("Password (min 8 characters, letters + numbers)", type="password")
            pw2 = st.text_input("Confirm password", type="password")
            go_r = st.form_submit_button("Create account")
        if go_r:
            ok, msg = auth.register(name, email_r, pw1, pw2)
            (st.success if ok else st.error)(msg)

    disclaimer()


def logout():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()


# ------------------------------ dashboard ------------------------------
def dashboard_view(art, user):
    hero()
    st.subheader(f"Welcome, {user['name']} 👋")
    df = history_df(user["id"])

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Total predictions", len(df))
    with c2:
        metric_card("Most predicted disease",
                    df["Predicted Disease"].mode().iloc[0] if len(df) else "—")
    with c3:
        metric_card("Last prediction", df["Date"].iloc[0][:16] if len(df) else "—")

    st.write("")
    st.button("➕ Start a new prediction", on_click=goto, args=(PAGE_PRED,))

    left, right = st.columns(2)
    with left:
        st.markdown("### Recent predictions")
        if len(df):
            st.dataframe(df.head(5)[["Date", "Patient", "Predicted Disease", "Model Used"]],
                         hide_index=True)
        else:
            st.info("No predictions yet. Use **Predict Disease** to make your first one.")
    with right:
        st.markdown("### Disease distribution (your predictions)")
        if len(df):
            counts = df["Predicted Disease"].value_counts().reset_index()
            counts.columns = ["Disease", "Count"]
            fig = px.bar(counts, x="Disease", y="Count", color="Disease",
                         color_discrete_sequence=px.colors.sequential.Teal_r)
            fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0), height=300)
            st.plotly_chart(fig, **stretch(st.plotly_chart))
        else:
            st.info("The chart appears after you save predictions.")

    st.markdown("### Model evaluation (from your actual test results)")
    if art.metrics and art.metrics.get("models"):
        rows = []
        for key, m in art.metrics["models"].items():
            for metric, label in [("accuracy", "Accuracy"), ("precision", "Precision"),
                                  ("recall", "Recall"), ("f1", "F1-Score")]:
                rows.append({"Model": prediction.MODEL_LABELS.get(key, key),
                             "Metric": label, "Score (%)": m[metric]})
        fig = px.bar(pd.DataFrame(rows), x="Model", y="Score (%)", color="Metric", barmode="group",
                     range_y=[0, 100], color_discrete_sequence=["#0B4F8A", "#0E7C86", "#4FB3BF", "#9AD5DC"])
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=380)
        st.plotly_chart(fig, **stretch(st.plotly_chart))
        st.caption(f"Evaluated on {art.metrics['dataset']['n_test']} held-out test records "
                   f"on {art.metrics.get('evaluated_at', 'n/a')}.")
    else:
        st.info("Evaluation results have not been loaded yet. Run `python train_models.py` "
                "(or place your own `reports/metrics.json`) to display them here.")
    disclaimer()


# ------------------------------ predict ------------------------------
def reset_prediction(features):
    st.session_state.pop("last_result", None)
    for f in features:
        st.session_state[f"sym_{f}"] = False


def predict_view(art, user):
    st.header("🩺 Predict Disease")
    disclaimer()

    for err in art.errors:
        st.warning(err)
    if not art.ready:
        st.error("The trained models could not be loaded, so predictions are unavailable. "
                 "Run `python train_models.py` or copy your model files into the `models/` folder.")
        return

    features = art.meta["features"]
    result = st.session_state.get("last_result")

    if result is None:
        with st.form("predict_form"):
            st.markdown("#### Patient information")
            c1, c2, c3, c4 = st.columns(4)
            ref = c1.text_input("Patient name / ID", max_chars=60,
                                help="Prefer an ID or initials - store only what you need.")
            age = c2.number_input("Age (years)", min_value=0, max_value=120, value=25, step=1)
            weight = c3.number_input("Weight (kg)", min_value=0.0, max_value=400.0, value=60.0, step=0.5)
            height = c4.number_input("Height (cm) - optional", min_value=0.0, max_value=300.0,
                                     value=0.0, step=1.0, help="Enter to calculate BMI; 0 = skip")
            st.caption("Age, weight and height are saved with the record/report but are **not** model "
                       "inputs - the trained model uses the symptoms below only.")

            st.markdown("#### Symptoms")
            cols = st.columns(3)
            for i, f in enumerate(features):
                cols[i % 3].checkbox(preprocessing.pretty(f), key=f"sym_{f}")
            compare = st.checkbox("Also show predictions from individual models", value=True)
            submitted = st.form_submit_button("🔍 Predict Disease")

        if submitted:
            picked = [f for f in features if st.session_state.get(f"sym_{f}")]
            errors = preprocessing.validate_patient(ref, age, weight, height, len(picked))
            if errors:
                for e in errors:
                    st.error(e)
            else:
                try:
                    res = prediction.run_prediction(art, picked, compare=compare)
                except prediction.PredictionError as e:
                    st.error(str(e))
                else:
                    st.session_state["last_result"] = {
                        "patient_ref": ref.strip(), "age": int(age), "weight_kg": float(weight),
                        "height_cm": float(height) if height else None,
                        "bmi": preprocessing.calc_bmi(weight, height), "symptoms": picked,
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "saved": False, **res,
                    }
                    st.rerun()
        return

    # ------------------------- result page -------------------------
    st.markdown(
        f"""<div class="result-card"><div class="small">Predicted disease</div>
        <div class="disease">{html.escape(result['predicted_disease'])}</div>
        <div class="model">Model: <b>{html.escape(result['model_used'])}</b></div></div>""",
        unsafe_allow_html=True)
    if result.get("primary_is_fallback"):
        st.warning("The stacking model file was not found, so a fallback model was used.")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Patient information")
        info = {"Patient": result["patient_ref"], "Age": result["age"],
                "Weight (kg)": result["weight_kg"],
                "Height (cm)": result["height_cm"] or "—", "BMI": result["bmi"] or "—",
                "Symptoms": ", ".join(preprocessing.pretty(s) for s in result["symptoms"])}
        st.table(pd.DataFrame(info.items(), columns=["Field", "Value"]).astype(str).set_index("Field"))
    with right:
        if result.get("individual"):
            st.markdown("#### Individual model predictions")
            st.table(pd.DataFrame(result["individual"].items(),
                                  columns=["Model", "Prediction"]).set_index("Model"))
        if result.get("scores"):
            st.markdown("#### Model confidence scores")
            sc = pd.DataFrame(result["scores"], columns=["Disease", "Score (%)"])
            fig = px.bar(sc, x="Score (%)", y="Disease", orientation="h", range_x=[0, 100],
                         color_discrete_sequence=["#0E7C86"])
            fig.update_layout(yaxis=dict(autorange="reversed"), height=260,
                              margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, **stretch(st.plotly_chart))
            st.caption(f"Scores from the stacking model. Measured calibration error on the test set: "
                       f"{result.get('ece')}%. They are model scores, not medical probabilities.")

    disclaimer()

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if result["saved"]:
            st.success("Saved to history ✔")
        elif st.button("💾 Save to history"):
            database.add_prediction(user["id"], result)
            st.session_state["last_result"]["saved"] = True
            st.rerun()
    with b2:
        st.download_button("📄 Download PDF", data=report.build_pdf(result),
                           file_name=f"prediction_{result['patient_ref']}.pdf", mime="application/pdf")
    with b3:
        st.download_button("📊 Download CSV", data=report.build_csv(result),
                           file_name=f"prediction_{result['patient_ref']}.csv", mime="text/csv")
    with b4:
        st.button("➕ New prediction", on_click=reset_prediction, args=(features,))


# ------------------------------ history ------------------------------
def history_view(user):
    st.header("📋 Prediction History")
    df = history_df(user["id"])
    if df.empty:
        st.info("You have no saved predictions yet.")
        return

    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    q = c1.text_input("🔎 Search (patient, symptom, disease)")
    diseases = c2.multiselect("Filter by disease", sorted(df["Predicted Disease"].unique()))
    dates = pd.to_datetime(df["Date"]).dt.date
    d_from = c3.date_input("From", value=dates.min())
    d_to = c4.date_input("To", value=dates.max())

    mask = (dates >= d_from) & (dates <= d_to)
    if diseases:
        mask &= df["Predicted Disease"].isin(diseases)
    if q:
        mask &= df.astype(str).apply(lambda col: col.str.contains(q, case=False, regex=False)).any(axis=1)
    out = df[mask]

    st.caption(f"Showing {len(out)} of {len(df)} records")
    st.dataframe(out, hide_index=True)
    st.download_button("⬇️ Export CSV", data=out.to_csv(index=False).encode("utf-8"),
                       file_name="prediction_history.csv", mime="text/csv")


# ------------------------------- about -------------------------------
def about_view(art):
    st.header("ℹ️ About Project")
    meta, metrics = art.meta or {}, art.metrics or {}
    ds = metrics.get("dataset", {})

    st.markdown("### Objective")
    st.write("To predict the most likely disease from a patient's reported symptoms using several "
             "machine-learning classifiers combined in a stacking ensemble, and to compare their performance.")

    st.markdown("### Dataset and target")
    if meta:
        st.write(f"**Target column:** `{meta.get('target', 'Disease')}`  \n"
                 f"**Features ({len(meta['features'])}, binary 0/1):** "
                 + ", ".join(preprocessing.pretty(f) for f in meta["features"]))
        st.write("**Disease classes:** " + ", ".join(meta["classes"]))
        if meta.get("dataset_note"):
            st.info(meta["dataset_note"])
    else:
        st.warning("Dataset details are unavailable (models/metadata.json not found).")
    if ds:
        st.write(f"**Records:** {ds['n_rows']} (train {ds['n_train']} / test {ds['n_test']})")

    st.markdown("### Data preprocessing")
    st.write("Rows without a disease label are removed and label text is trimmed. Missing symptom values "
             "are filled with the most frequent value (mode). The imputer is fitted on the training "
             "split only, then applied to test data and to every new prediction.")

    st.markdown("### Classification algorithms")
    st.write("- **Decision Tree** - rule-based splits on symptoms (max depth 8)\n"
             "- **Random Forest** - 200 decision trees voting together\n"
             "- **SVM** - RBF-kernel support vector classifier\n"
             "- **KNN** - 7 nearest neighbours\n"
             "- **Naive Bayes** - Bernoulli Naive Bayes (suited to 0/1 symptom features)")

    st.markdown("### Stacking ensemble architecture")
    st.write("The five models above are the **base learners**. Their predicted class probabilities "
             "(generated with 5-fold cross-validation) become the input of a **Logistic Regression "
             "meta-learner**, which makes the final prediction.")

    st.markdown("### Train-test split and evaluation")
    if ds:
        st.write(f"Stratified {int((1 - ds['test_size']) * 100)}/{int(ds['test_size'] * 100)} split "
                 f"(random_state={ds['random_state']}). Precision, recall and F1 are weighted averages "
                 f"over all disease classes.")
    if metrics.get("models"):
        tbl = pd.DataFrame(metrics["models"]).T.rename(columns={
            "accuracy": "Accuracy (%)", "precision": "Precision (%)", "recall": "Recall (%)",
            "f1": "F1-Score (%)", "roc_auc": "ROC-AUC (%)"})
        tbl.index = [prediction.MODEL_LABELS.get(k, k) for k in tbl.index]
        st.dataframe(tbl)
        st.caption(f"Loaded from reports/metrics.json (evaluated {metrics.get('evaluated_at', 'n/a')}).")
    else:
        st.info("Evaluation results have not been loaded yet.")

    st.markdown("### Limitations and future enhancements")
    st.write("- Only symptoms are used; age, weight and medical history are not model inputs.\n"
             "- Symptoms are binary (present/absent) - severity and duration are not captured.\n"
             "- Diseases with overlapping symptoms (e.g. Cold / Flu / COVID-19) are hard to separate.\n"
             "- Results depend on the dataset used and have not been clinically validated.\n"
             "- Future work: real clinical data, symptom severity, more features, explainability "
             "(SHAP), probability calibration, and deployment with a larger user base.")
    disclaimer()


# ------------------------------- main -------------------------------
def main():
    load_css()
    database.init_db()

    user = st.session_state.get("user")
    if not user:
        auth_view()
        return

    art = get_artifacts()
    if "nav" not in st.session_state:
        st.session_state["nav"] = PAGE_DASH

    with st.sidebar:
        st.markdown("## 🩺 Smart Disease\n## Prediction System")
        st.markdown(f"**{user['name']}**  \n{user['email']}")
        st.markdown("---")
        page = st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")
        st.markdown("---")
        st.caption("Academic project - not for medical use.")

    if page == PAGE_LOGOUT:
        logout()
    elif page == PAGE_DASH:
        dashboard_view(art, user)
    elif page == PAGE_PRED:
        predict_view(art, user)
    elif page == PAGE_HIST:
        history_view(user)
    elif page == PAGE_ABOUT:
        about_view(art)

    st.markdown('<div class="footer">Smart Disease Prediction System - Academic Machine Learning Project</div>',
                unsafe_allow_html=True)


main()
