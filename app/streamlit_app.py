"""Streamlit front end for the bank churn project.

    streamlit run app/streamlit_app.py

Talks to the FastAPI service when API_URL is set (e.g. API_URL=http://127.0.0.1:8000),
otherwise loads models/churn_model.joblib directly.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                      # so "import churn" works when Streamlit starts this file

import pandas as pd
import streamlit as st

from app import ui
from app.client import ChurnClient, ClientError
from churn import config

PRESETS = {
    "likely_leaver": dict(cs=600, country="Germany", gender="Male", age=45, tenure=3, balance=120000, products=1, cc=True, active=False, salary=55000),
    "likely_stayer": dict(cs=720, country="France", gender="Female", age=30, tenure=5, balance=60000, products=2, cc=True, active=True, salary=90000),
}
PLOT_CONFIG = {"displayModeBar": False}

st.set_page_config(page_title="Bank churn risk", layout="wide", initial_sidebar_state="collapsed")
st.markdown(ui.CSS, unsafe_allow_html=True)


@st.cache_resource
def get_client(api_url: str) -> ChurnClient:
    return ChurnClient(api_url)


client = get_client(os.getenv("API_URL", ""))


def safe(fn, *args, **kwargs):
    """Run a client call; show a readable message instead of a traceback."""
    try:
        return fn(*args, **kwargs)
    except ClientError as e:
        st.error(str(e))
        return None


# =============================================================================== sidebar
with st.sidebar:
    st.markdown("### Bank churn risk")
    health = client.health()
    if health.get("model_loaded"):
        st.markdown(ui.status(True, "Model ready"), unsafe_allow_html=True)
    elif health.get("error"):
        st.markdown(ui.status(False, "API not reachable"), unsafe_allow_html=True)
    else:
        st.markdown(ui.status(False, "Model not loaded"), unsafe_allow_html=True)
        st.caption("Train it first:  `python -m churn.train`" if not client.api_url else "The API reports no model.")
    st.caption(f"Scoring via: {client.mode}")
    if client.api_url:
        st.markdown(f"[API docs]({os.getenv('API_PUBLIC_URL', client.api_url)}/docs)")
    st.divider()
    st.caption("The score ranks customers; it is not a probability. See *About the model*.")

if health.get("error"):
    st.title("Bank churn risk")
    st.error(health["error"])
    st.markdown(f"The app is set to use the API at **{client.api_url}**. Check that:\n"
                f"1. the API is running (`uvicorn api.main:app --port <port>`) in another window;\n"
                f"2. the port in `API_URL` is the same port you started it on;\n"
                f"3. `{client.api_url}/docs` opens a page titled **Bank Churn Risk API**. If it shows something else, another program "
                "is using that port: start this API on a different port (for example 8010) and update `API_URL` to match.")
    st.stop()
if not health.get("model_loaded"):
    st.title("Bank churn risk")
    st.warning("There is no trained model yet. Run `python -m churn.train` (about a minute), then refresh this page.")
    st.stop()

info = safe(client.model_info) or {}
threshold = info.get("threshold", 0.65)
threshold_pct = info.get("threshold_percentile")

st.title("Bank churn risk")
flag_share = f", about the top {100 - threshold_pct:.1f}% of customers," if threshold_pct is not None else ""
st.markdown(f'<p class="lede">Score a customer, see what is driving the score, and decide who gets a retention call. '
            f'Customers scoring <b>{threshold:.2f} or higher</b>{flag_share} are flagged.</p>', unsafe_allow_html=True)

tab_score, tab_batch, tab_about = st.tabs(["Score a customer", "Batch scoring", "About the model"])

# =============================================================================== 1. score a customer
with tab_score:
    for k, v in PRESETS["likely_leaver"].items():
        st.session_state.setdefault(f"w_{k}", v)

    def load_preset(name: str):
        for k, v in PRESETS[name].items():
            st.session_state[f"w_{k}"] = v

    b1, b2, _ = st.columns([1.35, 1.35, 3.3])
    b1.button("Example: likely to leave", on_click=load_preset, args=("likely_leaver",))
    b2.button("Example: likely to stay", on_click=load_preset, args=("likely_stayer",))

    left, right = st.columns([5, 6], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="grp">Profile</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            gender = c1.selectbox("Gender", config.GENDERS, key="w_gender")
            country = c2.selectbox("Country", config.COUNTRIES, key="w_country")
            age = st.slider("Age", 18, 92, key="w_age")
            st.markdown('<div class="grp">Account</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            credit_score = c1.number_input("Credit score", 300, 900, key="w_cs", step=10)
            products = c2.selectbox("Products held", [1, 2, 3, 4], key="w_products")
            balance = c1.number_input("Balance", 0.0, 1e7, key="w_balance", step=5000.0, format="%.0f")
            salary = c2.number_input("Estimated salary", 0.0, 1e7, key="w_salary", step=5000.0, format="%.0f")
            tenure = st.slider("Years as customer", 0, 10, key="w_tenure")
            st.markdown('<div class="grp">Activity</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            credit_card = c1.toggle("Has credit card", key="w_cc")
            active = c2.toggle("Active member", key="w_active")

    customer = dict(credit_score=int(credit_score), country=country, gender=gender, age=int(age), tenure=int(tenure),
                    balance=float(balance), products=int(products), credit_card=bool(credit_card), active=bool(active), salary=float(salary))
    result = safe(client.predict, customer)

    with right:
        if result:
            st.markdown(ui.verdict(result["flagged"], result["risk_score"], result["threshold"]), unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Risk score", f"{result['risk_score']:.2f}", help="0 to 1. A ranking score, not a probability.")
            m2.metric("Percentile", f"{result['risk_percentile']:.0f}%", help="Share of training customers with a lower risk score than this one.")
            m3.metric("Risk band", result["risk_band"])
            st.markdown(ui.risk_strip(result["risk_percentile"], threshold_pct, result["threshold"], result["risk_band"]), unsafe_allow_html=True)

            drivers = result.get("drivers")
            if drivers:
                st.markdown("#### What is pushing this score")
                st.plotly_chart(ui.drivers_chart(drivers), theme=None, config=PLOT_CONFIG)
                up = [x for x in drivers if x["impact"] > 0]
                if up:
                    st.markdown(f"In plain words: the biggest push towards leaving is **{up[0]['label']} ({up[0]['value']})**"
                                + (f", then **{up[1]['label']} ({up[1]['value']})**." if len(up) > 1 else "."))
                else:
                    st.markdown("In plain words: nothing in this profile pushes the score up.")
            st.caption("Drivers come from SHAP: each bar is how far that factor moves this customer's score. They explain the model, not why a person will actually leave.")

# =============================================================================== 2. batch scoring
with tab_batch:
    st.markdown("### Score a file of customers")
    st.write("Upload a CSV with one customer per row. Column names can be the short ones below or the original dataset's "
             "(`CreditScore`, `Geography`, `NumOfProducts`, …). An optional `customer_id` column is carried through.")
    template = pd.DataFrame([dict(customer_id=1001, **{k: v for k, v in dict(credit_score=600, country="Germany", gender="Male", age=45, tenure=3, balance=120000, products=1, credit_card=1, active=0, salary=55000).items()}),
                             dict(customer_id=1002, credit_score=720, country="France", gender="Female", age=30, tenure=5, balance=60000, products=2, credit_card=1, active=1, salary=90000)])
    st.download_button("Download a template CSV", template.to_csv(index=False), "customers_template.csv", "text/csv")
    st.caption("Tip: customers the model was trained on get optimistic scores, so uploading the original 10,000-row file "
               "flags more people than the Power BI dashboard (which uses out-of-fold scores). Use new customers to see realistic behaviour.")
    up = st.file_uploader("Customer file", type="csv")
    if up is not None:
        try:
            raw = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read that file as CSV: {e}")
            raw = None
        if raw is not None:
            bar = st.progress(0.0, text=f"Scoring {len(raw):,} customers ...")
            scored = safe(client.predict_batch, raw, progress=lambda f: bar.progress(f, text=f"Scoring ... {f:.0%}"))
            bar.empty()
            if scored is not None:
                n, n_flag = len(scored), int(scored["flagged"].sum())
                k1, k2, k3 = st.columns(3)
                k1.metric("Customers scored", f"{n:,}")
                k2.metric("Flagged for a call", f"{n_flag:,}", f"{n_flag / n:.1%} of the file", delta_color="off", delta_arrow="off")
                k3.metric("Average risk score", f"{scored['risk_score'].mean():.2f}")

                st.plotly_chart(ui.score_histogram(scored["risk_score"], threshold), theme=None, config=PLOT_CONFIG)
                st.caption("Customers at or above the call threshold are coral.")

                f1, f2 = st.columns([1, 2])
                only_flagged = f1.toggle("Only flagged customers", value=True)
                bands = f2.multiselect("Risk band", config.RISK_BANDS, default=config.RISK_BANDS)
                view = scored[scored["risk_band"].isin(bands)]
                view = view[view["flagged"]] if only_flagged else view
                view = view.sort_values("risk_score", ascending=False).rename(columns={"top_driver": "biggest push towards leaving"})
                st.dataframe(view, hide_index=True, column_config={
                    "risk_score": st.column_config.ProgressColumn("risk_score", min_value=0.0, max_value=1.0, format="%.3f"),
                    "balance": st.column_config.NumberColumn(format="localized"), "salary": st.column_config.NumberColumn(format="localized"),
                    "risk_percentile": st.column_config.NumberColumn("percentile", format="%.1f")})
                st.caption(f"{len(view):,} rows shown. The 'biggest push' column is most meaningful for flagged customers.")
                st.download_button("Download all results (CSV)", scored.to_csv(index=False), "churn_scores.csv", "text/csv")

# =============================================================================== 3. about the model
with tab_about:
    st.markdown("### About the model")
    if info:
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Model", info["model_name"])
        a2.metric("Call threshold", f"{info['threshold']:.2f}", help=f"Chosen on the validation set by rule: {info['threshold_rule']}")
        a3.metric("Customers", f"{info.get('n_customers') or 0:,}", help="Number of customers in the training data.")
        a4.metric("Churn rate in data", f"{(info.get('churn_rate') or 0):.1%}")

        st.markdown("**Held-out test results** (1,500 customers the model never saw; 95% range from 1,000 bootstrap resamples)")
        ci = info.get("test_ci", {})
        rows = [{"metric": k, "value": v, "95% range": f"{ci[k][0]:.3f} to {ci[k][1]:.3f}" if k in ci else "n/a"} for k, v in info["test_scores"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        if info.get("cv_results"):
            st.markdown("**Candidate models** (5-fold cross-validation on the training data; the best ROC-AUC was tuned and used)")
            st.dataframe(pd.DataFrame(info["cv_results"]).round(4), hide_index=True)

    st.markdown("**How to read the score**")
    st.markdown("- The risk score is a **ranking** score from 0 to 1. The model is trained with class weights, so it is **not** a calibrated probability: "
                "a score of 0.8 does not mean an 80% chance of leaving.\n"
                "- The **percentile** compares the customer with out-of-fold scores of all training customers, so it says how unusual the score is.\n"
                "- The **threshold** turns the score into a yes/no call. It was picked on the validation set to balance precision and recall (F1). "
                "The best threshold for a bank depends on real offer costs, which this dataset does not contain.\n"
                "- The **drivers** are SHAP values: they explain the model's score, not the customer's reasons.")
    st.markdown("**Limits to state honestly**")
    st.markdown("- Trained on one public bank-churn dataset (10,000 customers); behaviour will drift in a real bank, so it needs monitoring and retraining.\n"
                "- Gender and country are model inputs. Before any real use, check fairness and the legal rules for using them.\n"
                "- The cost figures in the Power BI retention analysis (offer cost, value of a saved customer, success rate) are placeholders, not facts.")
    if info.get("versions"):
        st.caption("Trained with " + ", ".join(f"{k} {v}" for k, v in info["versions"].items()) + f" · {info.get('trained_at', '')}")
