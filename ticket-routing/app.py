"""Streamlit app: IT support ticket routing (Infosys Topaz-style helpdesk automation)."""
import io
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from preprocess import clean_text  # noqa: E402

st.set_page_config(page_title="Ticket Router", page_icon="🎫", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "ticket_classifier.joblib")


model = load_model()
classes = list(model.classes_)

EXAMPLES = {
    "Laptop won't start": "my laptop screen is black and the keyboard does not respond please send an engineer",
    "Login / access": "unable to login to my account please reset my password and grant access",
    "Admin rights": "need administrator rights to install software on my machine",
    "Purchase": "request purchase order for new equipment please approve the purchase",
    "Leave / HR": "please update my leave balance and payslip details in the hr system",
    "Storage full": "my mailbox is full and I need more disk space on the server",
}

st.title("🎫 IT Support Ticket Router")
st.caption("Classifies a support ticket into a team queue. Low-confidence tickets are sent to a human for triage.")

threshold = st.sidebar.slider("Auto-route confidence threshold", 0.30, 0.90, 0.60, 0.05,
                              help="Below this confidence the ticket goes to manual triage.")
st.sidebar.markdown("**Queues:** " + ", ".join(classes))
st.sidebar.markdown("Model: TF-IDF (1-2 grams) + Logistic Regression")

tab_single, tab_batch = st.tabs(["Single ticket", "Batch (CSV)"])

with tab_single:
    choice = st.selectbox("Try an example", ["(write your own)"] + list(EXAMPLES))
    default = EXAMPLES.get(choice, "")
    text = st.text_area("Ticket text", value=default, height=150,
                        placeholder="Describe the issue...")
    if st.button("Route ticket", type="primary"):
        cleaned = clean_text(text)
        if not cleaned:
            st.warning("Please enter some ticket text.")
        else:
            proba = model.predict_proba([cleaned])[0]
            order = proba.argsort()[::-1]
            top, conf = classes[order[0]], proba[order[0]]
            c1, c2 = st.columns(2)
            c1.metric("Predicted queue", top)
            c2.metric("Confidence", f"{conf:.0%}")
            if conf >= threshold:
                st.success(f"Auto-routed to **{top}**.")
            else:
                st.warning("Confidence below threshold: send to **manual triage**.")
            st.bar_chart(pd.Series(proba, index=classes).sort_values(ascending=False))

with tab_batch:
    st.write("Upload a CSV with a text column. You will get predictions, confidence and a routing decision.")
    file = st.file_uploader("CSV file", type="csv")
    if file is not None:
        data = pd.read_csv(file)
        col = st.selectbox("Text column", data.columns,
                           index=list(data.columns).index("Document") if "Document" in data.columns else 0)
        if st.button("Classify all"):
            cleaned = data[col].fillna("").map(clean_text)
            proba = model.predict_proba(cleaned)
            out = data.copy()
            out["predicted_queue"] = [classes[i] for i in proba.argmax(axis=1)]
            out["confidence"] = proba.max(axis=1).round(3)
            out["decision"] = ["auto-route" if c >= threshold else "manual triage"
                               for c in out["confidence"]]
            st.dataframe(out.head(200), use_container_width=True)
            a, b = st.columns(2)
            a.bar_chart(out["predicted_queue"].value_counts())
            b.metric("Auto-routed", f"{(out['decision'] == 'auto-route').mean():.0%}")
            buf = io.StringIO()
            out.to_csv(buf, index=False)
            st.download_button("Download results", buf.getvalue(), "routed_tickets.csv", "text/csv")
