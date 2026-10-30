"""Streamlit dashboard for the educational ECG classifier."""
from __future__ import annotations
import json
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from src.ecg_pipeline import bandpass_filter, features_from_segments, CLASS_NAMES
from src.data_loader import load_record_beats, RECORDS

st.set_page_config(page_title="ECG AI Classifier", page_icon="🫀", layout="wide")
st.title("🫀 AI-Based ECG Signal Classification")
st.caption("Educational research prototype — not for clinical diagnosis or treatment decisions.")

model_path = "artifacts/ecg_model.joblib"
metrics_path = "artifacts/metrics.json"

@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

@st.cache_data
def load_metrics():
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
metrics = load_metrics()

if model is None:
    st.warning("The trained model was not found. Run `python train_model.py` first, then refresh this page.")
    st.markdown("### Quick start")
    st.code("pip install -r requirements.txt\npython train_model.py\nstreamlit run app.py", language="bash")
    st.stop()

st.success(f"Model loaded. Selected baseline: {metrics.get('best_model', 'trained classifier') if metrics else 'trained classifier'}")
record_name = st.selectbox("Choose an ECG record", RECORDS, index=0)
if st.button("Load record and classify annotated beats", type="primary"):
    with st.spinner("Loading ECG record and preparing beat segments..."):
        try:
            segments, labels, meta = load_record_beats(record_name)
            X = features_from_segments(segments, meta["fs"])
            preds = model.predict(X)
            st.session_state["ecg_result"] = {
                "segments": segments, "labels": labels, "preds": preds,
                "meta": {"record": record_name, "fs": meta["fs"], "signal": meta["signal"]}
            }
        except Exception as exc:
            st.error(f"Could not load this record: {exc}")

result = st.session_state.get("ecg_result")
if result:
    meta = result["meta"]
    segments, labels, preds = result["segments"], result["labels"], result["preds"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Record", meta["record"])
    c2.metric("Annotated segments", len(segments))
    c3.metric("Sampling frequency", f"{meta['fs']:.0f} Hz")

    st.subheader("ECG waveform (first 2,000 samples)")
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(meta["signal"][:2000], linewidth=0.8)
    ax.set_xlabel("Sample number")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Example heartbeat and prediction")
    if len(segments):
        idx = st.slider("Choose a beat segment", 0, len(segments)-1, 0)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(segments[idx], linewidth=1.2)
        ax.set_title(f"True dataset label: {CLASS_NAMES.get(labels[idx], labels[idx])}")
        ax.set_xlabel("Relative sample")
        ax.set_ylabel("Normalised amplitude")
        ax.grid(alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)
        pred_name = CLASS_NAMES.get(str(preds[idx]), str(preds[idx]))
        st.info(f"Model prediction: **{pred_name}**")
        if str(preds[idx]) == str(labels[idx]):
            st.write("This prediction matches the dataset annotation for this example.")
        else:
            st.write("This prediction differs from the dataset annotation. Models can make errors.")
    st.caption("Class labels are simplified research groups derived from MIT-BIH annotations. They are not medical diagnoses.")

if metrics:
    st.divider()
    st.subheader("Model evaluation")
    st.warning(metrics.get("evaluation_note", "Review the evaluation method before interpreting these metrics."))
    for name, vals in metrics["results"].items():
        st.markdown(f"**{name}** — Accuracy: {vals['accuracy']:.3f} | Macro-F1: {vals['macro_f1']:.3f}")
        report = vals["classification_report"]
        rows = []
        for cls in ["Normal", "Supraventricular ectopic", "Ventricular ectopic"]:
            if cls in report:
                rows.append({"Class": cls, "Precision": report[cls]["precision"],
                             "Recall": report[cls]["recall"], "F1-score": report[cls]["f1-score"],
                             "Support": int(report[cls]["support"])})
        st.dataframe(rows, use_container_width=True)
    cm_path = "artifacts/confusion_matrix.png"
    if os.path.exists(cm_path):
        st.image(cm_path, caption="Confusion matrix for the selected best model")
    class_path = "artifacts/class_distribution.png"
    if os.path.exists(class_path):
        st.image(class_path, caption="Training dataset class distribution")
