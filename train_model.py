"""Train and compare Random Forest and SVM models on MIT-BIH ECG beat segments."""
from __future__ import annotations
import json
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)
from src.data_loader import build_dataset
from src.ecg_pipeline import features_from_segments, CLASS_NAMES

SEED = 42
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def main():
    print("Downloading/loading ECG records and preparing labelled beats...")
    segments, labels, groups, failures = build_dataset()
    print(f"Total segments: {len(labels)}")
    print("Class counts:", pd.Series(labels).value_counts().to_dict())

    # Simple baseline split. See README for the recording-wise evaluation limitation.
    X = features_from_segments(segments)
    X_train, X_test, y_train, y_test, seg_train, seg_test = train_test_split(
        X, labels, segments, test_size=0.20, random_state=SEED, stratify=labels
    )

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=250, class_weight="balanced", random_state=SEED,
            min_samples_leaf=2, n_jobs=-1
        ),
        "SVM": Pipeline([
            ("scale", StandardScaler()),
            ("classifier", SVC(class_weight="balanced", kernel="rbf", C=2.0, gamma="scale"))
        ]),
    }

    results = {}
    best_name, best_model, best_f1 = None, None, -1
    labels_order = ["N", "S", "V"]
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        p, r, f1, support = precision_recall_fscore_support(
            y_test, pred, labels=labels_order, average=None, zero_division=0
        )
        macro_f1 = float(np.mean(f1))
        results[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": macro_f1,
            "classification_report": classification_report(
                y_test, pred, labels=labels_order, target_names=[CLASS_NAMES[x] for x in labels_order],
                output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_test, pred, labels=labels_order).tolist(),
        }
        print(f"{name}: accuracy={results[name]['accuracy']:.4f}, macro-F1={macro_f1:.4f}")
        if macro_f1 > best_f1:
            best_name, best_model, best_f1 = name, model, macro_f1

    joblib.dump(best_model, os.path.join(ARTIFACT_DIR, "ecg_model.joblib"))
    np.savez_compressed(os.path.join(ARTIFACT_DIR, "sample_beats.npz"),
                        segments=seg_test[:100], labels=y_test[:100])
    with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "class_order": labels_order,
                   "class_names": CLASS_NAMES, "results": results,
                   "class_counts": pd.Series(labels).value_counts().to_dict(),
                   "records": sorted(set(groups.tolist())),
                   "skipped_records": failures,
                   "evaluation_note": "Stratified beat-level split; likely optimistic due to beats from same records appearing in train and test."},
                  f, indent=2)

    # Confusion matrix for the selected model
    best_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, best_pred, labels=labels_order)
    disp = ConfusionMatrixDisplay(cm, display_labels=[CLASS_NAMES[x] for x in labels_order])
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"Confusion Matrix — {best_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACT_DIR, "confusion_matrix.png"), dpi=160)
    plt.close(fig)

    # Class distribution plot
    counts = pd.Series(labels).value_counts().reindex(labels_order, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_xticklabels([CLASS_NAMES[x] for x in counts.index], rotation=15, ha="right")
    ax.set_ylabel("Number of segments")
    ax.set_title("ECG Heartbeat Class Distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACT_DIR, "class_distribution.png"), dpi=160)
    plt.close(fig)

    print(f"\nBest baseline model by macro-F1: {best_name}")
    print(f"Saved model and outputs to ./{ARTIFACT_DIR}/")
    print("IMPORTANT: report the beat-level split limitation; do not interpret this as clinical performance.")

if __name__ == "__main__":
    main()
