"""Download and load a small set of MIT-BIH records with WFDB."""
from __future__ import annotations
import os
from pathlib import Path
import wfdb
from .ecg_pipeline import bandpass_filter, extract_beat_segments

# A small starter subset. Expand after the pipeline runs successfully.
RECORDS = ["100", "101", "103", "105", "106", "108", "109", "111", "112", "113",
           "115", "116", "117", "118", "119", "121", "122", "123", "124"]

def load_record_beats(record_name, cache_dir="data/mitdb"):
    """Load one record, filter channel 0, and return segments, labels, metadata."""
    os.makedirs(cache_dir, exist_ok=True)
    try:
        record = wfdb.rdrecord(record_name, pn_dir="mitdb")
        annotation = wfdb.rdann(record_name, "atr", pn_dir="mitdb")
    except Exception as exc:
        raise RuntimeError(
            f"Could not download/read MIT-BIH record {record_name}. "
            "Check your internet connection and the WFDB/PhysioNet service."
        ) from exc

    fs = float(record.fs)
    signal = record.p_signal[:, 0]
    filtered = bandpass_filter(signal, fs)
    segments, labels = extract_beat_segments(
        filtered, annotation.sample, annotation.symbol, fs
    )
    return segments, labels, {"record": record_name, "fs": fs, "signal": filtered,
                              "annotation": annotation, "units": record.units[0]}

def build_dataset(records=None):
    """Collect heartbeat segments from records; returns X segments, y labels, record IDs."""
    records = records or RECORDS
    all_segments, all_labels, all_groups = [], [], []
    failures = []
    for rec in records:
        try:
            segments, labels, _ = load_record_beats(rec)
            if len(labels):
                all_segments.append(segments)
                all_labels.append(labels)
                all_groups.extend([rec] * len(labels))
                print(f"Record {rec}: extracted {len(labels)} labelled beats")
        except Exception as exc:
            failures.append((rec, str(exc)))
            print(f"WARNING: skipping record {rec}: {exc}")
    if not all_segments:
        raise RuntimeError("No ECG beats were loaded. Check internet access and dataset availability.")
    import numpy as np
    return np.concatenate(all_segments), np.concatenate(all_labels), np.asarray(all_groups), failures
