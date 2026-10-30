"""Preprocessing, segmentation, and feature extraction for ECG heartbeat classification."""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

# Simplified MIT-BIH annotation groups. This is a teaching mapping, not a clinical taxonomy.
CLASS_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
}
CLASS_NAMES = {"N": "Normal", "S": "Supraventricular ectopic", "V": "Ventricular ectopic"}

def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=3):
    """Apply a Butterworth bandpass filter; safely returns the original if too short."""
    x = np.asarray(signal, dtype=float)
    if len(x) < 20 or fs <= 2 * lowcut:
        return x.copy()
    nyquist = fs / 2.0
    high = min(highcut, nyquist * 0.95)
    low = min(lowcut, high * 0.5)
    if low <= 0 or high <= low:
        return x.copy()
    b, a = butter(order, [low / nyquist, high / nyquist], btype="band")
    try:
        return filtfilt(b, a, x)
    except ValueError:
        return x.copy()

def extract_beat_segments(signal, annotation_samples, annotation_symbols, fs,
                          before_seconds=0.25, after_seconds=0.40):
    """Extract fixed-size waveform windows around supported annotated beats."""
    x = np.asarray(signal, dtype=float)
    before = int(before_seconds * fs)
    after = int(after_seconds * fs)
    rows, labels = [], []
    for sample, symbol in zip(annotation_samples, annotation_symbols):
        if symbol not in CLASS_MAP:
            continue
        start, end = int(sample) - before, int(sample) + after
        if start < 0 or end > len(x):
            continue
        segment = x[start:end]
        if len(segment) != before + after:
            continue
        # Normalise each beat to reduce scale variation; avoid division by zero.
        segment = segment - np.mean(segment)
        std = np.std(segment)
        if std > 1e-8:
            segment = segment / std
        rows.append(segment.astype(np.float32))
        labels.append(CLASS_MAP[symbol])
    return np.asarray(rows), np.asarray(labels)

def features_from_segments(segments, fs=360):
    """Compute simple, explainable numerical features for each fixed-length beat."""
    if len(segments) == 0:
        return np.empty((0, 10), dtype=float)
    features = []
    for seg in np.asarray(segments):
        s = np.asarray(seg, dtype=float)
        # Relative position and amplitude statistics; simple baseline features.
        peak = int(np.argmax(np.abs(s)))
        energy = float(np.mean(s ** 2))
        diff = np.diff(s)
        features.append([
            float(np.mean(s)),
            float(np.std(s)),
            float(np.min(s)),
            float(np.max(s)),
            float(np.median(s)),
            float(np.percentile(s, 25)),
            float(np.percentile(s, 75)),
            float(peak / max(1, len(s))),
            energy,
            float(np.mean(np.abs(diff))) if len(diff) else 0.0,
        ])
    return np.asarray(features, dtype=float)

def simple_rr_features(annotation_samples, annotation_symbols, fs):
    """Placeholder helper for future RR interval feature extension."""
    samples = np.asarray(annotation_samples, dtype=int)
    if len(samples) < 2:
        return np.array([])
    return np.diff(samples) / float(fs)
