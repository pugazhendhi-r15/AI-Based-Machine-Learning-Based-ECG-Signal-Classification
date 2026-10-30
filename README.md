# AI-Based ECG Signal Classification Using Machine Learning

A beginner-friendly educational prototype that reads ECG recordings from the MIT-BIH Arrhythmia Database, filters the signal, extracts labelled heartbeat segments, trains Random Forest and SVM classifiers, evaluates them, and displays predictions in a Streamlit dashboard.

> **Important:** This is an educational research prototype, not a medical device. It must not be used for diagnosis or treatment decisions. Predictions may be wrong.

## Features
- Downloads public MIT-BIH Arrhythmia Database records through WFDB.
- Filters ECG signals with a Butterworth bandpass filter.
- Uses reference heartbeat annotations to create labelled segments.
- Classifies three simplified heartbeat groups:
  - `N`: normal and selected normal-conduction beats
  - `S`: supraventricular ectopic beats
  - `V`: ventricular ectopic beats
- Extracts basic waveform/statistical features.
- Compares Random Forest and SVM.
- Saves the trained model, metrics, confusion matrix, and sample segments.
- Provides a Streamlit dashboard.

## Important evaluation note
The included starter implementation uses a stratified beat-level split for an easy first run. Since beats from the same recording can be similar, this can produce optimistic metrics. For a stronger academic evaluation, change the split to be recording-wise/patient-wise and report the limitation in the report. Never present demo-mode outputs as real model results.

## Setup (Google Colab or local Python)

### 1. Install packages
```bash
pip install -r requirements.txt
```

### 2. Train the models
```bash
python train_model.py
```
The script downloads the selected MIT-BIH records on first run, prepares the data, trains two models, compares them, and saves the best model in `artifacts/`.

### 3. Launch the dashboard
```bash
streamlit run app.py
```
Open the local URL shown by Streamlit.

## Google Colab
1. Upload/extract this project ZIP in Colab.
2. Run `!pip install -r requirements.txt`.
3. Run `!python train_model.py`.
4. Run `!streamlit run app.py` (use a Colab-compatible tunnel if you need to open the web app; alternatively use the notebook plots and predictions).

## Project structure
```text
ECG_AI_Classification_Project/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
├── PRESENTATION_OUTLINE.md
├── src/
│   ├── __init__.py
│   ├── ecg_pipeline.py
│   └── data_loader.py
├── data/                 # downloaded files/cache created when training
└── artifacts/            # model, metrics, plots created after training
```

## Suggested final submission
- Source code and notebook/output screenshots
- Dataset source and heartbeat class mapping
- Preprocessing and EDA plots
- Model comparison table
- Confusion matrix and class-wise precision/recall/F1
- Dashboard screenshots
- Report and presentation
- A clear statement of limitations and educational-only use

## Dataset
MIT-BIH Arrhythmia Database: https://physionet.org/content/mitdb/1.0.0/
WFDB Python documentation: https://wfdb.io/software/python.html
