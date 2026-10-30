# Project Report: AI-Based ECG Signal Classification Using Machine Learning

## Abstract
This project demonstrates a machine-learning pipeline for classifying selected heartbeat patterns from ECG recordings. ECG records and reference annotations are obtained from the MIT-BIH Arrhythmia Database. The signal is filtered, fixed-length heartbeat windows are extracted, and simple numerical features are calculated. Random Forest and Support Vector Machine (SVM) classifiers are trained and compared. A Streamlit dashboard presents the waveform, example predictions, and evaluation outputs. The system is intended for learning and research only; it is not a clinical diagnostic tool.

## 1. Introduction
An electrocardiogram (ECG) records the electrical activity of the heart. ECG waveforms contain repeated patterns associated with cardiac electrical activity. Machine learning can be used to learn statistical patterns from labelled ECG examples and predict their assigned categories. This project combines digital signal processing and supervised machine learning in a reproducible prototype.

## 2. Problem Statement
To develop a system that processes labelled ECG recordings, extracts heartbeat segments, trains machine-learning classifiers to predict selected heartbeat groups, evaluates performance on held-out examples, and displays results through a dashboard.

## 3. Objectives
- Obtain a public labelled ECG dataset.
- Visualize and preprocess ECG signals.
- Extract heartbeat windows using reference annotations.
- Explore the distribution of selected classes.
- Train and compare Random Forest and SVM models.
- Evaluate using accuracy, precision, recall, F1-score, and confusion matrix.
- Build an interactive dashboard.
- Document limitations and intended educational use.

## 4. Dataset
The prototype uses the MIT-BIH Arrhythmia Database available through PhysioNet. The dataset contains annotated ECG recordings. The starter code uses a subset of record identifiers to reduce setup time. The selected record subset and any records that fail to load are reported in `artifacts/metrics.json` after training.

Simplified mapping used by this project:
- N group: N, L, R, e, j
- S group: A, a, J, S
- V group: V, E

This grouping is an educational simplification. It should be reviewed and justified against the official dataset documentation before academic submission. Other annotation symbols are excluded from this baseline.

## 5. Methodology
### 5.1 Data loading
WFDB reads ECG signals and their expert annotations from PhysioNet.

### 5.2 Preprocessing
A Butterworth bandpass filter is applied to reduce unwanted frequency components. Filter parameters should be reviewed for the chosen recording and signal-analysis objective.

### 5.3 Segmentation
For each supported annotation, the code extracts a fixed window around the annotated sample. Windows too close to the beginning or end of a record are skipped. Each segment is centred around an annotation sample, not a separately detected R peak.

### 5.4 Feature extraction
The baseline uses mean, standard deviation, minimum, maximum, median, quartiles, absolute-peak position, signal energy, and mean absolute first difference. These features are simple baseline descriptors, not a comprehensive clinical feature set.

### 5.5 Model training
Random Forest and SVM are trained using the extracted feature matrix. The Random Forest uses class weights to help account for class imbalance. The SVM pipeline scales features before fitting.

### 5.6 Evaluation
The code reports accuracy, per-class precision, recall, F1-score, macro-F1, and a confusion matrix. Macro-F1 gives each class equal weight, helping reveal performance on less frequent classes.

### 5.7 Dashboard
The Streamlit app loads the saved model, lets the user select a record, plots its waveform, shows a selected heartbeat segment, and displays the predicted group and evaluation results.

## 6. System Architecture
ECG records + annotations → bandpass filtering → heartbeat segmentation → feature extraction → train/test split → Random Forest and SVM → metric comparison → saved model → Streamlit dashboard.

## 7. Results
Run `python train_model.py` to generate actual results. Insert the measured metrics from `artifacts/metrics.json` and include screenshots of the generated plots and dashboard. Do not invent or copy example metrics: results depend on the records successfully loaded and the installed software versions.

## 8. Limitations
- This is a simplified educational classifier, not a medical device.
- The starter implementation uses a stratified beat-level split. Beats from the same record may appear in both training and test sets, making performance estimates optimistic.
- A recording-wise or patient-wise split should be implemented for stronger generalization testing.
- The class mapping excludes several annotation types and merges some symbols.
- Basic statistical features may not capture all useful ECG morphology.
- Performance depends on preprocessing, labels, dataset selection, and class imbalance.
- The model has not been clinically validated and should not be used for patient care.

## 9. Future Enhancements
- Use a record-wise split and report per-record performance.
- Add RR interval and waveform morphology features.
- Compare with a 1D convolutional neural network (CNN).
- Add downloadable prediction reports.
- Add confidence scores with careful calibration.
- Test on an independent ECG dataset and assess robustness.

## 10. Conclusion
The project demonstrates an end-to-end educational pipeline that combines ECG signal processing, labelled beat segmentation, feature extraction, machine learning, evaluation, and dashboard integration. Its main value is to teach the workflow and provide a foundation for further research, while clearly acknowledging that it is not suitable for clinical diagnosis.

## References
1. PhysioNet, MIT-BIH Arrhythmia Database: https://physionet.org/content/mitdb/1.0.0/
2. WFDB Python documentation: https://wfdb.io/software/python.html
3. Scikit-learn documentation: https://scikit-learn.org/stable/
4. SciPy documentation: https://docs.scipy.org/doc/scipy/
