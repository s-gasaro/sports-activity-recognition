# Recognising Physical Activities from Wearable Sensor Data: Traditional ML vs Deep Learning

This project compares traditional machine learning (scikit-learn) and deep learning (TensorFlow/Keras) on classifying 12 physical activities from body-worn sensors. All evaluation is on **people the models have never seen**.

- **Report:** [link]
- **Video:** [link]
- **Google Drive (raw dataset + saved models):** [link]

## Problem
Wearable sensors can track what someone is doing (walking, running, cycling, rope jumping), which supports fitness and sports monitoring. A model is only useful if it works for new users. People wear sensors differently and move differently, so a model trained on some people can fail on others. This project asks how well traditional ML and deep learning recognise activities for an unseen person, and what limits each approach.

## Dataset
**PAMAP2 Physical Activity Monitoring** (UCI ML Repository): Reiss, A. (2012). https://doi.org/10.24432/C5NW2H (CC BY 4.0)

- 3,850,505 raw rows in total; the 9 protocol files used here contain 2,872,533 rows
- 9 subjects; 3 IMUs (wrist, chest, ankle) at 100 Hz, plus a heart-rate monitor at ~9 Hz
- 12 protocol activities: lying, sitting, standing, walking, running, cycling, Nordic walking, ascending/descending stairs, vacuum cleaning, ironing, rope jumping
- The raw data is **not** in this repo. The notebook downloads it directly from UCI, and a zipped copy is in the Google Drive folder.

## Method
1. **Cleaning:**
   - Heart rate is forward-filled (it's sampled less often than the other sensors), and IMU dropouts are interpolated within each subject.
   - Saturated readings and spikes were checked and kept: they are rare and physically plausible.
   - Transient activity and subject 109 (only 1.1 minutes of data) are removed.
   - Invalid orientation columns, the saturating ±6g accelerometer, and temperature are dropped.
2. **Windowing:** 2.56 s windows (256 readings) with 50% overlap, never crossing activity or subject boundaries. This gives 14,975 windows.
3. **Subject-wise split:** train = subjects 102–107, validation = 108, test = 101. A random split would put near-identical overlapping windows in both train and test and inflate accuracy.
4. **Traditional ML:**
   - 216 time- and frequency-domain features per window
   - Logistic Regression, Random Forest, Histogram Gradient Boosting and SVM
   - Ablation by sensor group, then the Random Forest tuned with RandomizedSearchCV and GroupKFold by subject
5. **Deep learning:**
   - MLP, 1D CNN (Sequential API) and CNN-LSTM (Functional API)
   - Data fed through `tf.data`, with class weights, early stopping and augmentation
6. **Robustness check:** leave-one-subject-out evaluation across all 8 subjects.
7. **Tracking:** every run is logged to TensorBoard (`logs/`) and `results/experiments.csv`. The full table with observations is in `results/experiment_table.csv`.

## Results

**Experiments (validation subject 108, macro F1).** The full table, with parameters and an observation for each run, is in `results/experiment_table.csv`.

| ID | Model | Change | Macro F1 |
|---|---|---|---|
| ML-01 | Logistic Regression | baseline, all features | 0.541 |
| ML-02 | Random Forest | baseline, all features | 0.763 |
| ML-03 | Hist. Gradient Boosting | baseline, all features | 0.685 |
| ML-04 | SVM (RBF) | baseline, all features | 0.521 |
| ML-05 | Random Forest | no magnetometer | 0.812 |
| ML-06 | Random Forest | no heart rate | 0.720 |
| ML-07 | Random Forest | no magnetometer, no heart rate | 0.798 |
| ML-08 | Logistic Regression | no magnetometer, no heart rate | 0.564 |
| ML-09 | SVM (RBF) | no magnetometer, no heart rate | 0.516 |
| ML-10 | Hist. Gradient Boosting | no magnetometer | 0.819 |
| ML-11 | Random Forest | tuned (GroupKFold CV) | **0.897** |
| ML-12 | Random Forest | ML-11 settings, orientation-free features only (control) | 0.822 |
| DL-01 | MLP | engineered features | 0.682 |
| DL-02 | 1D CNN | raw 19 channels | 0.352 |
| DL-03 | 1D CNN | raw, no heart rate | 0.316 |
| DL-04 | 1D CNN | raw + augmentation | 0.379 |
| DL-05 | 1D CNN | raw + aug + BatchNorm + L2 + dropout 0.5 | 0.410 |
| DL-06 | 1D CNN | orientation-free magnitudes + heart rate | 0.862 |
| DL-07 | 1D CNN | raw + magnitudes + axis-flip augmentation | 0.854 |
| DL-08 | MLP | dropout 0.5, lr 3e-4 | 0.740 |
| DL-09 | CNN-LSTM | orientation-free input | 0.845 |
| DL-10 | 1D CNN | DL-06 with lr 3e-4 | 0.839 |
| DL-11 | 1D CNN | DL-06 with kernel 3 | **0.874** |
| DL-12 | 1D CNN | DL-06 with kernel 9 | 0.865 |
| DL-13 | 1D CNN | DL-06 with smaller network | 0.842 |
| DL-14 | 1D CNN | DL-06 with dropout 0.5 | 0.870 |

**Final test (subject 101, never used for training or selection)**

| Model | Accuracy | Macro F1 | Macro ROC-AUC |
|---|---|---|---|
| Random Forest (ML-11) | 0.855 | **0.864** | 0.985 |
| 1D CNN (DL-11) | 0.784 | 0.774 | 0.975 |

**Leave-one-subject-out (8 subjects, mean ± std macro F1)**

| Model | Macro F1 |
|---|---|
| Random Forest (ML-11) | **0.887 ± 0.037** (best on 8 of 8 subjects vs the CNN) |
| Random Forest, orientation-free features (ML-12) | 0.803 ± 0.032 |
| 1D CNN (DL-11) | 0.794 ± 0.028 |

## Key findings
- **The Random Forest beat the CNN on every subject, but mainly because of what its features contain.** Given only the orientation-free information the CNN sees, the Random Forest (ML-12) scores about the same as the CNN (0.803 vs 0.794). The advantage comes from posture information (the direction of gravity in per-axis features), not from the model type.
- **Input representation mattered more than model choice.** CNNs on raw x/y/z axes failed on new subjects (macro F1 0.32–0.41), and regularisation did not help, because sensor placement differed between people. Orientation-free inputs raised the CNN to about 0.86. Its hyperparameter changes then moved it by 0.02 or less.
- **That fix had a cost.** Without the direction of gravity, the CNN cannot separate sitting from standing: 92% of sitting windows were predicted as standing on the test subject.
- **Removing magnetometer features improved the Random Forest** (0.763 → 0.812). These features likely reflect the surroundings rather than movement.
- **High variance across people.** The Random Forest learning curve shows training ≈ 1.0 and cross-validation ≈ 0.85. Data from more people would help more than more data from the same people.

## Repository structure
```
├── pamap2_ml_vs_dl.ipynb   # full pipeline, runs top to bottom
├── requirements.txt
├── results/                # experiment logs and tables, final metrics, LOSO results
├── figures/                # all plots used in the report
└── logs/                   # TensorBoard logs for every experiment
```
Saved models (`best_random_forest.joblib`, `best_dl_model.h5` / `.keras`) and their normalisation stats are in the Google Drive folder.

## How to run
1. Open the notebook in Kaggle or Colab with internet access on. A GPU is recommended for the deep learning section.
2. If you're running locally, install the dependencies first: `pip install -r requirements.txt`.
3. Run all cells. The dataset (~650 MB) downloads automatically. A full run takes about 40 minutes on a Kaggle T4 GPU.

To view the experiment logs:
```
tensorboard --logdir logs
```
The random seed is 42. Random Forest results are fully reproducible. GPU training is not fully deterministic, so deep learning results can vary by about ±0.02 macro F1 between runs.
