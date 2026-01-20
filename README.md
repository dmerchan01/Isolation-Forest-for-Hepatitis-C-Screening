# Isolation Forest for Hepatitis C Screening

This project implements an **unsupervised anomaly detection pipeline** using **Isolation Forest** to identify abnormal liver profiles in patients from the **Hepatitis C Virus (HCV) dataset** published by the UCI Machine Learning Repository.

The model is trained **exclusively on healthy blood donors**, defining a baseline of normality. Any patient deviating significantly from this baseline is flagged as **suspicious**, enabling early screening and risk assessment.

---

## Project Overview

The system focuses on **anomaly detection rather than direct disease classification**.  
It answers the question:

> _Does this patient's biochemical profile deviate from what is considered normal?_

Key ideas:

- Normality is learned from **Blood Donors only**
- All patients are evaluated against this learned baseline
- Abnormal patterns are detected without using disease labels during training
- A data-driven strategy is used to **automatically select the contamination threshold**

---

## Dataset

- **Source:** UCI Machine Learning Repository
- **Dataset:** Hepatitis C Virus (HCV) dataset
- **Instances:** 615 patients
- **Features:** Clinical and biochemical blood markers
- **Classes (used only for evaluation):**
  - Blood Donor (normal)
  - Suspect Blood Donor
  - Hepatitis
  - Fibrosis
  - Cirrhosis

The dataset is accessed programmatically using the `ucimlrepo` library.

---

## Features Used

The following laboratory and demographic features are used:

- **Biochemical markers**
  - ALB, ALP, ALT, AST
  - BIL, CHE, CHOL
  - CREA, CGT, PROT

Missing values are handled using **median imputation**, and features are **standardized** using statistics computed **only from healthy donors**.

---

## Methodology

### 1. Normality Definition

Only patients labeled as **Blood Donors** are used to train the Isolation Forest.  
This establishes a clean baseline representing healthy liver profiles.

### 2. Isolation Forest Training

- Model: `IsolationForest` (scikit-learn)
- Trained on scaled healthy samples
- Evaluated on the entire population

### 3. Automatic Contamination Selection

A grid of contamination values is evaluated.  
The final contamination value is **automatically selected** using the following screening-oriented criterion:

> **Maximize recall (detect as many non-donors as possible) while maintaining a minimum precision threshold.**

This balances sensitivity and false alarms in a clinically realistic way.

### 4. Evaluation

Although the model is unsupervised, ground-truth labels are used **only for evaluation**, computing:

- Precision
- Recall
- Confusion tables by clinical category

---

## Project Structure

```
Isolation-Forest-Anomaly-Detection/
│
├── test.py                 # Main pipeline (data loading, training, evaluation)
├── README.md               # Project documentation
├── .gitignore              # Git ignore rules
├── venv/                   # Virtual environment (ignored)
```

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/Isolation-Forest-Anomaly-Detection.git
cd Isolation-Forest-Anomaly-Detection
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```

3. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn ucimlrepo
```

4. Run the analysis:

```bash
python test.py
```

---

## Results Interpretation

- **Anomaly score:** Lower values indicate stronger deviation from normality
- **Anomaly label:**
  - `1` → Normal
  - `-1` → Suspicious

Patients with advanced conditions (e.g., cirrhosis) tend to receive the lowest scores, while early-stage conditions may partially overlap with healthy donors.

---

## Limitations

- Isolation Forest does **not classify diseases**
- Some pathological cases may appear normal if their biochemical profile is mild
- Ground-truth labels are not used during training

This approach is best suited for **screening and risk prioritization**, not diagnosis.

---

## Future Work

- Integrate a **Random Forest classifier** for disease classification after screening
- Combine anomaly score with supervised predictions
- Feature importance analysis for explainability
- Threshold calibration using clinical cost functions

---

## License

This project is intended for **educational and research purposes**.
