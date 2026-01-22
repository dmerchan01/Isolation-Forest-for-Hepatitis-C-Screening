########### Imports ########### 

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


########### Dataset ########### 

# Fetch dataset from UCI repository
hcv = fetch_ucirepo(id=571)


########### Data Cleaning ########### 

# Convert to DataFrame
X_df = hcv.data.features.copy()     # features
y_df = hcv.data.targets.copy()      # target (Category)

# Build full data with Category
data = X_df.copy()
if "Category" in y_df.columns:
    data["Category"] = y_df["Category"].astype(str)
else:
    data["Category"] = y_df.iloc[:, 0].astype(str)

# Select features
features = data[
    ['ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE',
     'CHOL', 'CREA', 'CGT', 'PROT']
].copy()

# Add demographic variables
features['Sex'] = data['Sex'].map({'m': 0, 'f': 1})
features['Age'] = data['Age']

# Handle missing values
features = features.fillna(features.median(numeric_only=True))

# Check data
print("Features preview:")
print(features.head())

print("\nCategory counts:")
print(data["Category"].value_counts())


########### Define the training parameters ########### 

# Parameters
n_estimators = 200      # Number of trees
sample_size = 256       # Number of samples used to train each tree (max_samples)

# Testing multiple values
contamination_grid = [0.01, 0.03, 0.05, 0.08, 0.10]


########### Train the Isolation Forest - ONLY with Blood Donors (NORMAL) ###########

# Obtain normal instances (Blood Donors)
normal_label = "0=Blood Donor"
normal_mask = data["Category"] == normal_label

print(f"\nNormal (Blood Donor) rows: {normal_mask.sum()}")

# Isolate normal instances
X_normal = features.loc[normal_mask].copy()

# Scale features only for normal instances
scaler = StandardScaler()
X_normal_scaled = scaler.fit_transform(X_normal)

# Convert all features using the same scaler
X_all_scaled = scaler.transform(features)

# Create a "ground truth" binary label for evaluation:
# 0 = Blood Donor (normal), 1 = non-donor (anything else)
data["is_non_donor"] = (data["Category"] != normal_label).astype(int)


########### Contamination Sweep (Precision/Recall) ###########

results = []

for contamination in contamination_grid:

    # Train Isolation Forest
    iso_tmp = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=sample_size,
        random_state=42
    )
    iso_tmp.fit(X_normal_scaled)

    # Predict on all patients
    anomaly = iso_tmp.predict(X_all_scaled)  # 1 normal, -1 anomaly
    anomaly_score = iso_tmp.decision_function(X_all_scaled)

    # Convert predictions to binary for evaluation
    pred_anomaly = (anomaly == -1).astype(int)

    # Compute precision/recall against "is_non_donor"
    tp = ((data["is_non_donor"] == 1) & (pred_anomaly == 1)).sum()
    fp = ((data["is_non_donor"] == 0) & (pred_anomaly == 1)).sum()
    fn = ((data["is_non_donor"] == 1) & (pred_anomaly == 0)).sum()

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)

    results.append({
        "contamination": contamination,
        "n_anomalies": int((anomaly == -1).sum()),
        "tp_non_donor": int(tp),
        "fp_donor": int(fp),
        "fn_non_donor": int(fn),
        "precision": precision,
        "recall": recall
    })

# Show sweep results
results_df = pd.DataFrame(results)
print("\n=== Contamination Sweep Results (detecting non-donors as anomalies) ===")
print(results_df.to_string(index=False))


########### Auto-select contamination (For not loosing sick patients) ###########
# Goal: maximize recall (catch as many non-donors as possible)
# while keeping precision above a minimum acceptable level.

min_precision = 0.60     # adjust based on wanted precission, higher (e.g., 0.70) or looser (e.g., 0.50)

valid = results_df[results_df["precision"] >= min_precision].copy()

# If nothing meets the precision constraint, fall back to maximum recall overall
if len(valid) == 0:
    best_row = results_df.sort_values("recall", ascending=False).iloc[0]
    print(f"\n[Warning] No contamination met precision >= {min_precision}. Falling back to max recall overall.")
# Choose the row with highest recall
else:
    # Tie-breakers: higher precision, then fewer anomalies
    best_row = (
        valid.sort_values(["recall", "precision", "n_anomalies"], ascending=[False, False, True])
        .iloc[0]
    )

chosen_contamination = float(best_row["contamination"])

print("\n=== Auto-selected contamination ===")
print("Chosen contamination:", chosen_contamination)
print(best_row.to_string())


########### Train final model with chosen contamination ###########

iso_forest = IsolationForest(
    n_estimators=n_estimators,
    contamination=chosen_contamination,
    max_samples=sample_size,
    random_state=42
)
iso_forest.fit(X_normal_scaled)


########### Obtain Results across all patients ############

data["anomaly"] = iso_forest.predict(X_all_scaled)                  # 1 normal, -1 anomaly
data["anomaly_score"] = iso_forest.decision_function(X_all_scaled)  # lower = more abnormal

print("\nAnomaly counts (1=normal, -1=anomaly):")
print(data["anomaly"].value_counts())

print("\nCrosstab (Category vs anomaly):")
print(pd.crosstab(data["Category"], data["anomaly"]))


########### Show most suspicious patients ###########

most_suspicious = data.sort_values("anomaly_score").head(10)
print("\nTop 10 most suspicious patients (lowest scores):")
print(most_suspicious[["Category", "anomaly", "anomaly_score"]].head(10))


########### Visualization ###########

plt.figure(figsize=(10, 5))

normal = data[data["anomaly"] == 1]
anomalies = data[data["anomaly"] == -1]

plt.scatter(normal.index, normal["anomaly_score"], label="Normal", alpha=0.6)
plt.scatter(anomalies.index, anomalies["anomaly_score"], label="Suspicious", alpha=0.6)

plt.xlabel("Patient index")
plt.ylabel("Anomaly score")
plt.legend()
plt.title("Isolation Forest anomaly scores (trained on Blood Donors)")
plt.show()


########### Import the model ###########

# Save the trained model for future use
joblib.dump(iso_forest, "hepatitis_detection.pkl")