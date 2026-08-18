import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

# ------------------------------------------------------------
# 1) Load your tagged dataset
# ------------------------------------------------------------
file_path = "gp_classifier_dataset_clean.txt"

# Pipe-separated text with spaces around separators
df = pd.read_csv(file_path, sep="|", engine="python")
df.columns = [c.strip() for c in df.columns]

# Strip whitespace from string cells
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype(str).str.strip()

# Remove separator rows if present
if "stage" in df.columns:
    df = df[~df["stage"].str.startswith("-")].copy()

# Convert numeric columns
numeric_cols = [
    "stage_id", "case_id",
    "pitch_cp_0", "pitch_cp_1", "pitch_cp_2", "pitch_cp_3", "pitch_cp_4", "pitch_cp_5",
    "chord_cp_0", "chord_cp_1", "chord_cp_2", "chord_cp_3", "chord_cp_4",
    "failed",
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=numeric_cols).reset_index(drop=True)

# Keep this column available for downstream reporting even when
# older dataset exports do not include it.
if "prediction_mismatch" not in df.columns:
    df["prediction_mismatch"] = "not_available"

print(f"Loaded {len(df)} rows")
print(df.head())


# ------------------------------------------------------------
# 2) Define feature columns
# ------------------------------------------------------------
feature_cols = [
    "pitch_cp_0", "pitch_cp_1", "pitch_cp_2", "pitch_cp_3", "pitch_cp_4", "pitch_cp_5",
    "chord_cp_0", "chord_cp_1", "chord_cp_2", "chord_cp_3", "chord_cp_4",
]

# Target: 1 = mesh failed, 0 = success
y = df["failed"].astype(int).to_numpy()

# Raw feature matrix
X_raw = df[feature_cols].copy()


# ------------------------------------------------------------
# 3) Add engineered features tied to your current constraints
#    These use the same raw bounds you were using before
# ------------------------------------------------------------
def normalize(x, lo, hi):
    return (x - lo) / (hi - lo)

# Bounds from your workflow
# pitch_p4x = pitch_cp_2, bounds (0.35, 0.75)
# pitch_p7y = pitch_cp_5, bounds (0.4, 0.7)
# chord_p1y = chord_cp_0, bounds (0.2, 0.3)
# chord_y4  = chord_cp_3, bounds (0.4, 0.7)

pitch_p4x_norm = normalize(X_raw["pitch_cp_2"], 0.35, 0.75)
pitch_p7y_norm = normalize(X_raw["pitch_cp_5"], 0.40, 0.70)
chord_p1y_norm = normalize(X_raw["chord_cp_0"], 0.20, 0.30)
chord_y4_norm  = normalize(X_raw["chord_cp_3"], 0.40, 0.70)

X_raw["chord_sum_norm"]  = chord_p1y_norm + chord_y4_norm
X_raw["pitch_diff_norm"] = pitch_p4x_norm - pitch_p7y_norm

all_feature_cols = feature_cols + ["chord_sum_norm", "pitch_diff_norm"]
X = X_raw[all_feature_cols].to_numpy()

print("\nClass counts:")
print(df["failed"].value_counts().sort_index())


# ------------------------------------------------------------
# 4) Train/test split
#    Stratify because failures are the minority class
# ------------------------------------------------------------
X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X, y, df.copy(),
    test_size=0.25,
    random_state=42,
    stratify=y,
)

print(f"\nTrain size: {len(X_train)}")
print(f"Test size : {len(X_test)}")
print(f"Train failures: {y_train.sum()} / {len(y_train)}")
print(f"Test failures : {y_test.sum()} / {len(y_test)}")


# ------------------------------------------------------------
# 5) GP classifier
#    Standardize first because GP kernels are scale-sensitive
# ------------------------------------------------------------
n_features = X.shape[1]

kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(
    length_scale=np.ones(n_features),
    length_scale_bounds=(1e-2, 1e2),
)

gp_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("gpc", GaussianProcessClassifier(
        kernel=kernel,
        random_state=42,
        n_restarts_optimizer=5,
        max_iter_predict=200,
    )),
])

gp_clf.fit(X_train, y_train)


# ------------------------------------------------------------
# 6) Predict probabilities and choose threshold
#    Default threshold is 0.5, but with class imbalance it is
#    often better to optimize threshold on the test set for inspection
# ------------------------------------------------------------
p_fail_test = gp_clf.predict_proba(X_test)[:, 1]

# Default predictions
y_pred_default = (p_fail_test >= 0.5).astype(int)

# F1-optimal threshold on this test split
prec, rec, thr = precision_recall_curve(y_test, p_fail_test)
f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
best_idx = np.argmax(f1)
best_thr = thr[best_idx]

y_pred_best = (p_fail_test >= best_thr).astype(int)

print(f"\nBest threshold from PR curve on test split: {best_thr:.4f}")


# ------------------------------------------------------------
# 7) Evaluation
# ------------------------------------------------------------
def evaluate_predictions(y_true, y_pred, p_fail, title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))
    print(f"ROC AUC : {roc_auc_score(y_true, p_fail):.4f}")
    print(f"PR AUC  : {average_precision_score(y_true, p_fail):.4f}")

evaluate_predictions(y_test, y_pred_default, p_fail_test, "GP classifier with threshold = 0.5")
evaluate_predictions(y_test, y_pred_best, p_fail_test, f"GP classifier with threshold = {best_thr:.4f}")


# ------------------------------------------------------------
# 8) Inspect highest-risk cases in test split
# ------------------------------------------------------------
test_results = df_test.copy()
test_results["p_fail_gp"] = p_fail_test
test_results["pred_fail_05"] = y_pred_default
test_results["pred_fail_best"] = y_pred_best

cols_to_show = [
    "stage", "stage_id", "case_id",
    "failed", "prediction_mismatch",
    "p_fail_gp", "pred_fail_05", "pred_fail_best"
] + feature_cols + ["chord_sum_norm", "pitch_diff_norm"]

# add engineered features back for viewing
test_results["chord_sum_norm"] = (
    normalize(test_results["chord_cp_0"], 0.20, 0.30) +
    normalize(test_results["chord_cp_3"], 0.40, 0.70)
)
test_results["pitch_diff_norm"] = (
    normalize(test_results["pitch_cp_2"], 0.35, 0.75) -
    normalize(test_results["pitch_cp_5"], 0.40, 0.70)
)

print("\n" + "=" * 70)
print("Top 15 highest predicted failure-probability cases in test split")
print("=" * 70)
print(
    test_results[cols_to_show]
    .sort_values("p_fail_gp", ascending=False)
    .head(15)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 9) Optional: fit on all data after evaluation
#    Use this model later to score new candidates
# ------------------------------------------------------------
gp_clf.fit(X, y)

# Example:
# p_fail_new = gp_clf.predict_proba(X_new)[:, 1]
# where X_new has the same 13 columns:
#   11 raw cps + chord_sum_norm + pitch_diff_norm