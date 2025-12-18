import os
import json
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, Model

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

# ============================================================
# 0) REPRO + CPU ONLY
# ============================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.config.set_visible_devices([], "GPU")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass


# ============================================================
# 1) CONFIG
# ============================================================
TRAIN_CSV_PATH = "data/Model_3LB_ALL.csv"
OOS_CSV_PATH   = "data/Model_3LB_ALL_oos.csv"

TARGET_COL = "outputC"
DROP_COLS = ["output", "outputC"]

VAL_SPLIT = 0.20
EPOCHS = 200
BATCH  = 64

# Model save paths
ARTIFACT_DIR = "artifacts_tf_mixer"
BEST_MODEL_PATH = os.path.join(ARTIFACT_DIR, "best_model.keras")
META_PATH       = os.path.join(ARTIFACT_DIR, "meta.json")

# Threshold behavior
# - If you want "fixed 0.5", leave it.
# - If you want "best threshold from val", set USE_BEST_VAL_THRESHOLD=True
USE_BEST_VAL_THRESHOLD = True
THRESH_SWEEP_STEPS = 201  # for best-F1 threshold search


# ============================================================
# 2) UTIL: BASIC HYGIENE
# ============================================================
def clean_features(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    X[~np.isfinite(X)] = np.nan
    if np.isnan(X).any():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return X

def assert_binary(y: np.ndarray, name="y"):
    uniq = np.unique(y[~np.isnan(y)])
    if not set(uniq.tolist()).issubset({0.0, 1.0}):
        raise ValueError(f"{name} must be binary 0/1. Found: {uniq[:20]}")

def chronological_split(X, y, val_split: float):
    n = len(X)
    split = int((1.0 - val_split) * n)
    if split <= 0 or split >= n:
        raise ValueError("Bad VAL_SPLIT; leaves empty train or val.")
    return (X[:split], y[:split]), (X[split:], y[split:])

def best_threshold_by_f1(y_true, p, steps=201):
    thresholds = np.linspace(0.0, 1.0, steps)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        y_pred = (p >= t).astype(int)
        # manual f1 for speed/clarity
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, float(best_f1)


# ============================================================
# 3) MODEL
# ============================================================
def mixer_block(x, feature_dim, expansion_factor=4, dropout=0.2):
    y = layers.LayerNormalization()(x)
    y = layers.Dense(feature_dim * expansion_factor, activation="gelu")(y)
    y = layers.Dropout(dropout)(y)
    y = layers.Dense(feature_dim)(y)
    x = layers.Add()([x, y])

    y = layers.LayerNormalization()(x)
    y = layers.Dense(feature_dim * expansion_factor, activation="gelu")(y)
    y = layers.Dropout(dropout)(y)
    y = layers.Dense(feature_dim)(y)
    x = layers.Add()([x, y])
    return x

def build_tf_binary_mixer(input_dim: int, n_blocks=3, expansion_factor=4, dropout=0.2) -> Model:
    inputs = layers.Input(shape=(input_dim,), name="features")
    norm = layers.Normalization(name="normalizer")(inputs)

    x = norm
    for _ in range(n_blocks):
        x = mixer_block(x, feature_dim=input_dim, expansion_factor=expansion_factor, dropout=dropout)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid", name="outputC")(x)

    model = Model(inputs, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="val_auc_proxy", curve="ROC"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model

def tf_callbacks(best_path: str):
    os.makedirs(os.path.dirname(best_path), exist_ok=True)

    # Save best model by validation AUC
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_path,
        monitor="val_val_auc_proxy",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_val_auc_proxy",
        mode="max",
        patience=12,
        restore_best_weights=False,  # we reload from disk, so keep this False
        verbose=1,
    )

    lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_val_auc_proxy",
        mode="max",
        factor=0.5,
        patience=6,
        min_lr=1e-6,
        verbose=1,
    )
    return [ckpt, early, lr_reduce]


# ============================================================
# 4) LOAD TRAIN, TRAIN, SAVE META
# ============================================================
train_df = pd.read_csv(TRAIN_CSV_PATH)
feature_cols = [c for c in train_df.columns if c not in DROP_COLS]

X_all = clean_features(train_df[feature_cols].values)
y_all = train_df[TARGET_COL].values.astype(np.float32)

assert_binary(y_all, "y_all")

(X_train, y_train), (X_val, y_val) = chronological_split(X_all, y_all, VAL_SPLIT)

print(f"Train rows: {len(X_train)} | Val rows: {len(X_val)} | Features: {X_train.shape[1]}")

# Class weights (only if both classes exist)
classes = np.unique(y_train)
class_weight_dict = None
if len(classes) == 2:
    cw = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train.astype(int))
    class_weight_dict = {0: float(cw[0]), 1: float(cw[1])}
else:
    print(f"WARNING: y_train has classes {classes}; disabling class_weight.")

print("Class weights:", class_weight_dict)

# Build + adapt
model = build_tf_binary_mixer(input_dim=X_train.shape[1], n_blocks=3, expansion_factor=4, dropout=0.2)
model.get_layer("normalizer").adapt(X_train)

model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=tf_callbacks(BEST_MODEL_PATH),
    class_weight=class_weight_dict,
    verbose=2,
)

if not os.path.exists(BEST_MODEL_PATH):
    raise RuntimeError("Best model file not found. ModelCheckpoint did not write. Check monitor name/metrics.")

# Decide threshold (fixed or best-on-val)
val_proba = model.predict(X_val, verbose=0).ravel()
if USE_BEST_VAL_THRESHOLD:
    best_t, best_f1 = best_threshold_by_f1(y_val.astype(int), val_proba, steps=THRESH_SWEEP_STEPS)
else:
    best_t, best_f1 = 0.5, None

meta = {
    "feature_cols": feature_cols,
    "val_split": VAL_SPLIT,
    "best_model_path": BEST_MODEL_PATH,
    "use_best_val_threshold": USE_BEST_VAL_THRESHOLD,
    "threshold": best_t,
    "val_best_f1_at_threshold": best_f1,
    "seed": SEED,
}

os.makedirs(ARTIFACT_DIR, exist_ok=True)
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nSaved best model: {BEST_MODEL_PATH}")
print(f"Saved meta:       {META_PATH}")
print(f"Threshold used:   {best_t:.4f}" + (f" (val best F1={best_f1:.4f})" if best_f1 is not None else ""))


# ============================================================
# 5) RELOAD SAVED MODEL (clean separation)
# ============================================================
reloaded = tf.keras.models.load_model(BEST_MODEL_PATH)
print("\nReloaded model OK.")


# ============================================================
# 6) STREAMING OOS EVALUATION (iterate row-by-row)
# ============================================================
oos_df = pd.read_csv(OOS_CSV_PATH)

missing_in_oos = [c for c in feature_cols if c not in oos_df.columns]
if missing_in_oos:
    raise ValueError(f"OOS file missing features: {missing_in_oos}")

X_oos = clean_features(oos_df[feature_cols].values)
y_oos = oos_df[TARGET_COL].values.astype(np.float32)
assert_binary(y_oos, "y_oos")

t = float(best_t)

# Running confusion matrix counts
tn = fp = fn = tp = 0

# Track probs for AUC/AP (store for true computation)
all_p = []
all_y = []

# Optional: track running metrics snapshots
snap_every = 5000  # print progress every N rows

def compute_running_metrics(tn, fp, fn, tp):
    total = tn + fp + fn + tp
    acc = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return acc, precision, recall, f1

print("\n=======================================")
print("Streaming OOS evaluation (row-by-row)")
print("=======================================")

for i in range(len(X_oos)):
    x = X_oos[i:i+1]  # keep batch dimension
    y_true = int(y_oos[i])

    p = float(reloaded.predict(x, verbose=0).ravel()[0])
    y_pred = 1 if p >= t else 0

    all_p.append(p)
    all_y.append(y_true)

    if y_true == 1 and y_pred == 1:
        tp += 1
    elif y_true == 0 and y_pred == 0:
        tn += 1
    elif y_true == 0 and y_pred == 1:
        fp += 1
    else:
        fn += 1

    if (i + 1) % snap_every == 0:
        acc, prec, rec, f1 = compute_running_metrics(tn, fp, fn, tp)
        print(f"  row={i+1:>8d} | acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")

# Final threshold-based metrics
acc, prec, rec, f1 = compute_running_metrics(tn, fp, fn, tp)
cm = np.array([[tn, fp],
               [fn, tp]], dtype=int)

# Final prob-based metrics (true computations)
all_p = np.array(all_p, dtype=np.float64)
all_y = np.array(all_y, dtype=int)

roc = roc_auc_score(all_y, all_p) if len(np.unique(all_y)) == 2 else float("nan")
ap  = average_precision_score(all_y, all_p) if len(np.unique(all_y)) == 2 else float("nan")

print("\n=======================================")
print("OOS FINAL REPORT")
print("=======================================")
print(f"Rows evaluated: {len(all_y)}")
print(f"Threshold:      {t:.4f}")
print(f"Pos rate:       {all_y.mean():.6f}")
print("\nThreshold metrics:")
print(f"  Accuracy:  {acc:.6f}")
print(f"  Precision: {prec:.6f}")
print(f"  Recall:    {rec:.6f}")
print(f"  F1:        {f1:.6f}")

print("\nProbability metrics:")
print(f"  ROC-AUC:   {roc:.6f}")
print(f"  PR-AUC:    {ap:.6f}")

print("\nConfusion Matrix (rows=actual, cols=pred):")
print(cm)

print("\nClassification report:")
print(classification_report(all_y, (all_p >= t).astype(int), digits=4, zero_division=0))

print("\nDone.")
