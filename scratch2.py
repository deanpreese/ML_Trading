import os
import json
from datetime import datetime

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score,
)

tf.config.set_visible_devices([], "GPU")
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# 1. CONFIG
# ============================================================
TRAIN_CSV_PATH = "data/Model_3LB_ALL.csv"
TEST_CSV_PATH  = "data/Model_3LB_ALL_oos.csv"

TARGET_COL = "outputC"
DROP_COLS = ["output", "outputC"]

TRAIN_SPLIT = 0.80
THRESHOLD = 0.50

EPOCHS = 200
BATCH_SIZE = 64

# Where to save artifacts
RUNS_DIR = "runs"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(RUNS_DIR, RUN_ID)
os.makedirs(RUN_DIR, exist_ok=True)

MODEL_PATH = os.path.join(RUN_DIR, "final_model.keras")
METRICS_PATH = os.path.join(RUN_DIR, "metrics.json")
HISTORY_PATH = os.path.join(RUN_DIR, "history.csv")
CM_PATH = os.path.join(RUN_DIR, "confusion_matrix.csv")
REPORT_PATH = os.path.join(RUN_DIR, "classification_report.txt")
FEATURES_PATH = os.path.join(RUN_DIR, "feature_cols.json")

# ============================================================
# 2. Load data
# ============================================================
train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df  = pd.read_csv(TEST_CSV_PATH)

feature_cols = [c for c in train_df.columns if c not in DROP_COLS]

missing_in_test = [c for c in feature_cols if c not in test_df.columns]
if missing_in_test:
    raise ValueError(f"Test file is missing feature columns: {missing_in_test}")

X_all = train_df[feature_cols].values.astype("float32")
y_all = train_df[TARGET_COL].values.astype("int32")  # ensure integer classes

X_test = test_df[feature_cols].values.astype("float32")
y_test = test_df[TARGET_COL].values.astype("int32")

n_samples = len(train_df)
print(f"Total train+val samples: {n_samples}")
print(f"OOS test samples:        {len(test_df)}")

# ============================================================
# 3. Chronological split: train / validation
# ============================================================
train_end = int(TRAIN_SPLIT * n_samples)

X_train, y_train = X_all[:train_end], y_all[:train_end]
X_val,   y_val   = X_all[train_end:], y_all[train_end:]

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val   shape: {X_val.shape}, y_val   shape: {y_val.shape}")
print(f"X_test  shape: {X_test.shape}, y_test  shape: {y_test.shape}")

# ============================================================
# 4. Class weights (for imbalance)
# ============================================================
classes, counts = np.unique(y_train, return_counts=True)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}

print("\nClass distribution (train):")
for c, cnt, w in zip(classes, counts, class_weights):
    print(f"  Class {int(c)}: count={cnt}, weight={w:.4f}")

print("\nUsing class weights:", class_weight_dict)

# ============================================================
# 5. Model definition
# ============================================================
def mixer_block(x, feature_dim, expansion_factor=4):
    y = layers.LayerNormalization()(x)
    y = layers.Dense(feature_dim * expansion_factor, activation="gelu")(y)
    y = layers.Dense(feature_dim)(y)
    x = layers.Add()([x, y])

    y = layers.LayerNormalization()(x)
    y = layers.Dense(feature_dim * expansion_factor, activation="gelu")(y)
    y = layers.Dense(feature_dim)(y)
    x = layers.Add()([x, y])
    return x

def build_binary_mixer(input_dim: int) -> Model:
    inputs = layers.Input(shape=(input_dim,), name="features")
    norm = layers.Normalization(name="normalizer")(inputs)

    x = norm
    z = norm
    y = norm

    for _ in range(3):
        x = mixer_block(x, feature_dim=input_dim, expansion_factor=4)
        y = mixer_block(y, feature_dim=input_dim, expansion_factor=2)
        z = mixer_block(z, feature_dim=input_dim, expansion_factor=1)

    x = layers.Dense(128, activation="relu")(x)
    y = layers.Dense(128, activation="relu")(y)
    z = layers.Dense(128, activation="relu")(z)

    x = layers.Average()([x, y, z])
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(64, activation="relu")(x)
    output = layers.Dense(1, activation="sigmoid", name="outputC")(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model

def get_callbacks():
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=12,
        restore_best_weights=True,
        verbose=1,
    )

    lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc",
        mode="max",
        factor=0.5,
        patience=6,
        min_lr=1e-6,
        verbose=1,
    )
    
    #checkpoint_dir = 'checkpoints/'
    #trained_dir = 'trained_models/'
    ##checkpoint_model = os.path.join(checkpoint_dir, f"model_{model_name}_classifier.keras")
    #model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #    checkpoint_model, monitor='val_loss', verbose=1,
    #    save_best_only=True, save_weights_only=False, mode='min'
    #    )

    return [early_stop, lr_reduce]

# ============================================================
# 6. Build, adapt normalizer, train
# ============================================================
input_dim = X_train.shape[1]
model = build_binary_mixer(input_dim)
model.summary()

normalizer = model.get_layer("normalizer")
normalizer.adapt(X_train)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=get_callbacks(),
    class_weight=class_weight_dict,
    verbose=2,
)

# Save training history
hist_df = pd.DataFrame(history.history)
hist_df.to_csv(HISTORY_PATH, index=False)

# ============================================================
# 7. Validation evaluation
# ============================================================
print("\n======================")
print("Validation evaluation")
print("======================")
val_results = model.evaluate(X_val, y_val, verbose=2)
val_metric_dict = dict(zip(model.metrics_names, [float(x) for x in val_results]))
print(val_metric_dict)

# ============================================================
# 8. OOS Test evaluation
# ============================================================
print("\n=======================================")
print("Out-of-sample (OOS) test set evaluation")
print("=======================================")

test_results = model.evaluate(X_test, y_test, verbose=0)
keras_metric_dict = dict(zip(model.metrics_names, [float(x) for x in test_results]))

print("\nRaw Keras metrics on OOS:")
for k, v in keras_metric_dict.items():
    print(f"  {k}: {v:.6f}")

y_proba = model.predict(X_test, verbose=0).ravel()
y_pred = (y_proba >= THRESHOLD).astype(int)

acc = float(accuracy_score(y_test, y_pred))
auc = float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) > 1 else float("nan")
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="binary", zero_division=0
)
precision, recall, f1 = float(precision), float(recall), float(f1)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = [int(x) for x in cm.ravel()]

print("\nKey classification metrics on OOS:")
print(f"  Accuracy:  {acc:.4f}")
print(f"  AUC:       {auc:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-score:  {f1:.4f}")

print("\nConfusion Matrix (OOS, threshold={:.2f}):".format(THRESHOLD))
print("            Pred 0     Pred 1")
print(f"Actual 0    {tn:7d}   {fp:7d}")
print(f"Actual 1    {fn:7d}   {tp:7d}")

report = classification_report(y_test, y_pred, digits=4, zero_division=0)
print("\nDetailed classification report (OOS):")
print(report)

# ============================================================
# 9. SAVE MODEL + METRICS ARTIFACTS
# ============================================================
# Save final model (includes the adapted Normalization layer)
model.save(MODEL_PATH)

# Save confusion matrix as CSV
cm_df = pd.DataFrame(
    cm,
    index=["actual_0", "actual_1"],
    columns=["pred_0", "pred_1"],
)
cm_df.to_csv(CM_PATH, index=True)

# Save classification report text
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report)

# Save feature list
with open(FEATURES_PATH, "w", encoding="utf-8") as f:
    json.dump(feature_cols, f, indent=2)

# Consolidated metrics JSON (single source of truth for your run)
artifact = {
    "run_id": RUN_ID,
    "paths": {
        "train_csv": TRAIN_CSV_PATH,
        "test_csv": TEST_CSV_PATH,
        "model": MODEL_PATH,
        "metrics_json": METRICS_PATH,
        "history_csv": HISTORY_PATH,
        "confusion_matrix_csv": CM_PATH,
        "classification_report_txt": REPORT_PATH,
        "feature_cols_json": FEATURES_PATH,
    },
    "config": {
        "target_col": TARGET_COL,
        "drop_cols": DROP_COLS,
        "train_split": TRAIN_SPLIT,
        "threshold": THRESHOLD,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seed": 42,
    },
    "shapes": {
        "X_train": list(X_train.shape),
        "X_val": list(X_val.shape),
        "X_test": list(X_test.shape),
    },
    "class_distribution_train": {
        str(int(c)): {"count": int(cnt), "weight": float(w)}
        for c, cnt, w in zip(classes, counts, class_weights)
    },
    "val_metrics_keras": val_metric_dict,
    "oos_metrics_keras": keras_metric_dict,
    "oos_metrics_sklearn": {
        "accuracy": acc,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {
            "tn": tn, "fp": fp, "fn": fn, "tp": tp
        },
        "actual_positive_rate": float(y_test.mean()),
        "predicted_positive_rate": float(y_pred.mean()),
    },
    "history_last_epoch": {
        k: float(v[-1]) for k, v in history.history.items()
        if isinstance(v, (list, tuple)) and len(v) > 0
    },
}

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(artifact, f, indent=2)

print("\n=======================================")
print("Saved run artifacts to:", RUN_DIR)
print("Model:", MODEL_PATH)
print("Metrics:", METRICS_PATH)
print("History:", HISTORY_PATH)
print("Confusion matrix:", CM_PATH)
print("Report:", REPORT_PATH)
print("Done.")
