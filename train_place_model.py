import json
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import joblib



DATA_PATH = "attempts_runs/attempts_20260117_011254.jsonl"  # <-- update if needed


def safe_get(d, keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_rows(path: str):
    X = []
    y = []

    with open(path, "r") as f:
        for line in f:
            r = json.loads(line)
            if r.get("action") != "place":
                continue

            feats = r.get("features", {}) or {}
            label = bool(r.get("success", False))

            desired = feats.get("desired", None)
            if not desired or len(desired) != 3:
                continue

            cand = feats.get("candidate_meta", {}) or {}

            # -------------------------
            # ONLY PRE-ACTION FEATURES
            # -------------------------
            des_x = float(desired[0])
            des_y = float(desired[1])
            des_z = float(desired[2])

            # heuristic_score is computed BEFORE the action (from reachability + IK/FK error)
            heuristic = safe_get(cand, ["heuristic_score"], np.nan)

            # known before action (computed from AABBs etc.)
            target_top_z = feats.get("target_top_z", np.nan)
            obj_half_h = feats.get("obj_half_h", np.nan)
            clearance = feats.get("clearance", np.nan)

            # Feature vector (NO attempt_rank / attempt_pool / pool_code)
            X.append([des_x, des_y, des_z, heuristic, target_top_z, obj_half_h, clearance])
            y.append(1 if label else 0)

    return np.array(X, dtype=float), np.array(y, dtype=int)


def main():
    path = Path(DATA_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset: {path}")

    X, y = load_rows(str(path))
    if len(y) == 0:
        raise RuntimeError("No place rows loaded. Check DATA_PATH and that action=='place' exists.")

    print("Loaded place rows:", len(y), " | success rate:", float(y.mean()))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Model 1: Logistic Regression baseline
    lr = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    lr.fit(X_train, y_train)

    p_lr = lr.predict_proba(X_test)[:, 1]
    pred_lr = (p_lr >= 0.5).astype(int)

    print("\n=== Logistic Regression ===")
    print("AUC:", round(roc_auc_score(y_test, p_lr), 3))
    print(confusion_matrix(y_test, pred_lr))
    print(classification_report(y_test, pred_lr, digits=3))

    # Model 2: Random Forest (nonlinear baseline)
    rf = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
        )),
    ])
    rf.fit(X_train, y_train)

    joblib.dump(rf, "models/place_rf.joblib")
    print("Saved model to models/place_rf.joblib")

    p_rf = rf.predict_proba(X_test)[:, 1]
    pred_rf = (p_rf >= 0.5).astype(int)

    print("\n=== Random Forest ===")
    print("AUC:", round(roc_auc_score(y_test, p_rf), 3))
    print(confusion_matrix(y_test, pred_rf))
    print(classification_report(y_test, pred_rf, digits=3))

    # Feature importances (RF only)
    feature_names = ["des_x", "des_y", "des_z", "heuristic", "target_top_z", "obj_half_h", "clearance"]
    importances = rf.named_steps["clf"].feature_importances_
    order = np.argsort(importances)[::-1]

    print("\nTop RF feature importances:")
    for i in order[:len(feature_names)]:
        print(f"{feature_names[i]:>12}: {importances[i]:.4f}")


if __name__ == "__main__":
    main()
