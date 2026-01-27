"""
trains a binary classifier to predict whether a robotic placement
will succeed, based on features known BEFORE attempting the action:
  - Target position (x, y, z)
  - Heuristic reachability score (IK/FK error)
  - Object dimensions (half-height, clearance)
  - Target surface height

"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    precision_recall_curve,
    roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import joblib


# Auto-detect latest dataset
RUNS_DIR = Path("attempts_runs")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def find_latest_dataset():
    """
    Find the most recent dataset in attempts_runs/ directory.
    
    Returns:
        Path to latest JSONL file, or None if not found
    """
    if not RUNS_DIR.exists():
        return None
    
    jsonl_files = list(RUNS_DIR.glob("attempts_*.jsonl"))
    if not jsonl_files:
        return None
    
    # Sort by modification time, newest first
    jsonl_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return jsonl_files[0]


def safe_get(d, keys, default=None):
    """
    Safely navigate nested dictionary.
    
    Args:
        d: Dictionary to navigate
        keys: List of keys to traverse
        default: Default value if key not found
    
    Returns:
        Value at nested key path, or default
    """
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_placement_data(path: Path):
    """
    Load placement attempts from JSONL log file.
    
    CRITICAL: Only use features known BEFORE the placement attempt.
    Post-action features (xy_err, tilt, drift) would be data leakage.
    
    Args:
        path: Path to JSONL file
    
    Returns:
        (X, y, metadata) where:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (1=success, 0=failure)
            metadata: List of dicts with additional info per sample
    """
    X = []
    y = []
    metadata = []

    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON at line {line_num}")
                continue

            # Only process placement actions
            if r.get("action") != "place":
                continue

            feats = r.get("features", {}) or {}
            label = bool(r.get("success", False))

            # Extract desired position (known before action)
            desired = feats.get("desired", None)
            if not desired or len(desired) != 3:
                continue

            cand = feats.get("candidate_meta", {}) or {}

            # -------------------------
            # PRE-ACTION FEATURES ONLY
            # -------------------------
            des_x = float(desired[0])
            des_y = float(desired[1])
            des_z = float(desired[2])

            # Heuristic score: IK→FK error (computed before execution)
            heuristic = safe_get(cand, ["heuristic_score"], np.nan)

            # Object and target properties (known before action)
            target_top_z = feats.get("target_top_z", np.nan)
            obj_half_h = feats.get("obj_half_h", np.nan)
            clearance = feats.get("clearance", np.nan)

            # Feature vector (7 features)
            feature_vec = [
                des_x, 
                des_y, 
                des_z, 
                heuristic, 
                target_top_z, 
                obj_half_h, 
                clearance
            ]

            X.append(feature_vec)
            y.append(1 if label else 0)
            
            # Store metadata for analysis
            metadata.append({
                "episode": r.get("episode", -1),
                "object": r.get("object", "unknown"),
                "target": r.get("target", "unknown"),
                "fail_type": r.get("fail_type", None),
                "xy_err": feats.get("xy_err", None),
                "tilt_deg": feats.get("tilt_deg", None),
            })

    return np.array(X, dtype=float), np.array(y, dtype=int), metadata


def print_dataset_stats(y, metadata):
    """Print statistics about the loaded dataset."""
    n_total = len(y)
    n_success = y.sum()
    n_failure = n_total - n_success
    success_rate = n_success / n_total if n_total > 0 else 0

    print("\n" + "=" * 70)
    print("📊 Dataset Statistics")
    print("=" * 70)
    print(f"Total samples:     {n_total}")
    print(f"Successes:         {n_success} ({100*success_rate:.1f}%)")
    print(f"Failures:          {n_failure} ({100*(1-success_rate):.1f}%)")
    
    # Failure type breakdown
    fail_types = {}
    for i, label in enumerate(y):
        if label == 0:  # failure
            ft = metadata[i].get("fail_type", "unknown")
            fail_types[ft] = fail_types.get(ft, 0) + 1
    
    if fail_types:
        print("\nFailure breakdown:")
        for ft, count in sorted(fail_types.items(), key=lambda x: -x[1]):
            print(f"  {ft}: {count} ({100*count/n_failure:.1f}%)")
    
    print("=" * 70)


def train_and_evaluate_model(X_train, X_test, y_train, y_test, 
                             model_name, model_pipeline):
    """
    Train a model and print comprehensive evaluation metrics.
    
    Args:
        X_train, X_test, y_train, y_test: Train/test splits
        model_name: Name of the model (for display)
        model_pipeline: sklearn Pipeline object
    
    Returns:
        Trained pipeline
    """
    print(f"\n{'=' * 70}")
    print(f"Training {model_name}")
    print("=" * 70)
    
    # Train
    model_pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nPerformance Metrics:")
    print(f"  AUC-ROC: {auc:.3f}")
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"              Predicted")
    print(f"              Fail  Success")
    print(f"  Actual Fail  {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       Success {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                               target_names=["Failure", "Success"],
                               digits=3))
    
    # Cross-validation score
    cv_scores = cross_val_score(model_pipeline, X_train, y_train, 
                               cv=5, scoring='roc_auc')
    print(f"5-Fold CV AUC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    
    return model_pipeline


def print_feature_importances(model_pipeline, feature_names):
    """
    Print feature importances for tree-based models.
    
    Args:
        model_pipeline: Trained sklearn Pipeline
        feature_names: List of feature names
    """
    # Check if model has feature_importances_
    clf = model_pipeline.named_steps.get("clf")
    if not hasattr(clf, "feature_importances_"):
        return
    
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1]
    
    print("\nFeature Importances:")
    print("  " + "-" * 40)
    for i in order:
        bar_length = int(importances[i] * 40)
        bar = "#" * bar_length
        print(f"  {feature_names[i]:>15}: {importances[i]:.4f} {bar}")
    print("  " + "-" * 40)


def save_model(model, model_name, metrics):
    """
    Save trained model with metadata.
    
    Args:
        model: Trained sklearn Pipeline
        model_name: Name for the saved file
        metrics: Dict of performance metrics
    """
    save_path = MODELS_DIR / f"{model_name}.joblib"
    
    # Save model with metadata
    model_data = {
        "model": model,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
        "feature_names": [
            "des_x", "des_y", "des_z", 
            "heuristic_score", 
            "target_top_z", "obj_half_h", "clearance"
        ]
    }
    
    joblib.dump(model_data, save_path)
    print(f"\n💾 Saved model to: {save_path}")


def main():
    """Main training pipeline."""
    print("\n" + "=" * 70)
    print("🎓 ML Model Training for Placement Success Prediction")
    print("=" * 70)
    
    # Find latest dataset
    data_path = find_latest_dataset()
    
    if data_path is None:
        print("\n❌ Error: No dataset found in attempts_runs/")
        print("   Run `python collect_dataset.py` first to collect training data.")
        return
    
    print(f"\n📂 Loading dataset: {data_path.name}")
    print(f"   Modified: {datetime.fromtimestamp(data_path.stat().st_mtime)}")
    
    # Load data
    X, y, metadata = load_placement_data(data_path)
    
    if len(y) == 0:
        print("\n❌ Error: No placement attempts found in dataset.")
        print("   Make sure the dataset contains 'place' actions.")
        return
    
    # Print dataset statistics
    print_dataset_stats(y, metadata)
    
    # Check for class imbalance
    if y.sum() < 10 or (len(y) - y.sum()) < 10:
        print("\n⚠️  Warning: Very few samples in one class. Consider collecting more data.")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"\n📊 Train/Test Split:")
    print(f"   Training:   {len(y_train)} samples ({y_train.sum()} successes)")
    print(f"   Testing:    {len(y_test)} samples ({y_test.sum()} successes)")
    
    feature_names = [
        "des_x", "des_y", "des_z", 
        "heuristic_score", 
        "target_top_z", "obj_half_h", "clearance"
    ]
    
    # -------------------------
    # Model 1: Logistic Regression (Interpretable Baseline)
    # -------------------------
    lr_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ])
    
    lr_model = train_and_evaluate_model(
        X_train, X_test, y_train, y_test,
        "Logistic Regression",
        lr_pipeline
    )
    
    # Get LR coefficients
    lr_clf = lr_model.named_steps["clf"]
    print("\nLogistic Regression Coefficients:")
    print("  " + "-" * 40)
    for name, coef in zip(feature_names, lr_clf.coef_[0]):
        sign = "+" if coef >= 0 else "-"
        print(f"  {name:>15}: {sign}{abs(coef):6.3f}")
    print("  " + "-" * 40)
    
    lr_auc = roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1])
    
    # -------------------------
    # Model 2: Random Forest (Nonlinear, Production Model)
    # -------------------------
    rf_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced_subsample",
        )),
    ])
    
    rf_model = train_and_evaluate_model(
        X_train, X_test, y_train, y_test,
        "Random Forest",
        rf_pipeline
    )
    
    print_feature_importances(rf_model, feature_names)
    
    rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
    
    # -------------------------
    # Model Comparison
    # -------------------------
    print("\n" + "=" * 70)
    print("Model Comparison")
    print("=" * 70)
    print(f"  Logistic Regression AUC: {lr_auc:.3f}")
    print(f"  Random Forest AUC:       {rf_auc:.3f}")
    
    if rf_auc > lr_auc:
        print(f"\n  * Random Forest wins by {rf_auc - lr_auc:.3f}")
        best_model = rf_model
        best_name = "place_rf"
        best_auc = rf_auc
    else:
        print(f"\n  * Logistic Regression wins by {lr_auc - rf_auc:.3f}")
        best_model = lr_model
        best_name = "place_lr"
        best_auc = lr_auc
    
    # Save best model
    save_model(best_model, best_name, {"auc": best_auc})
    
    # Also save both models for comparison
    save_model(lr_model, "place_lr", {"auc": lr_auc})
    save_model(rf_model, "place_rf", {"auc": rf_auc})
    
    # -------------------------
    # Final Summary
    # -------------------------
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"  Dataset:        {data_path.name}")
    print(f"  Samples:        {len(y)} ({y.sum()} successes, {len(y)-y.sum()} failures)")
    print(f"  Best Model:     {best_name}")
    print(f"  Best AUC:       {best_auc:.3f}")
    print(f"  Models saved:   {MODELS_DIR}/")
    print("=" * 70)
    print("\nNext steps:")
    print("   1. Run `python test_reliability.py` to test the model")
    print("   2. Run `python simulation.py` to use the model interactively")
    print("   3. Collect more data with `python collect_dataset.py` to improve performance")


if __name__ == "__main__":
    main()
