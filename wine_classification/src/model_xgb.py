import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, Any


def compute_scale_pos_weight(y: pd.Series) -> float:
    y_arr = np.asarray(y)
    num_1 = float(np.sum(y_arr == 1))
    num_0 = float(np.sum(y_arr == 0))

    if num_1 == 0:
        raise ValueError("No samples present from class 1 in y.")

    return num_0 / num_1


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    scale_pos_weight: float = None,
    early_stopping_rounds: int = 30,
    random_state: int = 42,
    eval_metric: str = "logloss"
) -> XGBClassifier:
    """
    Trains an XGBoost classifier with class-imbalance handling.
    """
    # Compute scale_pos_weight
    if scale_pos_weight is None:
        scale_pos_weight = compute_scale_pos_weight(y_train)
        print(f"scale_pos_weight: {scale_pos_weight:.3f}")

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        eval_metric=eval_metric,
        use_label_encoder=False
    )

    if X_val is not None and y_val is not None:
        eval_set = [(X_train, y_train), (X_val, y_val)]
        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False
        )

        y_val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        print(f"Validation Accuracy (XGBoost): {val_acc:.4f}")

    else:
        model.fit(X_train, y_train)

    return model


def evaluate_xgb_on_test(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    report_str = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== XGBoost Test Performance ===")
    print(f"Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(report_str)
    print("\nConfusion Matrix:")
    print(cm)

    return {
        "accuracy": test_acc,
        "confusion_matrix": cm,
        "classification_report": report_str
    }
