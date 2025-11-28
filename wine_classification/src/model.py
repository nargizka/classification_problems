import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .config import RANDOM_STATE

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    max_depth: int ,
    n_estimators: int = 200,
    random_state: int = RANDOM_STATE,
) -> RandomForestClassifier:
    """
    Trains a RandomForestClassifier.
    """
    rf = RandomForestClassifier(
           n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced"
    )
    rf.fit(X_train, y_train)

    if X_val is not None and y_val is not None:
        y_val_pred = rf.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        print(f"Validation Accuracy: {val_acc:.4f}")

    return rf

def evaluate_on_test(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=False)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Test Set Performance ===")
    print(f"Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)

    return {
        "accuracy": test_acc,
        "confusion_matrix": cm,
    }

