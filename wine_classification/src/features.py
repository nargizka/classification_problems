import pandas as pd
from sklearn.model_selection import train_test_split
from .config import QUALITY_THRESHOLD, RANDOM_STATE, TEST_SIZE, VAL_SIZE

def engineer_features(df: pd.DataFrame, quality_threshold: int = QUALITY_THRESHOLD):
    if "quality" not in df.columns:
        raise ValueError("DataFrame must contain 'quality' column.")

    df = df.copy()
    df["good_quality"] = (df["quality"] >= quality_threshold).astype(int)

    X = df.drop(columns=["quality", "good_quality"])
    y = df["good_quality"]

    return X, y

def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
