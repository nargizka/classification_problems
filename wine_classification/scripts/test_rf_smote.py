import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import RED_WINE_CSV
from src.data import load_wine_data
from src.features import engineer_features, split_train_val_test, apply_SMOTE
from src.model import train_random_forest, evaluate_on_test
from src.visualize import plot_feature_importances

def main():
    df = load_wine_data(kind='red')
    X, y = engineer_features(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y)

    # Applies SMOTE on training dataset to handle imbalanced classes
    X_train_res, y_train_res = apply_SMOTE(X_train, y_train)

    rf_model = train_random_forest(
        X_train_res, y_train_res,
        X_val, y_val,
        n_estimators=300,
        max_depth=None
    )

    evaluate_on_test(rf_model, X_test, y_test)
    plot_feature_importances(rf_model, X.columns)

if __name__ == "__main__":
    main()
