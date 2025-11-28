import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import RED_WINE_CSV, PROCESSED_DATA_DIR
from src.data import load_wine_data
from src.features import engineer_features, split_train_val_test
from src.model import train_random_forest, evaluate_on_test
from src.visualize import plot_feature_importances

def main():
    df = load_wine_data(kind="red")
    X, y = engineer_features(df)
    print("\nTarget distribution (0=not good, 1=good):")
    print(y.value_counts(normalize=True))

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y)
    print("\nTrain/Val/Test sizes:")
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # Train RF
    rf_model = train_random_forest(X_train, y_train, X_val, y_val)
    # Test 
    metrics = evaluate_on_test(rf_model, X_test, y_test)

    # Feature importances
    plot_feature_importances(rf_model, feature_names=X.columns.tolist())

if __name__ == "__main__":
    main()

