import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import RED_WINE_CSV
from src.data import load_wine_data
from src.features import engineer_features, split_train_val_test, apply_SMOTE
from src.model_xgb import train_xgboost, evaluate_xgb_on_test
from src.visualize import plot_feature_importances


def main() -> None:
    df = load_wine_data(kind = 'red')

    X, y = engineer_features(df)
    print("\nTarget distribution (0/1):")
    print(y.value_counts(normalize=True))

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y)
    print("\nTrain/Val/Test dataset sizes:")
    print(len(X_train), len(X_val), len(X_test))

    # X_train, y_train = apply_SMOTE(X_train, y_train)

    xgb_model = train_xgboost(
        X_train, y_train,
        X_val, y_val,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05
    )

    # Test
    metrics = evaluate_xgb_on_test(xgb_model, X_test, y_test)

    # Feature importances
    plot_feature_importances(
        xgb_model,
        feature_names=X.columns.tolist(),
        title="XGBoost Feature Importances"
    )


if __name__ == "__main__":
    main()
