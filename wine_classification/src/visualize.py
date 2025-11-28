from typing import List
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def plot_feature_importances(
    model: RandomForestClassifier,
    feature_names: List[str],
    top_n: int,
    title: str = "RF Feature Importances",
):
    """
    Bar plot of feature importances.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    if top_n is not None:
        indices = indices[:top_n]

    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_features)), sorted_importances)
    plt.xticks(range(len(sorted_features)), sorted_features, rotation=45, ha="right")
    plt.ylabel("Feature Importance")
    plt.title(title)
    plt.tight_layout()
    plt.show()

