
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from metrics_utils import compute_metrics


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATA_FILE = BASE_DIR / "data" / "dry_bean.csv"

TARGET_COL = "Class"
RANDOM_STATE = 42


def load_data():
    """Load the UCI Dry Bean dataset."""

    return pd.read_csv(DATA_FILE)


def build_models():
    """Create the five classifiers required for the assignment."""

    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            )),
        ]),

        "Decision Tree": Pipeline([
            ("classifier", DecisionTreeClassifier(
                max_depth=8,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
            )),
        ]),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(
                n_neighbors=7,
            )),
        ]),

        "Naive Bayes": Pipeline([
            ("classifier", GaussianNB()),
        ]),

        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                C=1.0,
                probability=True,
                random_state=RANDOM_STATE,
            )),
        ]),
    }


def main():
    """Train, evaluate and save all five classifiers."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()

    X = data.drop(columns=[TARGET_COL])
    y = data[TARGET_COL]

    print("Dataset shape:", data.shape)
    print("\nTarget distribution:")
    print(y.value_counts().sort_index())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("\nTraining instances:", len(X_train))
    print("Testing instances:", len(X_test))

    # Save a labelled test dataset for the Streamlit application.
    test_data = X_test.copy()
    test_data[TARGET_COL] = y_test
    test_data.to_csv(BASE_DIR / "test_data.csv", index=False)

    results = []

    for model_name, model in build_models().items():
        print(f"\nTraining {model_name}...")

        # The scaler, when used, is fitted only on training data.
        model.fit(X_train, y_train)

        file_name = model_name.lower().replace(" ", "_") + ".joblib"
        model_path = MODEL_DIR / file_name

        joblib.dump(model, model_path)

        metrics = compute_metrics(model, X_test, y_test)

        results.append({
            "ML Model Name": model_name,
            **metrics,
        })

        print("Saved model:", model_path.name)
        print({
            name: round(value, 4)
            for name, value in metrics.items()
        })

    results_df = pd.DataFrame(results).round(4)
    results_path = MODEL_DIR / "metrics.csv"
    results_df.to_csv(results_path, index=False)

    print("\nFinal model comparison:")
    print(results_df.to_string(index=False))

    print("\nTraining and evaluation completed.")
    print("Test data:", BASE_DIR / "test_data.csv")
    print("Metrics:", results_path)


if __name__ == "__main__":
    main()
