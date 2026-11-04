
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
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
RANDOM_STATE = 2026
TEST_SIZE = 0.20
CV_FOLDS = 5
CV_SCORING = "f1_macro"


def load_data():
    """Load the UCI Dry Bean dataset."""

    return pd.read_csv(DATA_FILE)


def build_search_spaces():
    """
    Create the five classifiers and the hyperparameter grid for each.

    Every grid is searched with stratified K-fold cross-validation on the
    training split only, so the test data is never used for tuning.
    """

    return {
        "Logistic Regression": (
            Pipeline([
                ("scaler", StandardScaler()),
                # newton-cg with a tight tolerance converges fully, so the
                # result is the same on different machines / library builds
                # (lbfgs stopped early and gave slightly different weights).
                ("classifier", LogisticRegression(
                    solver="newton-cg",
                    tol=1e-8,
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                )),
            ]),
            {
                "classifier__C": [0.01, 0.1, 1, 10, 100],
            },
        ),

        "Decision Tree": (
            Pipeline([
                ("classifier", DecisionTreeClassifier(
                    random_state=RANDOM_STATE,
                )),
            ]),
            {
                "classifier__criterion": ["gini", "entropy"],
                "classifier__max_depth": [6, 8, 10, 12, None],
                "classifier__min_samples_leaf": [1, 5, 10, 20],
            },
        ),

        "KNN": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier()),
            ]),
            {
                "classifier__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
                "classifier__weights": ["uniform", "distance"],
                "classifier__p": [1, 2],
            },
        ),

        # The shape features are skewed, so the grid also tries a
        # Yeo-Johnson power transform to make them closer to Gaussian.
        "Naive Bayes": (
            Pipeline([
                ("transform", "passthrough"),
                ("classifier", GaussianNB()),
            ]),
            {
                "transform": ["passthrough", PowerTransformer()],
                "classifier__var_smoothing": [1e-9, 1e-7, 1e-5],
            },
        ),

        "Support Vector Machine": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", SVC(
                    kernel="rbf",
                    probability=True,
                    random_state=RANDOM_STATE,
                )),
            ]),
            {
                "classifier__C": [1, 10, 100],
                "classifier__gamma": ["scale", 0.01, 0.1],
            },
        ),
    }


def describe_params(params):
    """Turn GridSearchCV best_params_ into a short readable string."""

    parts = []

    for key, value in params.items():
        name = key.replace("classifier__", "")

        if name == "transform":
            value = "none" if value == "passthrough" else "PowerTransformer"

        parts.append(f"{name}={value}")

    return ", ".join(parts)


def main():
    """Tune, train, evaluate and save all five classifiers."""

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
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("\nTraining instances:", len(X_train))
    print("Testing instances:", len(X_test))

    # Save a labelled test dataset for the Streamlit application.
    test_data = X_test.copy()
    test_data[TARGET_COL] = y_test
    test_data.to_csv(BASE_DIR / "test_data.csv", index=False)

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []
    tuning = []

    for model_name, (pipeline, grid) in build_search_spaces().items():
        print(f"\nTuning {model_name} ({CV_FOLDS}-fold CV, {CV_SCORING})...")

        search = GridSearchCV(
            pipeline,
            grid,
            scoring=CV_SCORING,
            cv=cv,
            n_jobs=-1,
        )

        # Scalers / transforms are fitted inside each CV fold, and the
        # final refit uses the full training split only.
        search.fit(X_train, y_train)
        model = search.best_estimator_

        file_name = model_name.lower().replace(" ", "_") + ".joblib"
        model_path = MODEL_DIR / file_name

        joblib.dump(model, model_path)

        metrics = compute_metrics(model, X_test, y_test)

        results.append({
            "ML Model Name": model_name,
            **metrics,
        })

        tuning.append({
            "ML Model Name": model_name,
            "Best Parameters": describe_params(search.best_params_),
            "CV F1 (macro)": round(search.best_score_, 4),
        })

        print("Best parameters:", describe_params(search.best_params_))
        print("Best CV F1 (macro):", round(search.best_score_, 4))
        print("Saved model:", model_path.name)
        print({
            name: round(value, 4)
            for name, value in metrics.items()
        })

    results_df = pd.DataFrame(results).round(4)
    results_path = MODEL_DIR / "metrics.csv"
    results_df.to_csv(results_path, index=False)

    tuning_df = pd.DataFrame(tuning)
    tuning_path = MODEL_DIR / "best_params.csv"
    tuning_df.to_csv(tuning_path, index=False)

    print("\nSelected hyperparameters:")
    print(tuning_df.to_string(index=False))

    print("\nFinal model comparison on the test data:")
    print(results_df.to_string(index=False))

    print("\nTraining and evaluation completed.")
    print("Test data:", BASE_DIR / "test_data.csv")
    print("Metrics:", results_path)
    print("Best parameters:", tuning_path)


if __name__ == "__main__":
    main()
