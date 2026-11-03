
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.metrics import classification_report, confusion_matrix

from metrics_utils import compute_metrics


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
SAMPLE_TEST_FILE = BASE_DIR / "test_data.csv"

TARGET_COL = "Class"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(
    page_title="Dry Bean Variety Classifier",
    page_icon="🌱",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    """Load a saved model by its display name."""

    model_path = MODEL_DIR / MODEL_FILES[model_name]

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Saved model file not found: {model_path.name}. "
            "Run python train_models.py first."
        )

    return joblib.load(model_path)


def read_test_csv(uploaded_file):
    """Read the uploaded CSV and return None if it cannot be parsed."""

    try:
        return pd.read_csv(uploaded_file)
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ):
        return None


def check_columns(df: pd.DataFrame, model) -> list:
    """Find required feature columns missing from the CSV."""

    required_features = list(model.feature_names_in_)

    return [
        column
        for column in required_features
        if column not in df.columns
    ]


def validate_test_data(df: pd.DataFrame, model):
    """Validate features and target labels before model evaluation."""

    if df.empty:
        raise ValueError("The CSV contains a header but no data rows.")

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"The CSV must contain a '{TARGET_COL}' column with true labels."
        )

    missing = check_columns(df, model)

    if missing:
        raise ValueError(
            "The CSV is missing required feature columns: "
            + ", ".join(missing)
        )

    feature_names = list(model.feature_names_in_)

    X_test = df[feature_names].copy()
    y_test = df[TARGET_COL].astype(str).str.strip().copy()

    try:
        X_test = X_test.apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as error:
        raise ValueError(
            "All feature values must be numeric."
        ) from error

    if X_test.isna().any().any():
        raise ValueError(
            "The CSV contains missing feature values. "
            "Please clean the data and upload it again."
        )

    if not np.isfinite(X_test.to_numpy(dtype=float)).all():
        raise ValueError(
            "Feature columns contain infinite or invalid numeric values."
        )

    model_classes = set(model.classes_)

    if not y_test.isin(model_classes).all():
        unknown = sorted(set(y_test) - model_classes)
        raise ValueError(
            "The target column contains labels the model was not trained "
            "on: " + ", ".join(unknown)
        )

    if y_test.nunique() < 2:
        raise ValueError(
            "The test data must contain at least two classes to calculate AUC."
        )

    return X_test, y_test


def plot_confusion_matrix(y_true, y_pred, labels):
    """Create the blue heatmap-style confusion matrix."""

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    figure, axis = plt.subplots(figsize=(5.5, 4.6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Greens",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=axis,
    )

    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Confusion matrix")
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")

    figure.tight_layout()

    return figure


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🌱 Control panel")
st.sidebar.write("Choose a model and give it some labelled test data.")

model_name = st.sidebar.selectbox(
    "Select classification model",
    list(MODEL_FILES.keys()),
)

uploaded = st.sidebar.file_uploader(
    "Upload TEST data (CSV)",
    type=["csv"],
    help="Must contain the 16 Dry Bean feature columns plus the 'Class' column.",
)

use_sample = False

if SAMPLE_TEST_FILE.is_file():
    use_sample = st.sidebar.checkbox(
        "Use the bundled sample test_data.csv",
        value=False,
    )

    st.sidebar.download_button(
        "Download sample test_data.csv",
        data=SAMPLE_TEST_FILE.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
    )

st.sidebar.caption(
    "Models are pre-trained. Nothing is retrained on your upload."
)


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("Dry Bean Variety Classifier")

st.write(
    "This app compares five classical machine-learning models on the "
    "**UCI Dry Bean Dataset**. Upload a labelled test CSV, pick a model, "
    "and see how well it tells apart the 7 registered bean varieties "
    "(Seker, Barbunya, Bombay, Cali, Horoz, Sira, Dermason) from shape "
    "and size measurements extracted from grain images."
)

if uploaded is not None:
    test_df = read_test_csv(uploaded)

    if test_df is None:
        st.error(
            "That file could not be read as a CSV. "
            "Please upload a valid, non-empty CSV file."
        )
        st.stop()

elif use_sample:
    try:
        test_df = pd.read_csv(SAMPLE_TEST_FILE)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        st.error(f"Could not read the sample test data: {error}")
        st.stop()

else:
    test_df = None

if test_df is None:
    st.info(
        "👈 Upload a test CSV (or tick the bundled sample) "
        "in the sidebar to begin."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Load selected model and validate the input
# ---------------------------------------------------------------------------
try:
    model = load_model(model_name)
    X_test, y_test = validate_test_data(test_df, model)

except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
    st.error(str(error))
    st.stop()

except Exception as error:
    st.error(f"Could not load the selected model: {error}")
    st.stop()


# ---------------------------------------------------------------------------
# 1. Uploaded data preview
# ---------------------------------------------------------------------------
st.subheader("1. Uploaded data preview")

column1, column2, column3 = st.columns(3)

column1.metric("Rows", len(test_df))
column2.metric("Features used", X_test.shape[1])
column3.metric("Classes in target", y_test.nunique())

st.dataframe(
    test_df.head(10),
    width="stretch",
)


# ---------------------------------------------------------------------------
# 2. Metrics for the selected model
# ---------------------------------------------------------------------------
st.subheader(f"2. Evaluation metrics - {model_name}")

try:
    scores = compute_metrics(model, X_test, y_test)

except ValueError as error:
    st.error(str(error))
    st.stop()

metric_columns = st.columns(6)

for column, (metric_name, value) in zip(
    metric_columns,
    scores.items(),
):
    column.metric(metric_name, f"{value:.4f}")


# ---------------------------------------------------------------------------
# 3. Confusion matrix and classification report
# ---------------------------------------------------------------------------
st.subheader(
    f"3. Confusion matrix and classification report - {model_name}"
)

predictions = model.predict(X_test)

left, right = st.columns([1, 1.3])

with left:
    figure = plot_confusion_matrix(
        y_test,
        predictions,
        list(model.classes_),
    )

    st.pyplot(figure)
    plt.close(figure)

with right:
    report = classification_report(
        y_test,
        predictions,
        labels=list(model.classes_),
        output_dict=True,
        zero_division=0,
    )

    st.dataframe(
        pd.DataFrame(report).transpose().round(4),
        width="stretch",
    )


# ---------------------------------------------------------------------------
# 4. Comparison across all five saved models
# ---------------------------------------------------------------------------
st.subheader("4. All five models on this test data")

comparison_rows = []

try:
    for name in MODEL_FILES:
        saved_model = load_model(name)

        expected_features = list(saved_model.feature_names_in_)

        if expected_features != list(X_test.columns):
            raise ValueError(
                f"{name} expects a different set or order of feature columns."
            )

        model_scores = compute_metrics(
            saved_model,
            X_test,
            y_test,
        )

        comparison_rows.append({
            "ML Model Name": name,
            **model_scores,
        })

except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
    st.error(f"Could not compare all models: {error}")
    st.stop()

comparison = (
    pd.DataFrame(comparison_rows)
    .set_index("ML Model Name")
    .round(4)
)

st.dataframe(
    comparison,
    width="stretch",
)

best_model = comparison["MCC"].idxmax()

st.success(
    f"Highest MCC on this data: **{best_model}** "
    f"({comparison.loc[best_model, 'MCC']:.4f})"
)

st.bar_chart(comparison[["Accuracy", "F1", "MCC"]])
