import json
import os
import joblib
import pandas as pd


PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "random_forest_soh_model.joblib"
)

METADATA_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "random_forest_soh_model_metadata.json"
)


# Load model
soh_model = joblib.load(MODEL_PATH)


# Load metadata
with open(METADATA_PATH, "r") as f:
    soh_metadata = json.load(f)


SOH_FEATURES = soh_metadata["features"]


def predict_soh(input_data):
    """
    Predict battery SOH using the saved Random Forest model.

    Parameters
    ----------
    input_data : dict or pandas.DataFrame
        Battery measurements containing all SOH model features.

    Returns
    -------
    float or pandas.Series
        Predicted SOH percentage.
    """

    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])

    elif isinstance(input_data, pd.DataFrame):
        input_df = input_data.copy()

    else:
        raise TypeError(
            "input_data must be a dictionary or pandas DataFrame."
        )

    missing_features = [
        feature
        for feature in SOH_FEATURES
        if feature not in input_df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required features: "
            + ", ".join(missing_features)
        )

    X_input = input_df[SOH_FEATURES].copy()

    if X_input.isna().any().any():
        missing_columns = (
            X_input.columns[
                X_input.isna().any()
            ].tolist()
        )

        raise ValueError(
            "Input contains missing values in: "
            + ", ".join(missing_columns)
        )

    predictions = soh_model.predict(X_input)

    return predictions
