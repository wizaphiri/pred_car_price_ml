# iter over multiple model, eveluate, select best performing and serialize

# src/modelx.py

from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import ElasticNet, Ridge
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_preprocessing import load_data, preprocess_data


# Config

FEATURES = [
    "vehicle.make",
    "vehicle.model",
    "vehicle.year",
    "vehicle.engine",
    "vehicle.transmission",
    "retailListing.miles"
]

TARGET = "retailListing.price"



# Train / Test Split
def split_data(df):
    """Select model features and target, then split data."""

    X = df[FEATURES].copy()

    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    return X_train, X_test, y_train, y_test



# pre-processing pipeline

def build_preprocessor(X_train):
    """Build preprocessing pipeline for numerical and categorical features."""

    numerical_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_features = X_train.select_dtypes(
        include=["object", "str"]
    ).columns

    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "num",
            numerical_pipeline,
            numerical_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ])

    return preprocessor


# model definitions

def get_models():
    """Return regression models for comparison."""

    models = {

        "Ridge Regression": Ridge(),

        "ElasticNet": ElasticNet(
            alpha=0.1,
            l1_ratio=0.5,
            max_iter=5000
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            random_state=42
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    }

    return models


# model evaluation 

def evaluate_model(model, X_test, y_test):
    """Evaluate a regression model."""

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
    y_test,
    predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    return mae, rmse, r2


# model training (multiple models)

def train_models(X_train, X_test, y_train, y_test):
    """Train and evaluate all candidate models."""

    models = get_models()

    results = []

    trained_models = {}

    for name, estimator in models.items():

        print(
            f"\nTraining {name}..."
        )

        # Create preprocessing pipeline
        preprocessor = build_preprocessor(
            X_train
        )

        # Combine preprocessing + model
        pipeline = Pipeline([
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                estimator
            )
        ])

        # train
        pipeline.fit(
            X_train,
            y_train
        )

        # evaluate
        mae, rmse, r2 = evaluate_model(
            pipeline,
            X_test,
            y_test
        )

        print(
            f"MAE:  ${mae:,.2f}"
        )

        print(
            f"RMSE: ${rmse:,.2f}"
        )

        print(
            f"R²:   {r2:.3f}"
        )

        results.append({
            "model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        trained_models[name] = pipeline

    results_df = pd.DataFrame(
        results
    )

    return results_df, trained_models


# Save Best Model - joblib

def save_best_model(
    results_df,
    trained_models
):
    """Select and save the best-performing model."""

    # Select model with lowest error - MAE
    best_model_name = (
        results_df
        .sort_values("MAE")
        .iloc[0]["model"]
    )

    best_model = trained_models[
        best_model_name
    ]

    # Create models directory
    model_path = (
        Path(__file__).resolve().parent.parent
        / "models"
        / "car_price_model.joblib"
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save complete pipeline
    joblib.dump(
        best_model,
        model_path
    )

    return best_model_name, model_path


# Main

if __name__ == "__main__":

    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] "
        "Starting model training..."
    )

    # Load data

    df = load_data()

    if df.empty:

        print(
            "\nModel training stopped because "
            "no data is available."
        )

        exit()


    # pre-process data

    processed_df = preprocess_data(
        df
    )

    print(
        f"\nProcessed dataset:"
        f" {processed_df.shape[0]:,} rows × "
        f"{processed_df.shape[1]:,} columns"
    )

    # Check required features

    missing_features = [
        column
        for column in FEATURES + [TARGET]
        if column not in processed_df.columns
    ]

    if missing_features:

        print(
            "\nRequired columns missing:"
        )

        for column in missing_features:
            print(f"- {column}")

        exit()

    # train / test split

    X_train, X_test, y_train, y_test = split_data(
        processed_df
    )

    print(
        f"\nTraining data: {X_train.shape}"
    )

    print(
        f"Testing data:  {X_test.shape}"
    )

    # baseline
    mean_price = y_train.mean()

    baseline_predictions = [
        mean_price
    ] * len(y_test)

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions
    )

    print("\nBaseline")
    print("------------------------------")

    print(
        f"Mean car price: ${mean_price:,.2f}"
    )

    print(
        f"Baseline MAE:   ${baseline_mae:,.2f}"
    )

     # train models

    results_df, trained_models = train_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # Compare models

    results_df = results_df.sort_values(
        "MAE"
    )

    print("\nModel Comparison")
    print("------------------------------")

    print(
        results_df.to_string(
            index=False
        )
    )

    # select and save best model

    best_model_name, model_path = save_best_model(
        results_df,
        trained_models
    )

    print("\nBest Model")
    print("------------------------------")

    print(
        f"Selected model: {best_model_name}"
    )

    print(
        f"Saved to: {model_path}"
    )

    print(
        f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] "
        "Model training completed."
    )
