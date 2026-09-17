# src/preprocessing.py
# data-acquisition-auto-dev-api
from datetime import datetime
from data_acquisition import get_listings

import pandas as pd

def select_features(df):
    """Select candidate features for the ML model."""

    selected_columns = [
        "vehicle.make",
        "vehicle.model",
        "vehicle.year",
        "vehicle.engine",
        "vehicle.transmission",
        "retailListing.miles",
        "retailListing.price",
        "history.accidentCount",
        "history.oneOwner",
        "history.ownerCount",
        "history.personalUse",
        "history.usageType"
    ]

    available_columns = [
        column
        for column in selected_columns
        if column in df.columns
    ]

    return df[available_columns].copy()

def remove_high_missing_columns(df, threshold=0.70):
    """Drop columns where the percentage of missing values exceeds threshold."""

    missing_percentage = df.isna().mean()

    columns_to_drop = missing_percentage[
        missing_percentage > threshold
    ].index

    return df.drop(columns=columns_to_drop)


def clean_data(df):
    """Perform general data cleaning."""

    df = df.copy()

    # Remove duplicate records
    df = df.drop_duplicates()

    return df


def engineer_features(df):
    """Create features required for modelling."""

    df = df.copy()

    # Vehicle age
    df["vehicle_age"] = 2026 - df["vehicle.year"]

    return df


def encode_features(df):
    """Encode categorical variables."""

    df = df.copy()

    # Encoding will be implemented once
    # we decide which categorical variables
    # are going into the model.

    return df


def scale_features(df):
    """Scale numerical features."""

    df = df.copy()

    # Scaling will be implemented using
    # a scikit-learn pipeline.

    return df


def split_data(df):
    """Split data into training and testing sets."""

    # We'll implement this after defining
    # X and y.

    return df


def preprocess_data(df):
    """Run the complete preprocessing pipeline."""

    df = select_features(df)
    df = remove_high_missing_columns(df)
    df = clean_data(df)
    df = engineer_features(df)
    df = encode_features(df)
    df = scale_features(df)
    # df = split_data(df)

    return df


if __name__ == "__main__":

    df = get_listings()

    processed_df = preprocess_data(df)

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {processed_df.shape[0]:,} rows and {processed_df.shape[1]:,} columns extracted from API and processed")
    # print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}]", processed_df.columns.tolist())