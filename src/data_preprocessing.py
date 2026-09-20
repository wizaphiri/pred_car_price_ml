# src/data_preprocessing.py

from datetime import datetime
from pathlib import Path

import pandas as pd

from data_acquisition import get_listings


def get_data_path():
    """Return the path to the raw vehicle dataset."""

    return (
        Path(__file__).resolve().parent.parent
        / "data"
        / "raw"
        / "auto_dev_listings.csv"
    )


def load_data():
    """
    Load raw vehicle listings from CSV.

    If the CSV does not exist, acquire the data
    from Auto.dev using data_acquisition.py.
    """

    input_path = get_data_path()

    try:

        # Load existing CSV
        if input_path.exists():

            print("\nLoading existing raw dataset...")

            df = pd.read_csv(input_path)

            if df.empty:
                print("\nRaw dataset exists but contains no records.")
                print("Attempting to acquire fresh data...")

            else:
                print(
                    f"Loaded {len(df):,} records from CSV."
                )

                return df


        # CSV does not exist or empty - api call
        print("\nRaw dataset not found.")
        print(f"Expected file: {input_path}")

        print(
            "\nAcquiring vehicle listings from Auto.dev..."
        )

        df = get_listings()

        # Check whether API returned data
        if df.empty:

            print(
                "\nNo data was retrieved from Auto.dev."
            )

            print(
                "Preprocessing cannot continue until "
                "vehicle data is available."
            )

            return pd.DataFrame()

        # Save acquired data
        input_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df.to_csv(
            input_path,
            index=False
        )

        print(
            f"\nSaved {len(df):,} records to:"
        )

        print(input_path)

        return df

    except pd.errors.EmptyDataError:

        print(
            "\nThe raw dataset file is empty."
        )

        return pd.DataFrame()

    except pd.errors.ParserError:

        print(
            "\nUnable to read the raw dataset."
        )

        print(
            "The CSV file may be corrupted or incorrectly formatted."
        )

        return pd.DataFrame()

    except Exception as e:

        print(
            f"\nUnexpected error loading/acquiring data: {e}"
        )

        return pd.DataFrame()


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
    """Drop columns where missing values exceed the threshold."""

    missing_percentage = df.isna().mean()

    columns_to_drop = missing_percentage[
        missing_percentage > threshold
    ].index

    if len(columns_to_drop) > 0:

        print(
            "\nColumns removed due to high missing values:"
        )

        for column in columns_to_drop:

            print(
                f"- {column}: "
                f"{missing_percentage[column]:.1%} missing"
            )

    return df.drop(columns=columns_to_drop)


def clean_data(df):
    """Perform general data cleaning."""

    df = df.copy()

    initial_rows = len(df)

    df = df.drop_duplicates()

    duplicates_removed = initial_rows - len(df)

    print(
        f"\nDuplicates removed: "
        f"{duplicates_removed:,}"
    )

    return df


def engineer_features(df):
    """Create features required for modelling."""

    df = df.copy()

    current_year = datetime.now().year

    if "vehicle.year" in df.columns:

        df["vehicle_age"] = (
            current_year - df["vehicle.year"]
        )

    return df


def encode_features(df):
    """Encode categorical variables."""

    df = df.copy()

    # Encoding will be implemented using
    # a scikit-learn pipeline during modelling.

    return df


def scale_features(df):
    """Scale numerical features."""

    df = df.copy()

    # Scaling will be implemented using a scikit-learn pipeline during modelling.

    return df


def preprocess_data(df):
    """Run the complete preprocessing pipeline."""

    df = select_features(df)

    df = remove_high_missing_columns(df)

    df = clean_data(df)

    df = engineer_features(df)

    df = encode_features(df)

    df = scale_features(df)

    return df


if __name__ == "__main__":

    # Load existing data / acquire new data
    df = load_data()

    # Stop if no data is available
    if df.empty:

        print(
            "\nPreprocessing stopped because "
            "no data is available."
        )

    else:

        # Run preprocessing
        processed_df = preprocess_data(df)

        print(
            f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] "
            f"{processed_df.shape[0]:,} rows and "
            f"{processed_df.shape[1]:,} columns "
            f"loaded and processed."
        )

        print("\nProcessed columns:")

        for column in processed_df.columns:

            print(f"- {column}")
