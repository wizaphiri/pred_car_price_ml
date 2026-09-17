from data_acquisition import get_listings
from data_preprocessing import preprocess_data

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def split_data(df):
    """Split the processed data into training and testing sets."""

    X = df.drop(columns=["retailListing.price"])
    y = df["retailListing.price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


def build_preprocessor(X_train):
    """Create preprocessing pipeline for numerical and categorical features."""

    numerical_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_features = X_train.select_dtypes(
        include=["str", "object"]
    ).columns

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return preprocessor


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on the test data."""

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(
        y_test,
        y_pred,
        squared=False
    )
    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation")
    print(f"MAE:  ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R²:   {r2:.3f}")

    return mae, rmse, r2


if __name__ == "__main__":

    # Acquire data
    df = get_listings()

    # Preprocess data
    processed_df = preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(processed_df)

    print(f"Training data: {X_train.shape}")
    print(f"Testing data: {X_test.shape}")

    # -------------------------
    # Baseline
    # -------------------------

    y_mean = y_train.mean()
    y_pred_baseline = [y_mean] * len(y_test)

    baseline_mae = mean_absolute_error(
        y_test,
        y_pred_baseline
    )

    print("\nBaseline")
    print(f"Mean car price: ${y_mean:,.2f}")
    print(f"Baseline MAE:   ${baseline_mae:,.2f}")

    # -------------------------
    # Build Model
    # -------------------------

    preprocessor = build_preprocessor(X_train)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", Ridge())
    ])

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    evaluate_model(model, X_test, y_test)