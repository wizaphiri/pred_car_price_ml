import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# page configuration
st.set_page_config(
    page_title="Auto Valuation Engine",
    page_icon="🚗",
    layout="centered"
)


# custom styling

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load trained model

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "car_price_model.joblib"
)


@st.cache_resource
def load_model():
    """Load the trained machine learning pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# Clear function

def clear_inputs():
    """Reset all input fields and prediction."""

    st.session_state.make = "Toyota"
    st.session_state.model_name = ""
    st.session_state.year = 2022
    st.session_state.engine = "1.5L"
    st.session_state.transmission = "Automatic"
    st.session_state.mileage = 50000
    st.session_state.prediction = None


# Application

st.title("🚗 Auto Valuation Engine")

st.write(
    "Estimate the market price of a used vehicle "
    "using a trained machine learning model."
)

st.divider()

# Load model

try:

    model = load_model()

except FileNotFoundError as e:

    st.error(str(e))

    st.info(
        "Please train the model first by running "
        "`src/modelx.py`."
    )

    st.stop()


# Initialize session state

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "make" not in st.session_state:
    st.session_state.make = "Toyota"

if "model_name" not in st.session_state:
    st.session_state.model_name = ""

if "year" not in st.session_state:
    st.session_state.year = 2022

if "engine" not in st.session_state:
    st.session_state.engine = "1.5L"

if "transmission" not in st.session_state:
    st.session_state.transmission = "Automatic"

if "mileage" not in st.session_state:
    st.session_state.mileage = 50000


# Vehicle inputs params

st.subheader("Vehicle Details")

col1, col2 = st.columns(2)


with col1:

    make = st.selectbox(
        "Make",
        [
            "Toyota",
            "Honda",
            "Nissan",
            "Mazda",
            "Ford"
        ],
        key="make"
    )

    model_name = st.text_input(
        "Model",
        placeholder="e.g. Corolla",
        key="model_name"
    )

    year = st.number_input(
        "Year",
        min_value=2000,
        max_value=2026,
        value=2022,
        step=1,
        key="year"
    )


with col2:

    engine = st.selectbox(
        "Engine",
        [
            "1.5L",
            "1.8L",
            "2.0L",
            "2.4L",
            "2.5L",
            "3.0L",
            "3.5L"
        ],
        key="engine"
    )

    transmission = st.selectbox(
        "Transmission",
        [
            "Automatic",
            "Manual"
        ],
        key="transmission"
    )

    mileage = st.number_input(
        "Mileage",
        min_value=0,
        max_value=300000,
        value=50000,
        step=1000,
        key="mileage"
    )


# prediction buttons

st.divider()

col1, col2 = st.columns([3, 1])


with col1:

    predict_button = st.button(
        "Estimate Vehicle Price",
        type="primary",
        use_container_width=True
    )


with col2:

    st.button(
        "Clear",
        use_container_width=True,
        on_click=clear_inputs
    )


# prediction

if predict_button:

    if not model_name.strip():

        st.session_state.prediction = None

        st.warning(
            "Please enter the vehicle model."
        )

    else:

        vehicle = pd.DataFrame({
            "vehicle.make": [make],
            "vehicle.model": [model_name.strip()],
            "vehicle.year": [year],
            "vehicle.engine": [engine],
            "vehicle.transmission": [transmission],
            "retailListing.miles": [mileage]
        })

        try:

            prediction = model.predict(
                vehicle
            )[0]

            st.session_state.prediction = prediction

        except Exception as e:

            st.session_state.prediction = None

            st.error(
                f"Unable to generate prediction: {e}"
            )


# display prediction

if st.session_state.prediction is not None:

    st.subheader(
        "Estimated Market Price"
    )

    st.success(
        f"${st.session_state.prediction:,.0f}"
    )

    st.caption(
        "Estimated market value — powered by "
        "machine learning and patterns learned "
        "from vehicle listings."
    )
