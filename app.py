
from pathlib import Path
from datetime import date
import lzma
import shutil
import tempfile
import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Freight Rate Prediction",
    page_icon="🚚",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent

# Exact compressed filename uploaded to GitHub
COMPRESSED_MODEL_PATH = BASE_DIR / "freight_rate_model .pkl"

DEFAULT_FEATURES = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "year",
    "month_num",
    "day",
    "day_of_week"
]


# --------------------------------------------------
# 2. Load and extract model
# --------------------------------------------------

@st.cache_resource
def load_model():

    if not COMPRESSED_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Compressed model not found: {COMPRESSED_MODEL_PATH}. "
            "Check the filename in your GitHub repository."
        )

    temp_model_path = None

    try:
        # Extract compressed model into a temporary file
        with tempfile.NamedTemporaryFile(
            suffix=".pkl",
            delete=False
        ) as temp_file:
            temp_model_path = Path(temp_file.name)

        with lzma.open(
            COMPRESSED_MODEL_PATH,
            "rb"
        ) as compressed_file:

            with open(temp_model_path, "wb") as output_file:
                shutil.copyfileobj(
                    compressed_file,
                    output_file
                )

        # Load extracted Joblib artifact
        artifact = joblib.load(temp_model_path)

        # Support different saved artifact formats
        if isinstance(artifact, dict):

            model = artifact.get(
                "model",
                artifact.get("final_model")
            )

            features = artifact.get(
                "features",
                DEFAULT_FEATURES
            )

        elif isinstance(artifact, (tuple, list)):

            model = artifact[0]

            features = (
                artifact[1]
                if len(artifact) > 1
                else DEFAULT_FEATURES
            )

        else:

            model = artifact
            features = DEFAULT_FEATURES

        if model is None:
            raise ValueError(
                "No trained model was found in the artifact."
            )

        return model, list(features)

    finally:
        # Remove temporary extracted file
        if temp_model_path and temp_model_path.exists():
            temp_model_path.unlink()


# --------------------------------------------------
# 3. Load model safely
# --------------------------------------------------

try:

    model, features = load_model()

except Exception as error:

    st.error("Unable to load the model.")

    st.code(
        f"Error type: {type(error).__name__}\n"
        f"Error details: {repr(error)}"
    )

    st.stop()


# --------------------------------------------------
# 4. Application heading
# --------------------------------------------------

st.title("🚚 Freight Rate Prediction")

st.write(
    "Predict freight transportation rates using "
    "a trained Random Forest Machine Learning model."
)

st.divider()


# --------------------------------------------------
# 5. Input form
# --------------------------------------------------

st.subheader("Enter Load Details")

with st.form("freight_prediction_form"):

    pickup = st.text_input(
        "Pickup Location",
        placeholder="Example: Lexington"
    )

    delivery = st.text_input(
        "Delivery Location",
        placeholder="Example: Fort Wayne"
    )

    distance = st.number_input(
        "Distance (miles)",
        min_value=1,
        max_value=10000,
        value=360
    )

    equipment = st.selectbox(
        "Equipment Type",
        ["Dry Van", "Reefer", "Flatbed"]
    )

    weight = st.number_input(
        "Load Weight (lbs)",
        min_value=1,
        max_value=100000,
        value=32000
    )

    selected_date = st.date_input(
        "Pickup Date",
        value=date.today()
    )

    submitted = st.form_submit_button(
        "Predict Freight Rate",
        use_container_width=True
    )


# --------------------------------------------------
# 6. Make prediction
# --------------------------------------------------

if submitted:

    if not pickup.strip() or not delivery.strip():

        st.warning(
            "Please enter both pickup and delivery locations."
        )

    else:

        input_data = pd.DataFrame([{
            "pickup": pickup.strip(),
            "delivery": delivery.strip(),
            "distance": distance,
            "equipment": equipment,
            "weight": weight,
            "year": selected_date.year,
            "month_num": selected_date.month,
            "day": selected_date.day,
            "day_of_week": selected_date.weekday()
        }])

        try:

            # Match training feature order
            input_data = input_data[features]

            prediction = float(
                model.predict(input_data)[0]
            )

            if not pd.notna(prediction) or prediction <= 0:

                st.error(
                    "The model returned an invalid prediction."
                )

            else:

                st.success(
                    "Prediction completed successfully!"
                )

                st.metric(
                    label="Predicted Freight Rate",
                    value=f"${prediction:,.2f}"
                )

                st.caption(
                    "This is an estimated freight rate "
                    "generated by your trained model."
                )

        except Exception as error:

            st.error("Prediction failed.")

            st.code(
                f"{type(error).__name__}: {repr(error)}"
            )


# --------------------------------------------------
# 7. Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Freight Rate Prediction | "
    "Random Forest Regressor | "
    "Machine Learning Project"
)
