import streamlit as st
import torch
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from model_definition import PitstopModel

# Configure page
st.set_page_config(
    page_title="F1 Pitstop Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Import the model class


@st.cache_resource
def load_model():
    """Load the trained model with automatic feature detection"""
    try:
        model_path = "./model/pitstopmodel.pth"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location="cpu")

            # Detect input size from the first layer
            if "network.0.weight" in checkpoint:
                input_size = checkpoint["network.0.weight"].shape[1]
            elif "model_state_dict" in checkpoint:
                input_size = checkpoint["model_state_dict"]["network.0.weight"].shape[1]
            else:
                input_size = 9  # default

            model = PitstopModel(input_size)

            # Load state dict
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            model.eval()

            # Load scaler
            from sklearn.preprocessing import MinMaxScaler

            scaler_path = "model/scaler.pkl"
            try:
                import joblib

                scaler = joblib.load(scaler_path)
            except Exception:
                # Create fallback scaler with typical F1 ranges
                scaler = MinMaxScaler()
                dummy_data = np.array(
                    [
                        [0, 1, 0, 0, 0, 0, 1, 60.0, 0],  # min values
                        [42, 70, 7, 50, 2, 1, 20, 120.0, 35],  # max values
                    ]
                )
                scaler.fit(dummy_data)

            return model, scaler, True, input_size
        else:
            return None, None, False, 0
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, False, 0


def test_model(model, scaler):
    """Test if the model works with sample data"""
    try:
        # Create sample input that matches expected format
        sample_data = [16, 25, 4, 15, 0, 0, 5, 85.5, 2]  # Regular values

        # Use the predict_pitstop function to test
        probability, prediction = predict_pitstop(model, scaler, sample_data, 0.3)

        return True, probability
    except Exception as e:
        return False, str(e)


def predict_pitstop(model, scaler, input_data, threshold=0.3):
    """Make pitstop prediction"""
    try:
        # Define feature names to match the scaler
        feature_names = [
            "driver",
            "lap_number",
            "compound",
            "tyre_age",
            "track_status",
            "is_pit",
            "position",
            "lap_time",
            "track",
        ]

        # Convert input to DataFrame to avoid sklearn warning
        input_df = pd.DataFrame([input_data], columns=feature_names)

        # Apply scaler normalization
        input_scaled = scaler.transform(input_df)

        # Apply feature weights (same as training)
        input_scaled_weighted = input_scaled.copy()
        # Apply enhanced feature weights AFTER scaling
        input_scaled_weighted[0][3] *= 2.5  # tyre_age
        input_scaled_weighted[0][1] *= 2.0  # lap_number
        input_scaled_weighted[0][6] *= 1.2  # position

        # Convert to tensor
        input_tensor = torch.tensor(input_scaled_weighted, dtype=torch.float32)

        with torch.no_grad():
            output = model(input_tensor)
            probability = torch.sigmoid(output).item()
            prediction = 1 if probability > threshold else 0

        return probability, prediction
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return 0.0, 0


@st.cache_data
def load_sample_data():
    """Load sample data for visualization"""
    try:
        df = pd.read_csv("f1_pitstop_dataset_processed.csv")
        return df.head(1000)  # Return first 1000 rows for performance
    except:
        return None


def main():
    # Header
    st.title("🏎️ F1 Pitstop Prediction Model")
    st.markdown("---")

    # Define mappings for categorical variables
    compound_mapping = {
        0: "HARD",
        1: "HYPERSOFT",
        2: "INTERMEDIATE",
        3: "MEDIUM",
        4: "SOFT",
        5: "SUPERSOFT",
        6: "ULTRASOFT",
        7: "WET",
    }

    driver_mapping = {
        0: "AIT",
        1: "ALB",
        2: "ALO",
        3: "ANT",
        4: "BEA",
        5: "BOR",
        6: "BOT",
        7: "COL",
        8: "DEV",
        9: "DOO",
        10: "ERI",
        11: "FIT",
        12: "GAS",
        13: "GIO",
        14: "GRO",
        15: "HAD",
        16: "HAM",
        17: "HAR",
        18: "HUL",
        19: "KUB",
        20: "KVY",
        21: "LAT",
        22: "LAW",
        23: "LEC",
        24: "MAG",
        25: "MAZ",
        26: "MSC",
        27: "NOR",
        28: "OCO",
        29: "PER",
        30: "PIA",
        31: "RAI",
        32: "RIC",
        33: "RUS",
        34: "SAI",
        35: "SAR",
        36: "SIR",
        37: "STR",
        38: "TSU",
        39: "VAN",
        40: "VER",
        41: "VET",
        42: "ZHO",
    }

    track_mapping = {
        0: "70th Anniversary Grand Prix",
        1: "Abu Dhabi Grand Prix",
        2: "Australian Grand Prix",
        3: "Austrian Grand Prix",
        4: "Azerbaijan Grand Prix",
        5: "Bahrain Grand Prix",
        6: "Belgian Grand Prix",
        7: "Brazilian Grand Prix",
        8: "British Grand Prix",
        9: "Canadian Grand Prix",
        10: "Chinese Grand Prix",
        11: "Dutch Grand Prix",
        12: "Eifel Grand Prix",
        13: "Emilia Romagna Grand Prix",
        14: "French Grand Prix",
        15: "German Grand Prix",
        16: "Hungarian Grand Prix",
        17: "Italian Grand Prix",
        18: "Japanese Grand Prix",
        19: "Las Vegas Grand Prix",
        20: "Mexican Grand Prix",
        21: "Mexico City Grand Prix",
        22: "Miami Grand Prix",
        23: "Monaco Grand Prix",
        24: "Portuguese Grand Prix",
        25: "Qatar Grand Prix",
        26: "Russian Grand Prix",
        27: "Sakhir Grand Prix",
        28: "Saudi Arabian Grand Prix",
        29: "Singapore Grand Prix",
        30: "Spanish Grand Prix",
        31: "Styrian Grand Prix",
        32: "São Paulo Grand Prix",
        33: "Turkish Grand Prix",
        34: "Tuscan Grand Prix",
        35: "United States Grand Prix",
    }

    # Create reverse mappings for lookups
    compound_reverse = {v: k for k, v in compound_mapping.items()}
    driver_reverse = {v: k for k, v in driver_mapping.items()}
    track_reverse = {v: k for k, v in track_mapping.items()}

    # Load model
    model, scaler, model_loaded, input_size = load_model()

    if not model_loaded:
        st.error(
            "❌ Could not load the trained model. Please ensure 'model/pitstopmodel.pth' exists."
        )
        st.stop()

    # Test model functionality
    model_works, test_result = test_model(model, scaler)
    if not model_works:
        st.error(f"❌ Model test failed: {test_result}")
        st.stop()

    # Sidebar for input
    st.sidebar.header("🔧 Race Parameters")
    st.sidebar.markdown("Enter the current race conditions:")

    # Input fields based on the dataset columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Driver & Race Info")
        driver_name = st.selectbox(
            "Driver",
            options=list(driver_mapping.values()),
            index=16,  # Default to HAM
            help="Select the driver",
        )
        driver_code = driver_reverse[driver_name]

        lap_number = st.slider("Lap Number", min_value=1, max_value=70, value=25)
        position = st.slider("Current Position", min_value=1, max_value=20, value=5)

        track_name = st.selectbox(
            "Track",
            options=list(track_mapping.values()),
            index=2,  # Default to Australian Grand Prix
            help="Select the race track",
        )
        track_code = track_reverse[track_name]

    with col2:
        st.subheader("🛞 Tyre & Strategy")
        compound_name = st.selectbox(
            "Tyre Compound",
            options=list(compound_mapping.values()),
            index=4,  # Default to SOFT
            help="Select tyre compound type",
        )
        compound_code = compound_reverse[compound_name]

        tyre_age_laps = st.slider(
            "Tyre Age (laps)",
            min_value=0,
            max_value=50,
            value=15,
            help="Number of laps on current tyres",
        )
        # Convert to normalized format using exact same method as prepare_dataset.py
        tyre_age_normalized = (
            tyre_age_laps / 77.0
        )  # Divide by max tyre_age from training data

        track_status_options = ["Green", "Yellow", "Red"]
        track_status_name = st.selectbox(
            "Track Status",
            options=track_status_options,
            index=0,
            help="Current track conditions",
        )
        # Map to training data values: Green=1.0, Yellow=12.0, Red=4.0 (based on training data)
        track_status_mapping = {"Green": 1.0, "Yellow": 12.0, "Red": 4.0}
        track_status_code = track_status_mapping[track_status_name]

        is_pit_options = ["No", "Yes"]
        is_pit_name = st.selectbox("Currently in Pit", options=is_pit_options, index=0)
        is_pit_code = is_pit_options.index(is_pit_name)

        lap_time_seconds = st.slider(
            "Lap Time (seconds)",
            min_value=60.0,
            max_value=120.0,
            value=85.5,
            step=0.1,
            help="Current lap time in seconds",
        )
        # Convert to normalized format using exact same method as prepare_dataset.py
        lap_time_normalized = (
            lap_time_seconds / 2526.25
        )  # Divide by max lap_time from training data

    # Set optimal threshold as constant
    threshold = 0.3  # Optimal threshold from model evaluation

    # Prepare input data in the same order as training
    # Features: [driver, lap_number, compound, tyre_age, track_status, is_pit, position, lap_time, track]
    base_features = [
        float(driver_code),
        float(lap_number),
        float(compound_code),
        float(tyre_age_normalized),  # Already normalized 0-1 as expected by model
        float(track_status_code),
        float(is_pit_code),
        float(position),
        float(lap_time_normalized),  # Already normalized 0-1 as expected by model
    ]

    # Add track feature if model expects 9+ features
    if input_size >= 9:
        base_features.append(float(track_code))

    # Pad or trim to match exact model input size
    input_data = base_features[:input_size]
    while len(input_data) < input_size:
        input_data.append(0.0)  # Pad with zeros if needed

    # Make prediction
    if st.button("🚀 Predict Pitstop", type="primary"):
        probability, prediction = predict_pitstop(model, scaler, input_data, threshold)

        # Display results
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Pit Probability", f"{probability:.1%}")
            st.write("Pit Prediction", prediction)

        with col2:
            decision = "🔴 PIT NOW!" if prediction == 1 else "🟢 STAY OUT"
            st.metric("Decision", decision)

        with col3:
            confidence = (
                "High"
                if abs(probability - 0.3) > 0.2
                else "Medium"
                if abs(probability - 0.3) > 0.1
                else "Low"
            )
            st.metric("Confidence", confidence)

        # Probability gauge
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=probability * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Pit Probability (%)"},
                delta={"reference": 30},  # Updated to show difference from threshold
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": "red" if prediction == 1 else "green"},
                    "steps": [
                        {"range": [0, 15], "color": "lightgreen"},
                        {"range": [15, 30], "color": "yellow"},
                        {"range": [30, 60], "color": "orange"},
                        {"range": [60, 100], "color": "red"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": 30,  # Updated threshold line
                    },
                },
            )
        )

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Strategic advice
        st.subheader("📋 Strategic Advice")
        if prediction == 1:
            st.error(f"**RECOMMENDATION: PIT NOW** (Probability: {probability:.1%})")
            st.write(
                f"Based on current conditions for **{driver_name}** at **{track_name}**, this is an optimal pit window."
            )
            if tyre_age_laps > 25:
                st.write(
                    f"🛞 **High tyre degradation** on {compound_name} tyres ({tyre_age_laps} laps) - pit stop recommended for fresh tyres"
                )
            if track_status_name != "Green":
                st.write(
                    f"⚠️ **Track conditions ({track_status_name})** favor a pit stop opportunity"
                )
            if lap_time_seconds > 90.0:
                st.write(
                    f"⏱️ **Slow lap times** ({lap_time_seconds:.1f}s) indicate tyre degradation"
                )
        else:
            st.success(f"**RECOMMENDATION: STAY OUT** (Probability: {probability:.1%})")
            st.write(
                f"Current conditions for **{driver_name}** at **{track_name}** suggest staying on track is optimal."
            )
            if tyre_age_laps < 10:
                st.write(
                    f"🛞 **Fresh {compound_name} tyres** ({tyre_age_laps} laps) - continue current stint"
                )
            if position <= 3:
                st.write("🏆 **Strong position** - maintain track position")
            if lap_time_seconds < 85.0:
                st.write(
                    f"⏱️ **Good pace** ({lap_time_seconds:.1f}s) - tyres still performing well"
                )

    # Model information
    st.markdown("---")
    st.subheader("ℹ️ Model Information")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"""
        **Model Architecture:**
        - Input Features: {input_size}
        - Hidden Layers: [128, 64, 32, 16]
        - Output: Single probability
        - BatchNorm + Dropout
        """)

    with col2:
        st.info("""
        **Model Performance:**
        - Optimal Threshold: 30%
        - Enhanced Feature Weighting
        - Trained on F1 2017-2025 data
        - Real-time predictions
        """)

    with col3:
        st.info("""
        **Key Features:**
        - Tyre age (weighted 2.5x)
        - Lap number (weighted 2.0x)
        - Track position (weighted 1.2x)
        - Track conditions & compound
        """)

    # Sample data exploration
    sample_data = load_sample_data()
    if sample_data is not None:
        st.markdown("---")
        st.subheader("📈 Sample Race Data")

        # Show distribution of pit stops
        pit_counts = sample_data["will_pit_next_lap"].value_counts()
        fig = px.pie(
            values=pit_counts.values,
            names=["No Pit", "Pit"],
            title="Distribution of Pit Decisions in Sample Data",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show sample data
        if st.checkbox("Show Sample Data"):
            st.dataframe(sample_data)

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center'>
        <p>🏁 F1 Pitstop Prediction Model | Built with PyTorch & Streamlit</p>
        <p><small>This model is for educational purposes. Actual F1 strategy requires additional real-time data.</small></p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
