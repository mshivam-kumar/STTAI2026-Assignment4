"""
UrbanNest Analytics - Rent Prediction App
A Streamlit application for predicting house rental prices in Indian cities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="UrbanNest Analytics - Rent Prediction",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-message {
        font-size: 1.5rem;
        color: #006400;
        text-align: center;
        padding: 1rem;
        background-color: #d4edda;
        border-radius: 0.5rem;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🏠 UrbanNest Analytics</p>', unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #666;'>House Rent Prediction Engine</h3>", unsafe_allow_html=True)

st.markdown("---")

# Load model and encoders
@st.cache_resource
def load_model_and_encoders():
    """Load the trained model and encoders."""
    try:
        # Load model using gzip
        with gzip.open('models/best_rf_model.pkl.gz', 'rb') as f:
            model = pickle.load(f)

        # Load encoders
        with open('models/location_encoder.pkl', 'rb') as f:
            location_encoder = pickle.load(f)

        with open('models/city_encoder.pkl', 'rb') as f:
            city_encoder = pickle.load(f)

        with open('models/Status_encoder.pkl', 'rb') as f:
            status_encoder = pickle.load(f)

        with open('models/property_type_encoder.pkl', 'rb') as f:
            property_type_encoder = pickle.load(f)

        with open('models/feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)

        with open('models/categorical_mappings.pkl', 'rb') as f:
            categorical_mappings = pickle.load(f)

        return model, location_encoder, city_encoder, status_encoder, property_type_encoder, feature_columns, categorical_mappings
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, None, None, None


@st.cache_data
def load_city_location_mapping(all_locations):
    """Load city -> locations mapping from training data for dynamic UI filtering."""
    try:
        df = pd.read_csv('Dataset/train.csv', usecols=['city', 'location'])
        df = df.dropna(subset=['city', 'location']).copy()
        df['city'] = df['city'].astype(str).str.strip()
        df['location'] = df['location'].astype(str).str.strip()
        df = df[(df['city'] != '') & (df['location'] != '')]

        city_location_map = {
            city: sorted(group['location'].unique().tolist())
            for city, group in df.groupby('city')
        }
        return city_location_map
    except Exception:
        # Fallback: if dataset isn't available, keep global location list for all cities
        return {}

# Load model
model, location_encoder, city_encoder, status_encoder, property_type_encoder, feature_columns, categorical_mappings = load_model_and_encoders()

if model is None:
    st.error("Failed to load model. Please ensure all model files are present in the 'models/' directory.")
    st.stop()

# Get unique values for dropdowns
locations = sorted(categorical_mappings['location']['classes'])
cities = sorted(categorical_mappings['city']['classes'])
statuses = sorted(categorical_mappings['Status']['classes'])
property_types = sorted(categorical_mappings['property_type']['classes'])
city_location_map = load_city_location_mapping(locations)

# Create two columns for input form
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Location Information")

    city = st.selectbox(
        "City",
        options=cities,
        index=0,
        help="Select the city"
    )

    filtered_locations = city_location_map.get(city, locations)
    if not filtered_locations:
        filtered_locations = locations

    location = st.selectbox(
        "Location",
        options=filtered_locations,
        index=0,
        help="Select the area/neighborhood for the selected city"
    )

    latitude = st.number_input(
        "Latitude",
        value=19.0760,
        format="%.6f",
        help="Enter the latitude coordinate"
    )

    longitude = st.number_input(
        "Longitude",
        value=72.8777,
        format="%.6f",
        help="Enter the longitude coordinate"
    )

    status = st.selectbox(
        "Furnishing Status",
        options=statuses,
        index=0,
        help="Select the furnishing status"
    )

    property_type = st.selectbox(
        "Property Type",
        options=property_types,
        index=0,
        help="Select the type of property"
    )

with col2:
    st.subheader("🏗️ Property Details")

    num_bathrooms = st.number_input(
        "Number of Bathrooms",
        min_value=0,
        max_value=10,
        value=2,
        help="Total number of bathrooms"
    )

    num_balconies = st.number_input(
        "Number of Balconies",
        min_value=0,
        max_value=8,
        value=1,
        help="Total number of balconies"
    )

    is_negotiable = st.selectbox(
        "Is Rent Negotiable?",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        help="Whether the rent is negotiable"
    )

    security_deposit = st.number_input(
        "Security Deposit (INR)",
        min_value=0,
        max_value=11401010,
        value=0,
        step=1000,
        help="Security deposit amount in INR"
    )

    size_ft = st.number_input(
        "Size (sq ft)",
        min_value=150,
        max_value=14521,
        value=1000,
        help="Property size in square feet"
    )

    bhk = st.selectbox(
        "BHK/RK Type",
        options=[1, 0],
        format_func=lambda x: "BHK (Bedroom Hall Kitchen)" if x == 1 else "RK (Room Kitchen)",
        help="Type of dwelling (1=BHK, 0=RK)"
    )

    rooms_num = st.number_input(
        "Total Number of Rooms",
        min_value=1,
        max_value=12,
        value=2,
        help="Total count of rooms"
    )

    verification_days = st.number_input(
        "Days Since Posting",
        min_value=0.0,
        max_value=1825.0,
        value=30.0,
        help="Number of days since property was posted"
    )

# Predict button
st.markdown("---")
predict_button = st.button("🔮 Predict Rent Price", type="primary", use_container_width=True)

if predict_button:
    # Prepare input data
    try:
        # Encode categorical variables
        # Handle unseen categories by using the first class as default
        location_encoded = location_encoder.transform([location])[0] if location in location_encoder.classes_ else 0
        city_encoded = city_encoder.transform([city])[0] if city in city_encoder.classes_ else 0
        status_encoded = status_encoder.transform([status])[0] if status in status_encoder.classes_ else 0
        property_type_encoded = property_type_encoder.transform([property_type])[0] if property_type in property_type_encoder.classes_ else 0

        # Create feature DataFrame in the correct order
        feature_values = [
            location_encoded,
            city_encoded,
            latitude,
            longitude,
            num_bathrooms,
            num_balconies,
            is_negotiable,
            security_deposit,
            status_encoded,
            size_ft,
            bhk,
            rooms_num,
            property_type_encoded,
            verification_days
        ]
        features = pd.DataFrame([feature_values], columns=feature_columns)

        # Make prediction
        prediction = model.predict(features)[0]

        # Display result
        st.markdown(f"""
            <div class="success-message">
                <strong>🎯 Predicted Monthly Rent:</strong><br>
                ₹{prediction:,.2f} INR
            </div>
        """, unsafe_allow_html=True)

        # Additional information
        st.markdown("---")
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.info(f"📍 {location}, {city}")

        with col_b:
            st.info(f"🏠 {property_type}")

        with col_c:
            st.info(f"📏 {size_ft:,} sq ft")

    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.error("Please check your inputs and try again.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <p>Built with ❤️ using Random Forest Regressor | Grid Search Optimized</p>
        <p>Test MAE: ₹12,417.01 | Cities: Mumbai, Delhi, Pune, Hisar</p>
    </div>
""", unsafe_allow_html=True)
