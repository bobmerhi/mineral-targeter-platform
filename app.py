import streamlit as st
import folium
from streamlit_folium import st_folium
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from georemote import fetch_and_calculate_spatz

# ========================================================
# 1. SECURE CONFIGURATION VIA STREAMLIT CLOUD SECRETS
# ========================================================
# Streamlit will look for these variables in your "Advanced Settings -> Secrets" panel
try:
    IBM_API_KEY = st.secrets["IBM_API_KEY"]
    PROJECT_ID = st.secrets["PROJECT_ID"]
except KeyError:
    st.error("🔒 Streamlit Secrets missing! Please add 'IBM_API_KEY' and 'PROJECT_ID' to your Cloud Advanced Settings panel.")
    st.stop()

credentials = {
    "url": "https://ibm.com",
    "apikey": IBM_API_KEY,
    "token_type": "Bearer" # Bypasses the Cloud Pak version checking error
}

# Cache client connection to improve platform loading speeds
@st.cache_resource
def get_watsonx_client():
    client = APIClient(credentials)
    client.set.default_project(PROJECT_ID)
    return client

# ========================================================
# 2. PLATFORM FRONTEND INTERFACE
# ========================================================
st.set_page_config(page_title="GeoTarget AI Platform", layout="wide")

st.title("🌋 GeoTarget AI: Mineral & Gem Exploration Platform")
st.caption("Powered by IBM watsonx.ai & Automated Remote Sensing Engines")

# Sidebar - User Inputs
st.sidebar.header("🎯 Exploration Controls")
target_commodity = st.sidebar.selectbox("Select Target Commodity", ["Gold", "Copper", "Emeralds", "Diamonds"])

# DYNAMIC DATES PANEL (Historical Timeline Control)
selected_year = st.sidebar.slider("Select Analysis Year", min_value=1990, max_value=2026, value=2024, step=1)

# Target Location Coordinates
st.sidebar.subheader("📍 Target Area Location")
target_lat = st.sidebar.number_input("Latitude", value=-23.5505, format="%.5f")
target_lon = st.sidebar.number_input("Longitude", value=46.6333, format="%.5f")

# Split Dashboard Layout into 2 Functional Columns
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"🗺️ Target Mapping Location ({selected_year})")
    # Generate interactive terrain map
    m = folium.Map(location=[target_lat, target_lon], zoom_start=11)
    folium.Marker([target_lat, target_lon], popup=f"Target Coordinates ({selected_year})", icon=folium.Icon(color='red', icon='bolt')).add_to(m)
    st_folium(m, width=550, height=400)
    st.info(f"💡 Tip: Use the Timeline Slider in the sidebar to view metrics across different years.")

with col2:
    st.subheader(f"📊 Spatz Spectral Processing Metrics ({selected_year})")
    
    # Run the background remote sensing pipeline
    with st.spinner("Analyzing multi-spectral imagery..."):
        metrics = fetch_and_calculate_spatz(target_lat, target_lon, selected_year)
    
    # Display calculated Spatz ratios visually to user
    c1, c2, c3 = st.columns(3)
    c1.metric("Iron Oxide (Gossan)", metrics["Iron_Oxide_Ratio_Spatz"])
    c2.metric("Clay/Hydroxyl Ratio", metrics["Clay_Hydroxyl_Ratio"])
    c3.metric("Silicification Index", metrics["Silicification_Index"])
    st.caption(f"🛰️ Source Dataset Catalog: {metrics['Satellite_Used']}")
    
    st.divider()
    st.subheader("🤖 watsonx.ai Automated Geological Assessment")
    
    # Trigger button to activate the AI Agent
    if st.button("▶ Run AI Prospectivity Assessment"):
        with st.spinner("watsonx.ai Granite model is interpreting alteration telemetry..."):
            client = get_watsonx_client()
            
            prompt = f"""
            [Role: Senior Exploration Geologist & Remote Sensing Specialist]
            
            Evaluate these Spatz remote sensing values for {target_commodity} mineralization at coordinates ({target_lat}, {target_lon}) for the historical target year {selected_year}:
            - Iron Oxide / Ferric Index: {metrics['Iron_Oxide_Ratio_Spatz']}
            - Clay / Hydroxyl Index: {metrics['Clay_Hydroxyl_Ratio']}
            - Silicification Indicator: {metrics['Silicification_Index']}
            
            Task:
            Provide a brief, technical, and highly practical geological report explaining what structural or alteration anomalies these indicators represent. Suggest specific next steps for ground-truthing or field sampling.
            """
            
            model = ModelInference(model_id="ibm/granite-13b-instruct-v2", credentials=credentials, project_id=PROJECT_ID)
            ai_report = model.generate_text(prompt=prompt)
            st.markdown(ai_report)
