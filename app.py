import streamlit as st
import folium
from streamlit_folium import st_folium
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from georemote import fetch_and_calculate_spatz

# 1. Credentials with your working Cloud Pak bypass fix
IBM_API_KEY = "B_Nz4kWw4scaybtZaZfWi6NEFh-DiPRhZshIxVJm2cP"
PROJECT_ID = "c5ab169c-a585-44aa-829f-e33463e67ed9"
credentials = {
    "url": "https://ibm.com",
    "apikey": IBM_API_KEY,
    "token_type": "Bearer" 
}

st.set_page_config(page_title="GeoTarget AI", layout="wide")
st.title("🌋 GeoTarget AI: Mineral Exploration Platform")

# Sidebar Configuration
st.sidebar.header("🎯 Exploration Controls")
target_commodity = st.sidebar.selectbox("Commodity", ["Gold", "Copper", "Emeralds", "Diamonds"])

# DYNAMIC DATES PANEL (The timeline slider you requested)
selected_year = st.sidebar.slider("Select Analysis Year", min_value=1990, max_value=2026, value=2024, step=1)

target_lat = st.sidebar.number_input("Latitude", value=-23.5505, format="%.5f")
target_lon = st.sidebar.number_input("Longitude", value=46.6333, format="%.5f")

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"🗺️ Target Mapping Location ({selected_year})")
    m = folium.Map(location=[target_lat, target_lon], zoom_start=11)
    folium.Marker([target_lat, target_lon], popup=f"Target ({selected_year})", icon=folium.Icon(color='red')).add_to(m)
    st_folium(m, width=550, height=400)

with col2:
    st.subheader(f"📊 Spatz Spectral Processing ({selected_year})")
    
    # Run the real remote sensing pipeline
    metrics = fetch_and_calculate_spatz(target_lat, target_lon, selected_year)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Iron Oxide (Gossan)", metrics["Iron_Oxide_Ratio_Spatz"])
    c2.metric("Clay/Hydroxyl Ratio", metrics["Clay_Hydroxyl_Ratio"])
    c3.metric("Silicification Index", metrics["Silicification_Index"])
    st.caption(f"Source Data: {metrics['Satellite_Used']}")
    
    st.divider()
    
    if st.button("▶ Run AI Geological Assessment"):
        with st.spinner("watsonx.ai is interpreting remote sensing layers..."):
            client = APIClient(credentials)
            client.set.default_project(PROJECT_ID)
            
            prompt = f"""
            [Role: Expert Exploration Geologist]
            Evaluate these Spatz remote sensing values for {target_commodity} mineralization at coordinates {target_lat}, {target_lon} for the year {selected_year}:
            - Iron Oxide: {metrics['Iron_Oxide_Ratio_Spatz']}
            - Clay: {metrics['Clay_Hydroxyl_Ratio']}
            - Silicification: {metrics['Silicification_Index']}
            Provide a clear, brief technical recommendation for ground-truthing based on these values.
            """
            
            model = ModelInference(model_id="ibm/granite-13b-instruct-v2", credentials=credentials, project_id=PROJECT_ID)
            st.markdown(model.generate_text(prompt=prompt))
