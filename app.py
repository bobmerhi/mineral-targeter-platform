import streamlit as st
import folium
from streamlit_folium import st_folium
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from georemote import fetch_and_calculate_spatz

# 1. SECURE CONFIGURATION VIA STREAMLIT CLOUD SECRETS
try:
    IBM_API_KEY = st.secrets["IBM_API_KEY"]
    PROJECT_ID = st.secrets["PROJECT_ID"]
except KeyError:
    st.error("🔒 Streamlit Secrets missing! Please add 'IBM_API_KEY' and 'PROJECT_ID' to your Cloud Settings.")
    st.stop()

credentials = {
    "url": "https://ibm.com",
    "apikey": IBM_API_KEY,
    "token_type": "Bearer" 
}

@st.cache_resource
def get_watsonx_client():
    client = APIClient(credentials)
    client.set.default_project(PROJECT_ID)
    return client

# ========================================================
# 2. PLATFORM FRONTEND INTERFACE
# ========================================================
st.set_page_config(page_title="SatIntel AI Gold Platform", layout="wide")

st.title("🛰️ SatIntel: Comprehensive 5-Way Gold Exploration Hub")
st.caption("Synchronized Remote Sensing Analytics & IBM watsonx.ai Geological Reasoner")

# Sidebar
st.sidebar.header("🎯 Target Selection")
selected_year = st.sidebar.slider("Select Analysis Year", min_value=1990, max_value=2026, value=2024, step=1)
target_lat = st.sidebar.number_input("Latitude", value=-23.5505, format="%.5f")
target_lon = st.sidebar.number_input("Longitude", value=46.6333, format="%.5f")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"🗺️ Target Concession Location ({selected_year})")
    m = folium.Map(location=[target_lat, target_lon], zoom_start=11)
    folium.Marker([target_lat, target_lon], popup="Exploration Focus", icon=folium.Icon(color='gold', icon='star')).add_to(m)
    st_folium(m, width=550, height=400)
    st.info("💡 Pro Tip: Toggle different timeline years to trace weathering patterns over time.")

with col2:
    st.subheader("📊 The 5 Core Remote Sensing Ways To Target Gold")
    
    # Process the expanded pipeline
    with st.spinner("Compiling multi-spectral and spatial layers..."):
        m_data = fetch_and_calculate_spatz(target_lat, target_lon, selected_year)
    
    # Displaying the 5 Ways
    st.markdown("#### **WAY 1: Hydrothermal Alteration**")
    w1_c1, w1_c2 = st.columns(2)
    w1_c1.metric("Iron Oxide (Gossans)", m_data["Way_1_Iron_Oxide_Gossan"])
    w1_c2.metric("Clay/Hydroxyl Index", m_data["Way_1_Clay_Phyllic"])
    
    st.markdown("#### **WAY 2: Structural Geology**")
    st.metric("Fault Intersection Density", m_data["Way_2_Fault_Density_Index"])
    
    st.markdown("#### **WAY 3: Lithological Silicification**")
    st.metric("Quartz Veining Emissivity", m_data["Way_3_Silica_Flooding_Cap"])
    
    st.markdown("#### **WAY 4: Geobotanical Anomalies**")
    st.metric("Vegetation Stress Proxy (NDVI Shift)", m_data["Way_4_Geobotanical_Stress"])
    
    st.markdown("#### **WAY 5: GIS Predictive Synthesis**")
    st.metric("Weighted Linear Combination (WLC) Prospectivity", f"{m_data['Way_5_WLC_Score_Percent']}%")
    
    st.caption(f"🛰️ Source Pipeline: {m_data['Satellite_Used']}")
    
    st.divider()
    
    # 5-WAY AI MATRIX GENERATOR BUTTON
    if st.button("🚀 Generate Complete 5-Way Geological Synthesis"):
        with st.spinner("watsonx.ai is correlating all 5 exploration matrices..."):
            client = get_watsonx_client()
            
            prompt = f"""
            [Role: Principal Mining Consultant & Remote Sensing Mastermind]
            
            Task: Synthesize a definitive gold prospectivity report at coordinates ({target_lat}, {target_lon}) for Year {selected_year} based on the 5 core exploration vectors:
            
            1. Hydrothermal Alteration (Iron Oxide: {m_data['Way_1_Iron_Oxide_Gossan']}, Clay: {m_data['Way_1_Clay_Phyllic']}) [1]
            2. Structural Lineaments (Fault Density Index: {m_data['Way_2_Fault_Density_Index']}) [1]
            3. Silicification (Quartz Vein Cap Indicator: {m_data['Way_3_Silica_Flooding_Cap']}) [1]
            4. Geobotanical Stress (NDVI Toxicity Shift: {m_data['Way_4_Geobotanical_Stress']}) [1]
            5. GIS Multi-Criteria Modeling (WLC Prospectivity Score: {m_data['Way_5_WLC_Score_Percent']}%) [1]
            
            Format your final response cleanly with markdown headers matching each of the 5 Ways. Conclude with an explicit, high-conviction 'Drill/No-Drill' tactical recommendation.
            """
            
            model = ModelInference(model_id="ibm/granite-13b-instruct-v2", credentials=credentials, project_id=PROJECT_ID)
            st.markdown(model.generate_text(prompt=prompt))
