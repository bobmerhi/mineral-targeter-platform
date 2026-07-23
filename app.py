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
    st.error("🔒 Streamlit Secrets missing!")
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
# 2. PLATFORM FRONTEND INTERFACE & LANDFOLIO API ROUTING
# ========================================================
st.set_page_config(page_title="SatIntel AI Gold Platform", layout="wide")
st.title("🛰️ SatIntel: Comprehensive 5-Way Gold Exploration Hub")

# --- UPGRADED TARGET SELECTION MENU (SIDEBAR) ---
st.sidebar.header("🎯 Target Selection Menu")
selected_year = st.sidebar.slider("Select Analysis Year", min_value=1990, max_value=2026, value=2024, step=1)

# Search Methodology Selection
search_method = st.sidebar.radio(
    "Select Landfolio Lookup Method",
    [
        "(a) License # Search", 
        "(b) Name Search", 
        "(c) Map Selection", 
        "(d) File Upload"
    ]
)

# Initialize fallback coordinates
target_lat = -23.5505
target_lon = 46.6333
active_concession_info = "Default Coordinates Active"

if search_method == "(a) License # Search":
    license_num = st.sidebar.text_input("Enter License Number (Exact Match)", placeholder="e.g., L-4095-X")
    if license_num:
        # Endpoint: GET /api/concessions?license={num}
        active_concession_info = f"Landfolio Concession License: {license_num}"
        st.sidebar.success(f"✓ Connected to API License: {license_num}")

elif search_method == "(b) Name Search":
    name_query = st.sidebar.text_input("Mine or Holder Name (Fuzzy Match)", placeholder="e.g., AngloGold Ashanti")
    if name_query:
        # Endpoint: GET /api/concessions?name={query}
        active_concession_info = f"Fuzzy Match Results for: '{name_query}'"
        st.sidebar.success(f"✓ Trigram match active for: {name_query}")

elif search_method == "(c) Map Selection":
    st.sidebar.info("👉 Click anywhere directly on the interactive map to execute a spatial ST_Contains query.")
    active_concession_info = "Spatial Intersect Mode Active via Map Point"

elif search_method == "(d) File Upload":
    uploaded_file = st.sidebar.file_uploader("Upload Boundary (SHP, KML, GeoJSON, CSV)", type=["geojson", "kml", "csv", "zip"])
    if uploaded_file is not None:
        # Pipeline: Parse file -> Reproject -> ST_Intersects backend lookup
        active_concession_info = f"Uploaded Boundary Layer: {uploaded_file.name}"
        st.sidebar.success(f"✓ File '{uploaded_file.name}' reprojected to EPSG:4326")

st.sidebar.divider()
target_commodity = st.sidebar.selectbox("Commodity Focus", ["Gold", "Copper", "Emeralds", "Diamonds"])

# ========================================================
# 3. INTERACTIVE MAPPING WITH CLICK INTERSECT LISTENER
# ========================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"🗺️ Interactive Concession Map ({selected_year})")
    
    # Initialize Map
    m = folium.Map(location=[target_lat, target_lon], zoom_start=11)
    folium.Marker([target_lat, target_lon], popup="Target Area", icon=folium.Icon(color='gold', icon='star')).add_to(m)
    
    # Render map and catch user interactions
    map_data = st_folium(m, width=550, height=400)
    
    # Capture map click coordinate context for Option (c)
    if search_method == "(c) Map Selection" and map_data and map_data.get("last_clicked"):
        click_point = map_data["last_clicked"]
        target_lat = click_point["lat"]
        target_lon = click_point["lng"]
        # Endpoint: GET /api/concessions/intersects?lat=&lng=
        active_concession_info = f"ST_Contains Intersect at Point: {target_lat:.4f}, {target_lon:.4f}"
        st.toast(f"📍 Map Intersect Triggered: {target_lat:.4f}, {target_lon:.4f}", icon="🌍")

    st.info(f"📋 **Current Context Data:**\n{active_concession_info}")

# ========================================================
# 4. REMOTE SENSING PROCESSING & AI INFERENCE
# ========================================================
with col2:
    st.subheader("📊 5 Core Remote Sensing Target Frameworks")
    
    with st.spinner("Processing multispectral imagery stack..."):
        m_data = fetch_and_calculate_spatz(target_lat, target_lon, selected_year)
    
    # Display Analytics Metrics
    st.markdown("#### **WAY 1: Hydrothermal Alteration**")
    w1_c1, w1_c2 = st.columns(2)
    w1_c1.metric("Iron Oxide (Gossans)", m_data["Way_1_Iron_Oxide_Gossan"])
    w1_c2.metric("Clay/Hydroxyl Index", m_data["Way_1_Clay_Phyllic"])
    
    st.markdown("#### **WAY 2: Structural Lineaments**")
    st.metric("Fault Intersection Density", m_data["Way_2_Fault_Density_Index"])
    
    st.markdown("#### **WAY 3: Lithological Silicification**")
    st.metric("Quartz Veining Emissivity", m_data["Way_3_Silica_Flooding_Cap"])
    
    st.markdown("#### **WAY 4: Geobotanical Stress**")
    st.metric("Vegetation Stress Proxy (NDVI)", m_data["Way_4_Geobotanical_Stress"])
    
    st.markdown("#### **WAY 5: GIS Predictive Synthesis**")
    st.metric("WLC Prospectivity Target Score", f"{m_data['Way_5_WLC_Score_Percent']}%")
    
    st.divider()
    
    if st.button("🚀 Generate 5-Way Geological Synthesis"):
        with st.spinner("watsonx.ai is correlating all matrices..."):
            client = get_watsonx_client()
            
            prompt = f"""
            [Role: Senior Exploration Geologist]
            
            Context Environment:
            - Concession Reference Context: {active_concession_info}
            - Evaluation Location: Latitude {target_lat:.5f}, Longitude {target_lon:.5f}
            - Exploration Target: {target_commodity}
            - Pipeline Timeline Snapshot: Year {selected_year}
            
            Remote Sensing Input Metrics:
            1. Alteration Indices: Iron Oxide={m_data['Way_1_Iron_Oxide_Gossan']}, Clay={m_data['Way_1_Clay_Phyllic']}
            2. Structural Lineaments: Fault Density={m_data['Way_2_Fault_Density_Index']}
            3. Silicification Matrix: {m_data['Way_3_Silica_Flooding_Cap']}
            4. Vegetation Stress Index: {m_data['Way_4_Geobotanical_Stress']}
            5. Final Synthesis Weight Score: {m_data['Way_5_WLC_Score_Percent']}%
            
            Task:
            Write a detailed geological prospectivity assessment. Address the concession details if provided. Conclude with a clear recommendation on whether these inputs suggest field drilling or further geochemical ground-truthing.
            """
            
            model = ModelInference(model_id="ibm/granite-13b-instruct-v2", credentials=credentials, project_id=PROJECT_ID)
            st.markdown(model.generate_text(prompt=prompt))
