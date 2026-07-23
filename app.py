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
# 2. SESSION STATE MANAGEMENT FOR METADATA & GEOMETRIES
# ========================================================
if "map_center" not in st.session_state:
    st.session_state["map_center"] = [-23.5505, 46.6333]
if "active_polygon" not in st.session_state:
    st.session_state["active_polygon"] = None
if "concession_metadata" not in st.session_state:
    st.session_state["concession_metadata"] = {
        "License ID": "Default Grid",
        "Concession Name": "Baseline Exploration Block Alpha",
        "Holder / Operator": "Global Minerals Ltd",
        "Size (Hectares)": "2,450 Ha",
        "Grant Date": "2018-05-12",
        "Expiry Date": "2032-12-31",
        "Status": "Active Operational License"
    }

def create_mock_polygon(lat, lon, size=0.02):
    """Generates a standard GeoJSON polygon surrounding target coordinates."""
    return {
        "type": "Feature",
        "properties": {"name": "Concession Area Boundary"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [lon - size, lat - size],
                [lon + size, lat - size],
                [lon + size, lat + size],
                [lon - size, lat + size],
                [lon - size, lat - size]
            ]]
        }
    }

# ========================================================
# 3. SIDEBAR PARAMETERS & BASEMAP SELECTOR
# ========================================================
st.set_page_config(page_title="SatIntel AI Gold Platform", layout="wide")
st.title("🛰️ SatIntel: Comprehensive 5-Way Gold Exploration Hub")

st.sidebar.header("🎯 Target Selection Menu")

# BASEMAP TILE LAYER SWITCHER (The dropdown you requested)
selected_basemap = st.sidebar.selectbox(
    "🗺️ Select Map Layer View",
    ["Google Satellite Imagery", "OpenStreetMap (Standard)", "Esri Topographic Map", "Stamen Terrain"]
)

selected_year = st.sidebar.slider("Select Analysis Year", min_value=1990, max_value=2026, value=2024, step=1)

search_method = st.sidebar.radio(
    "Select Landfolio Lookup Method",
    ["(a) License # Search", "(b) Name Search", "(c) Map Selection", "(d) File Upload"]
)

# Process sidebar inputs and update database records
if search_method == "(a) License # Search":
    license_num = st.sidebar.text_input("Enter License Number (Exact Match)", placeholder="e.g., L-4095-X")
    if license_num:
        st.session_state["map_center"] = [-23.5400, 46.6200]
        st.session_state["active_polygon"] = create_mock_polygon(-23.5400, 46.6200, size=0.015)
        st.session_state["concession_metadata"] = {
            "License ID": license_num,
            "Concession Name": f"Gold Strike Zone {license_num}",
            "Holder / Operator": "Barrick Gold Exploration Corp",
            "Size (Hectares)": "5,120 Ha",
            "Grant Date": "2021-01-15",
            "Expiry Date": "2036-06-30",
            "Status": "Granted Exploration Phase"
        }
        st.sidebar.success(f"✓ License Loaded")

elif search_method == "(b) Name Search":
    name_query = st.sidebar.text_input("Mine or Holder Name (Fuzzy Match)", placeholder="e.g., Newmont")
    if name_query:
        st.session_state["map_center"] = [-23.5600, 46.6500]
        st.session_state["active_polygon"] = create_mock_polygon(-23.5600, 46.6500, size=0.025)
        st.session_state["concession_metadata"] = {
            "License ID": "L-9921-B",
            "Concession Name": f"Prospect Ridge ({name_query} Match)",
            "Holder / Operator": f"{name_query} Joint Venture Group",
            "Size (Hectares)": "12,840 Ha",
            "Grant Date": "2015-08-22",
            "Expiry Date": "2030-08-22",
            "Status": "Production Lease Option"
        }
        st.sidebar.success(f"✓ Operator Found")

elif search_method == "(c) Map Selection":
    st.sidebar.info("👉 Click anywhere on the map to query the underlying Landfolio database block.")

elif search_method == "(d) File Upload":
    uploaded_file = st.sidebar.file_uploader("Upload Boundary (GeoJSON, KML)", type=["geojson", "kml"])
    if uploaded_file is not None:
        st.session_state["map_center"] = [-23.5300, 46.6100]
        st.session_state["active_polygon"] = create_mock_polygon(-23.5300, 46.6100, size=0.03)
        st.session_state["concession_metadata"] = {
            "License ID": "UPLOADED-TEMP-01",
            "Concession Name": uploaded_file.name.split('.')[0].upper(),
            "Holder / Operator": "User Imported Layer",
            "Size (Hectares)": "3,110 Ha",
            "Grant Date": "N/A",
            "Expiry Date": "Pending Review",
            "Status": "Custom Local Polygon Layer"
        }
        st.sidebar.success(f"✓ Boundary Loaded")

st.sidebar.divider()
target_commodity = st.sidebar.selectbox("Commodity Focus", ["Gold", "Copper", "Emeralds", "Diamonds"])

# ========================================================
# 4. INTERACTIVE MAPPING WITH CUSTOM TILES CONTROLLER
# ========================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"🗺️ Interactive Concession Map Layer")
    
    # Configure basemap configurations dynamically
    if selected_basemap == "Google Satellite Imagery":
        m = folium.Map(
            location=st.session_state["map_center"], 
            zoom_start=12,
            tiles="https://google.com{x}&y={y}&z={z}",
            attr="Google Satellite"
        )
    elif selected_basemap == "Esri Topographic Map":
        m = folium.Map(
            location=st.session_state["map_center"], 
            zoom_start=12,
            tiles="https://arcgisonline.com{z}/{y}/{x}",
            attr="Esri Topo Map"
        )
    elif selected_basemap == "Stamen Terrain":
        m = folium.Map(location=st.session_state["map_center"], zoom_start=12, tiles="stamenterrain")
    else:
        m = folium.Map(location=st.session_state["map_center"], zoom_start=12) # Standard OpenStreetMap Default
        
    # Draw polygon layer overlay if active
    if st.session_state["active_polygon"]:
        folium.GeoJson(
            st.session_state["active_polygon"],
            name="Concession Boundary",
            style_function=lambda x: {
                "fillColor": "#FFD700",
                "color": "#FF4B4B",
                "weight": 3,
                "fillOpacity": 0.35,
            }
        ).add_to(m)
        
    # Launch Map Component
    map_data = st_folium(m, width=550, height=380, key=f"map_{selected_basemap}_{st.session_state['map_center']}")
    
    # Map click listener logic for option (c)
    if search_method == "(c) Map Selection" and map_data and map_data.get("last_clicked"):
        click_point = map_data["last_clicked"]
        lat, lng = click_point["lat"], click_point["lng"]
        
        st.session_state["map_center"] = [lat, lng]
        st.session_state["active_polygon"] = create_mock_polygon(lat, lng, size=0.012)
        st.session_state["concession_metadata"] = {
            "License ID": f"LAT:{lat:.2f}_LNG:{lng:.2f}",
            "Concession Name": f"Spatial Discovery Zone ({lat:.3f}N)",
            "Holder / Operator": "Discovered via Map Click Portal",
            "Size (Hectares)": "1,850 Ha",
            "Grant Date": "2024-07-24",
            "Expiry Date": "2035-12-31",
            "Status": "Unassigned Open Application Target"
        }
        st.rerun()

    # --- THE METADATA REGISTRATION TABLE PANEL (The database read fields you requested) ---
    st.write("### 📋 Landfolio Mining Database Registry")
    
    # We display the python meta dictionary as a clean native UI table view
    st.table(st.session_state["concession_metadata"])

# ========================================================
# 5. REMOTE SENSING DATA ENGINE & AI RUNTIME
# ========================================================
with col2:
    st.subheader("📊 5 Core Remote Sensing Target Frameworks")
    
    with st.spinner("Processing multispectral imagery stack..."):
        m_data = fetch_and_calculate_spatz(st.session_state["map_center"], st.session_state["map_center"], selected_year)
    
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
    st.caption(f"🛰️ Source Pipeline ID: {m_data['Satellite_Used']}")
    st.divider()
    
    if st.button("🚀 Generate 5-Way Geological Synthesis"):
        with st.spinner("watsonx.ai is correlating all matrices..."):
