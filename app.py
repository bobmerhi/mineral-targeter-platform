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
# 2. SESSION STATE STATE-MACHINE FOR MAPPING GEOMETRIES
# ========================================================
# This keeps track of our map view and active polygon features across updates
if "map_center" not in st.session_state:
    st.session_state["map_center"] = [-23.5505, 46.6333]
if "active_polygon" not in st.session_state:
    st.session_state["active_polygon"] = None
if "concession_text" not in st.session_state:
    st.session_state["concession_text"] = "Default Coordinates Active"

def create_mock_polygon(lat, lon, size=0.02):
    """Generates a standard square GeoJSON polygon surrounding target coordinates."""
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
# 3. UPGRADED TARGET SELECTION MENU (SIDEBAR)
# ========================================================
st.set_page_config(page_title="SatIntel AI Gold Platform", layout="wide")
st.title("🛰️ SatIntel: Comprehensive 5-Way Gold Exploration Hub")

st.sidebar.header("🎯 Target Selection Menu")
selected_year = st.sidebar.slider("Select Analysis Year", min_value=1990, max_value=2026, value=2024, step=1)

search_method = st.sidebar.radio(
    "Select Landfolio Lookup Method",
    ["(a) License # Search", "(b) Name Search", "(c) Map Selection", "(d) File Upload"]
)

# Reset active targets dynamically based on input changes
if search_method == "(a) License # Search":
    license_num = st.sidebar.text_input("Enter License Number (Exact Match)", placeholder="e.g., L-4095-X")
    if license_num:
        # Mocking incoming data from GET /api/concessions?license={num}
        st.session_state["map_center"] = [-23.5400, 46.6200] # Adjust Map Center
        st.session_state["active_polygon"] = create_mock_polygon(-23.5400, 46.6200, size=0.015)
        st.session_state["concession_text"] = f"Landfolio Concession License: {license_num}"
        st.sidebar.success(f"✓ Found License: {license_num}")

elif search_method == "(b) Name Search":
    name_query = st.sidebar.text_input("Mine or Holder Name (Fuzzy Match)", placeholder="e.g., AngloGold")
    if name_query:
        # Mocking results from GET /api/concessions?name={query}
        st.session_state["map_center"] = [-23.5600, 46.6500]
        st.session_state["active_polygon"] = create_mock_polygon(-23.5600, 46.6500, size=0.025)
        st.session_state["concession_text"] = f"Fuzzy Trigram Match: '{name_query}'"
        st.sidebar.success(f"✓ Connected to Match")

elif search_method == "(c) Map Selection":
    st.sidebar.info("👉 Click anywhere on the map to trigger a spatial ST_Contains query.")

elif search_method == "(d) File Upload":
    uploaded_file = st.sidebar.file_uploader("Upload Boundary (GeoJSON, KML)", type=["geojson", "kml"])
    if uploaded_file is not None:
        # In production, parse actual file string into a dictionary layer
        st.session_state["map_center"] = [-23.5300, 46.6100]
        st.session_state["active_polygon"] = create_mock_polygon(-23.5300, 46.6100, size=0.03)
        st.session_state["concession_text"] = f"Uploaded Boundary Layer: {uploaded_file.name}"
        st.sidebar.success(f"✓ Layer Reprojected to EPSG:4326")

st.sidebar.divider()
target_commodity = st.sidebar.selectbox("Commodity Focus", ["Gold", "Copper", "Emeralds", "Diamonds"])

# ========================================================
# 4. INTERACTIVE MAPPING WITH BOUNDARY RENDERERS
# ========================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"🗺️ Interactive Concession Map ({selected_year})")
    
    # Generate Map centered around current session center state variable
    m = folium.Map(location=st.session_state["map_center"], zoom_start=12)
    
    # DYNAMIC RENDERER: If an active polygon data layer exists, draw it onto the map
    if st.session_state["active_polygon"]:
        folium.GeoJson(
            st.session_state["active_polygon"],
            name="Concession Boundary",
            style_function=lambda x: {
                "fillColor": "#FFD700",  # Gold color fill
                "color": "#FF8C00",      # Dark orange border line
                "weight": 3,
                "fillOpacity": 0.4,
            }
        ).add_to(m)
    
    # Catch user clicking interactions
    map_data = st_folium(m, width=550, height=400, key=f"map_{st.session_state['map_center']}")
    
    # Handle option (c) Map Clicking spatial queries
    if search_method == "(c) Map Selection" and map_data and map_data.get("last_clicked"):
        click_point = map_data["last_clicked"]
        lat, lng = click_point["lat"], click_point["lng"]
        
        # Execute ST_Contains logic emulation -> generate polygon around click
        st.session_state["map_center"] = [lat, lng]
        st.session_state["active_polygon"] = create_mock_polygon(lat, lng, size=0.012)
        st.session_state["concession_text"] = f"ST_Contains Intersect at Point: {lat:.4f}, {lng:.4f}"
        st.rerun() # Forces visual map reload to overlay new shape immediately

    st.info(f"📋 **Active Concession Pipeline Data Context:**\n{st.session_state['concession_text']}")

# ========================================================
# 5. REMOTE SENSING DATA ENGINE & AI RUNTIME
# ========================================================
with col2:
    st.subheader("📊 5 Core Remote Sensing Target Frameworks")
    
    # Pass our active map center point straight to the imagery algorithms
    with st.spinner("Processing multispectral imagery stack..."):
        m_data = fetch_and_calculate_spatz(
            st.session_state["map_center"][0], 
            st.session_state["map_center"][1], 
            selected_year
        )
    
    # Render analytics layout
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
            client = get_watsonx_client()
            
            prompt = f"""
            [Role: Senior Exploration Geologist]
            Evaluate these Spatz remote sensing values for {target_commodity} mineralization at coordinates {st.session_state['map_center']} for the year {selected_year}:
            Context Block: {st.session_state['concession_text']}
            
            Metrics:
            - Iron Oxide: {m_data['Way_1_Iron_Oxide_Gossan']}, Clay: {m_data['Way_1_Clay_Phyllic']}
            - Structural Density: {m_data['Way_2_Fault_Density_Index']}
            - Quartz/Silica: {m_data['Way_3_Silica_Flooding_Cap']}
            - Vegetation Shift: {m_data['Way_4_Geobotanical_Stress']}
            - Combined WLC Confidence: {m_data['Way_5_WLC_Score_Percent']}%
            
            Provide a clear, brief technical recommendation for ground-truthing based on these values.
            """
            
            model = ModelInference(model_id="ibm/granite-13b-instruct-v2", credentials=credentials, project_id=PROJECT_ID)
            st.markdown(model.generate_text(prompt=prompt))
