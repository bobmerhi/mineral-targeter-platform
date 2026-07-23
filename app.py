import streamlit as st
import folium
from streamlit_folium import st_folium
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from georemote import fetch_and_calculate_spatz

# ========================================================
# 1. SECURE CONFIGURATION VIA STREAMLIT CLOUD SECRETS
# ========================================================
try:
    IBM_API_KEY = st.secrets["IBM_API_KEY"]
    PROJECT_ID = st.secrets["PROJECT_ID"]
except KeyError:
    st.error("🔒 Streamlit Secrets missing! Please verify your setup.")
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
# 2. SESSION STATE MANAGEMENT - MOZAMBIQUE GEOGRAPHIC FOCUS
# ========================================================
# Shifting default platform target coordinates directly to the Manica Gold Belt, Mozambique
if "map_center" not in st.session_state:
    st.session_state["map_center"] = [-18.9300, 32.8800]
if "active_polygon" not in st.session_state:
    st.session_state["active_polygon"] = None
if "concession_metadata" not in st.session_state:
    st.session_state["concession_metadata"] = {
        "Código da Licença (Code)": "Default Grid - MZ",
        "Nome da Concessão": "Manica Goldfield Block Baseline",
        "Titular (Holder Company)": "Explorações de Moçambique Lda",
        "Área / Dimensão": "4,120 Hectares (Ha)",
        "Data de Emissão": "2020-04-10",
        "Data de Validade (Expiry)": "2035-04-10",
        "Tipo de Direito / Estado": "Concessão Mineira Activa (Active)"
    }

def create_mock_polygon(lat, lon, size=0.02):
    """Generates a standard bounding box polygon localized for Mozambique coordinates."""
    return {
        "type": "Feature",
        "properties": {"name": "Mozambique Cadastre Polygon"},
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
# 3. SIDEBAR PARAMETERS & RE-ROUTED LOOKUPS
# ========================================================
st.set_page_config(page_title="SatIntel Moçambique AI", layout="wide")
st.title("🛰️ SatIntel: Mozambique Mining Cadastre & 5-Way Gold Targeting Portal")
st.caption("Synchronized with Landfolio MIREME Structural Registry Standards")

st.sidebar.header("🎯 Portal de Seleção de Alvos")

selected_basemap = st.sidebar.selectbox(
    "🗺️ Select Map Layer View",
    ["Google Satellite Imagery", "OpenStreetMap (Standard)", "Esri Topographic Map", "Stamen Terrain"]
)

selected_year = st.sidebar.slider("Select Analysis Year", min_value=1990, max_value=2026, value=2024, step=1)

search_method = st.sidebar.radio(
    "Select Landfolio Lookup Method",
    ["(a) License # Search", "(b) Name Search", "(c) Map Selection", "(d) File Upload"]
)

# Updating the query simulations to load realistic Mozambique cadastre records
if search_method == "(a) License # Search":
    license_num = st.sidebar.text_input("Enter License Number (Exact Match)", placeholder="e.g., 10425CM")
    if license_num:
        # Panning map directly into Mozambique gold belt anomalies
        st.session_state["map_center"] = [-18.9150, 32.8900]
        st.session_state["active_polygon"] = create_mock_polygon(-18.9150, 32.8900, size=0.015)
        st.session_state["concession_metadata"] = {
            "Código da Licença (Code)": license_num,
            "Nome da Concessão": f"Projecto Ouro Manica {license_num}",
            "Titular (Holder Company)": "Minerais de Manica Prospecção S.A.",
            "Área / Dimensão": "3,890 Ha",
            "Data de Emissão": "2022-01-14",
            "Data de Validade (Expiry)": "2037-01-14",
            "Tipo de Direito / Estado": "Licença de Prospecção e Pesquisa"
        }
        st.sidebar.success("✓ Licença Carregada")

elif search_method == "(b) Name Search":
    name_query = st.sidebar.text_input("Mine or Holder Name (Fuzzy Match)", placeholder="e.g., Kenmare or local corp")
    if name_query:
        st.session_state["map_center"] = [-18.9450, 32.8600]
        st.session_state["active_polygon"] = create_mock_polygon(-18.9450, 32.8600, size=0.025)
        st.session_state["concession_metadata"] = {
            "Código da Licença (Code)": "8941CM",
            "Nome da Concessão": f"Anomalia Norte ({name_query} Joint Venture)",
            "Titular (Holder Company)": f"{name_query} Moçambique Mineração Lda",
            "Área / Dimensão": "8,450 Ha",
            "Data de Emissão": "2019-11-20",
            "Data de Validade (Expiry)": "2034-11-20",
            "Tipo de Direito / Estado": "Concessão Mineira outorgada"
        }
        st.sidebar.success("✓ Titular Encontrado")

elif search_method == "(c) Map Selection":
    st.sidebar.info("👉 Clique em qualquer ponto de Moçambique no mapa para executar uma consulta espacial no cadastro.")

elif search_method == "(d) File Upload":
    uploaded_file = st.sidebar.file_uploader("Upload Boundary (GeoJSON, KML)", type=["geojson", "kml"])
    if uploaded_file is not None:
        st.session_state["map_center"] = [-18.9200, 32.8700]
        st.session_state["active_polygon"] = create_mock_polygon(-18.9200, 32.8700, size=0.03)
        st.session_state["concession_metadata"] = {
            "Código da Licença (Code)": "MZ-IMPORT-SHP",
            "Nome da Concessão": str(uploaded_file.name).upper(),
            "Titular (Holder Company)": "Camada Importada pelo Utilizador",
            "Área / Dimensão": "1,200 Ha",
            "Data de Emissão": "N/A",
            "Data de Validade (Expiry)": "Pendente",
            "Tipo de Direito / Estado": "Polígono Personalizado Local"
        }
        st.sidebar.success("✓ Boundary Loaded")

st.sidebar.divider()
target_commodity = st.sidebar.selectbox("Commodity Focus", ["Gold", "Copper", "Heavy Mineral Sands", "Diamonds"])

# ========================================================
# 4. INTERACTIVE MAPPING WITH CUSTOM TILES CONTROLLER
# ========================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🗺️ Portal de Cadastro Mineiro de Moçambique")
    
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
        m = folium.Map(location=st.session_state["map_center"], zoom_start=12)
        
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
        
    map_data = st_folium(m, width=550, height=380, key=f"map_{selected_basemap}_{st.session_state['map_center']}")
    
    if search_method == "(c) Map Selection" and map_data and map_data.get("last_clicked"):
        click_point = map_data["last_clicked"]
        lat, lng = click_point["lat"], click_point["lng"]
        
        st.session_state["map_center"] = [lat, lng]
        st.session_state["active_polygon"] = create_mock_polygon(lat, lng, size=0.012)
        st.session_state["concession_metadata"] = {
            "Código da Licença (Code)": f"MZ_LT:{lat:.2f}_LG:{lng:.2f}",
            "Nome da Concessão": f"Zona de Descoberta Espacial ({lat:.3f}S)",
            "Titular (Holder Company)": "Ministério dos Recursos Minerais e Energia",
            "Área / Dimensão": "2,150 Ha",
            "Data de Emissão": "2026-07-24",
            "Data de Validade (Expiry)": "2040-12-31",
            "Tipo de Direito / Estado": "Área Livre para Requerimento"
        }
        st.rerun()

    st.write("### 📋 Registo do Cadastro de Minas (Landfolio)")
    st.table(st.session_state["concession_metadata"])

# ========================================================
# 5. REMOTE SENSING DATA ENGINE & AI RUNTIME
# ========================================================
with col2:
    st.subheader("📊 5 Core Remote Sensing Target Frameworks")
    
    with st.spinner("Processing multispectral imagery stack..."):
        m_data = fetch_and_calculate_spatz(st.session_state["map_center"][0], st.session_state["map_center"][1], selected_year)
    
    st.markdown("#### **WAY 1: Hydrothermal Alteration**")
    w1_c1, w1_c2 = st.columns(2)
    w1_c1.metric("Iron Oxide (Gossans)", m_data["Way_1_Iron_Oxide_Gossan"])
    w1_c2.metric("Clay/Hydroxyl Index", m_data["Way_1_Clay_Phyllic"])
    
    st.markdown("#### **WAY 2: Structural Lineaments**")
    st.metric("Fault Intersection Density", m_data["Way_2_Fault_Density_Index"])
    
    st.markdown("#### **WAY 3: Lithological Silicification**")
    st.metric("Quartz Veining Emissivity", m_data["Way_3_Silica_Flooding_Cap"])
    
