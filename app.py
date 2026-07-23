import streamlit as st
import folium
from streamlit_folium import st_folium
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from georemote import fetch_and_calculate_spatz, get_real_mozambique_cadastre

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
# 2. SESSION STATE ARBITRATION
# ========================================================
if "map_center" not in st.session_state:
    st.session_state["map_center"] = [-15.8000, 33.6000] # Set to Tete Mining Concession Region
if "active_polygon" not in st.session_state:
    st.session_state["active_polygon"] = None
if "concession_metadata" not in st.session_state:
    st.session_state["concession_metadata"] = {
        "Código da Licença (Code)": "Digite uma Licença Valida",
        "Nome da Concessão": "Aguardando Consulta",
        "Titular (Holder Company)": "Aguardando Banco de Dados Real",
        "Área / Dimensão": "0.00 Ha",
        "Data de Emissão": "N/A",
        "Data de Validade (Expiry)": "N/A",
        "Tipo de Direito / Estado": "N/A",
        "Substâncias": "N/A"
    }

# ========================================================
# 3. INTERFACE BUILDER & LIVE DATABASE SWITCHER
# ========================================================
st.set_page_config(page_title="SatIntel Moçambique Real-Time AI", layout="wide")
st.title("🛰️ SatIntel: Mozambique Mining Cadastre Real-Time Platform")
st.caption("Live Production Database Synchronization with Landfolio MIREME REST API Servers")

st.sidebar.header("🎯 Portal de Seleção de Alvos")

selected_basemap = st.sidebar.selectbox(
    "🗺️ Select Map Layer View",
    ["Google Satellite Imagery", "OpenStreetMap (Standard)", "Esri Topographic Map", "Stamen Terrain"]
)

selected_year = st.sidebar.slider("Select Analysis Year", min_value=1990, max_value=2026, value=2024, step=1)
search_method = st.sidebar.radio("Select Landfolio Lookup Method", ["(a) License # Search", "(c) Map Selection"])

if search_method == "(a) License # Search":
    license_num = st.sidebar.text_input("Enter License Number (Real Database Match)", placeholder="e.g., 11521")
    if license_num:
        with st.sidebar.spinner("Buscando dados em tempo real no Cadastro Nacional..."):
            db_result = get_real_mozambique_cadastre(license_num)
            
            if db_result["found"]:
                st.session_state["map_center"] = [db_result["lat"], db_result["lon"]]
                st.session_state["active_polygon"] = db_result["polygon"]
                st.session_state["concession_metadata"] = db_result["metadata"]
                st.sidebar.success(f"✓ Concessão {license_num} encontrada e desenhada!")
            else:
                st.sidebar.error(f"❌ Licença '{license_num}' não encontrada nos servidores governamentais.")

elif search_method == "(c) Map Selection":
    st.sidebar.info("👉 Clique em qualquer ponto de Moçambique no mapa para capturar as coordenadas reais do terreno.")

st.sidebar.divider()
target_commodity = st.sidebar.selectbox("Commodity Focus", ["Gold", "Copper", "Lithium", "Heavy Mineral Sands", "Emeralds"])

# ========================================================
# 4. MAP AND REAL METADATA RENDERING
# ========================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🗺️ Live Geographic Registry View")
    
    if selected_basemap == "Google Satellite Imagery":
        m = folium.Map(location=st.session_state["map_center"], zoom_start=11, tiles="https://google.com{x}&y={y}&z={z}", attr="Google Satellite")
    elif selected_basemap == "Esri Topographic Map":
        m = folium.Map(location=st.session_state["map_center"], zoom_start=11, tiles="https://arcgisonline.com{z}/{y}/{x}", attr="Esri Topo Map")
    else:
        m = folium.Map(location=st.session_state["map_center"], zoom_start=11)
        
    if st.session_state["active_polygon"]:
        folium.GeoJson(
            st.session_state["active_polygon"],
            name="Real Concession Boundary",
            style_function=lambda x: {"fillColor": "#00E5FF", "color": "#004D40", "weight": 3, "fillOpacity": 0.4}
        ).add_to(m)
        
    map_data = st_folium(m, width=550, height=380, key=f"map_{selected_basemap}_{st.session_state['map_center']}")
    
    if search_method == "(c) Map Selection" and map_data and map_data.get("last_clicked"):
        click_point = map_data["last_clicked"]
        lat, lng = click_point["lat"], click_point["lng"]
        st.session_state["map_center"] = [lat, lng]
        st.session_state["active_polygon"] = None
        st.session_state["concession_metadata"] = {
            "Código da Licença (Code)": "Coordenadas Manuais",
            "Nome da Concessão": f"Ponto de Interesse ({lat:.4f}S)",
            "Titular (Holder Company)": "Exploração de Campo Directa",
            "Área / Dimensão": "Calculando...",
            "Data de Emissão": "N/A",
            "Data de Validade (Expiry)": "N/A",
            "Tipo de Direito / Estado": "Área de Pesquisa Livre",
            "Substâncias": "Alvo Selecionado Manualmente"
        }
        st.rerun()

    st.write("### 📋 Registo Oficial em Tempo Real (Trimble Landfolio)")
    st.table(st.session_state["concession_metadata"])

# ========================================================
# 5. REMOTE SENSING TARGET CHANNELS & IBM ENGINE
# ========================================================
with col2:
    st.subheader("📊 5 Core Remote Sensing Target Frameworks")
    
    with st.spinner("Processing multi-spectral analytics over target concession..."):
        m_data = fetch_and_calculate_spatz(st.session_state["map_center"][0], st.session_state["map_center"][1], selected_year)
    
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
            meta = st.session_state["concession_metadata"]
            
            p1 = "[Role: Senior Exploration Geologist Expert in Mozambique Metallurgy]\nEvaluate remote sensing values for " + str(target_commodity) + " at coordinates " + str(st.session_state['map_center']) + " for year " + str(selected_year) + ".\n\n"
            p2 = "Mozambique Landfolio Cadastre Record:\n- License Code: " + str(meta['Código da Licença (Code)']) + "\n- Concession Name: " + str(meta['Nome da Concessão']) + "\n- Registered Operator: " + str(meta['Titular (Holder Company)']) + "\n- Size Layer: " + str(meta['Área / Dimensão']) + "\n- Expiry: " + str(meta['Data de Validade (Expiry)']) + "\n- Status: " + str(meta['Tipo de Direito / Estado']) + "\n- Listed Substances: " + str(meta['Substâncias']) + "\n\n"
            p3 = "Telemetry Matrix:\n- Iron Oxide: " + str(m_data['Way_1_Iron_Oxide_Gossan']) + ", Clay: " + str(m_data['Way_1_Clay_Phyllic']) + "\n- Structural Fault Density: " + str(m_data['Way_2_Fault_Density_Index']) + "\n- Silicification Cap: " + str(m_data['Way_3_Silica_Flooding_Cap']) + "\n- Vegetation Shift: " + str(m_data['Way_4_Geobotanical_Stress']) + "\n- WLC Score: " + str(m_data['Way_5_WLC_Score_Percent']) + "%\n\n"
            p4 = "Task:\nWrite a technical appraisal evaluating the target within the framework of Mozambican geology. Detail field investigation protocols near these concessions based on the target commodities listed."
            
            complete_prompt = p1 + p2 + p3 + p4
            
            model = ModelInference(model_id="ibm/granite-13b-instruct-v2", credentials=credentials, project_id=PROJECT_ID)
            st.markdown(model.generate_text(prompt=complete_prompt))
