import streamlit as st
st.set_page_config(page_title="SatIntel Moçambique Real-Time AI", layout="wide")

import folium
from streamlit_folium import st_folium
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from georemote import fetch_and_calculate_spatz, get_real_mozambique_cadastre, fetch_satellite_imagery
from fpdf import FPDF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class TechnicalReportPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(40, 40, 40)
        self.cell(0, 10, 'SATINTEL - GEOLOGICAL & MINING INSIGHTS', 0, 1, 'L')
        self.set_draw_color(200, 200, 200)
        self.line(10, 20, 200, 20)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Página {self.page_no()}/{{nb}} - Gerado Automaticamente via SatIntel AI', 0, 0, 'C')


# ========================================================
# 1. PLATFORM CONFIGURATION & SECURITY GATEWAY
# ========================================================
try:
    IBM_API_KEY = st.secrets["WATSONX_APIKEY"]
    PROJECT_ID = st.secrets["WATSONX_PROJECT_ID"]
except KeyError:
    st.error("🔒 Streamlit Secrets missing! Please verify your setup.")
    st.stop()

credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": IBM_API_KEY
}


@st.cache_resource
def get_watsonx_client():
    client = APIClient(credentials=credentials)
    client.set.default_project(PROJECT_ID)
    return client

# ========================================================
# 2. APPLICATION RUNTIME SESSION STATE
# ========================================================
# Default center: real coordinates of license 11521 (Tete Platinum concession)
if "map_center" not in st.session_state:
    st.session_state["map_center"] = [-15.095314, 32.567917]
if "active_polygon" not in st.session_state:
    st.session_state["active_polygon"] = None
if "concession_metadata" not in st.session_state:
    st.session_state["concession_metadata"] = {
        "Código da Licença (Code)": "Aguardando Consulta",
        "Nome da Concessão": "Aguardando Consulta",
        "Titular (Holder Company)": "Aguardando Banco de Dados Real",
        "Área / Dimensão": "0.00 Ha",
        "Data de Emissão": "N/A",
        "Data de Validade (Expiry)": "N/A",
        "Tipo de Direito / Estado": "N/A",
        "Substâncias": "N/A"
    }
if "satellite_data" not in st.session_state:
    st.session_state["satellite_data"] = None
if "m_data" not in st.session_state:
    st.session_state["m_data"] = None

# ========================================================
# 3. INTERFACE BUILDER & STREAMLIT LAYOUT
# ========================================================
st.title("🛰️ SatIntel: Mozambique Mining Cadastre Real-Time Platform")
st.caption("Live Production Database Synchronization with Landfolio MIREME REST API Servers")

st.sidebar.header("🎯 Portal de Seleção de Alvos")

selected_basemap = st.sidebar.selectbox(
    "🗺️ Select Map Layer View",
    ["Esri World Imagery (Satellite)", "Google Satellite Imagery", "OpenStreetMap (Standard)", "Esri Topographic Map"]
)

selected_year = st.sidebar.slider("Select Analysis Year", min_value=1990, max_value=2026, value=2024, step=1)
search_method = st.sidebar.radio("Select Landfolio Lookup Method", ["(a) License # Search", "(c) Map Selection"])

if search_method == "(a) License # Search":
    license_num = st.sidebar.text_input("Enter License Number (Real Database Match)", placeholder="e.g., 11521")
    if license_num:
        with st.sidebar.spinner("Buscando dados em tempo real no Cadastro Nacional (INAMI)..."):
            db_result = get_real_mozambique_cadastre(license_num)
            if db_result["found"]:
                st.session_state["map_center"] = [db_result["lat"], db_result["lon"]]
                st.session_state["active_polygon"] = db_result["polygon"]
                st.session_state["concession_metadata"] = db_result["metadata"]
                st.session_state["satellite_data"] = None  # force re-fetch
                st.session_state["m_data"] = None
                st.sidebar.success(f"✓ Concessão {license_num} carregada! Geometry fetched from INAMI.")
            else:
                st.sidebar.error(f"❌ Licença '{license_num}' não encontrada nos servidores INAMI.")

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

    # Use proper working tile URLs
    if selected_basemap == "Esri World Imagery (Satellite)":
        m = folium.Map(
            location=st.session_state["map_center"],
            zoom_start=11,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery"
        )
    elif selected_basemap == "Google Satellite Imagery":
        m = folium.Map(
            location=st.session_state["map_center"],
            zoom_start=11,
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google Satellite"
        )
    elif selected_basemap == "Esri Topographic Map":
        m = folium.Map(
            location=st.session_state["map_center"],
            zoom_start=11,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
            attr="Esri Topographic"
        )
    else:
        m = folium.Map(location=st.session_state["map_center"], zoom_start=11)

    # Draw the concession polygon if loaded
    if st.session_state["active_polygon"]:
        folium.GeoJson(
            st.session_state["active_polygon"],
            name="Real Concession Boundary",
            style_function=lambda x: {
                "fillColor": "#00E5FF",
                "color": "#FFD700",
                "weight": 4,
                "fillOpacity": 0.3
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["name"],
                aliases=["Concession:"],
                style="background-color: #004D40; color: white; font-weight: bold; padding: 5px; border-radius: 3px;"
            )
        ).add_to(m)

        # Also add a marker at the center
        folium.Marker(
            location=st.session_state["map_center"],
            tooltip=st.session_state["concession_metadata"].get("Nome da Concessão", "Concession Center"),
            icon=folium.Icon(color="red", icon="info-sign")
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
        st.session_state["satellite_data"] = None
        st.session_state["m_data"] = None
        st.rerun()

    st.write("### 📋 Registo Oficial em Tempo Real (Trimble Landfolio / INAMI)")
    st.table(st.session_state["concession_metadata"])

# ========================================================
# 5. REMOTE SENSING TARGET CHANNELS & IBM ENGINE
# ========================================================
with col2:
    st.subheader("📊 5 Core Remote Sensing Target Frameworks")

    # Fetch satellite data if not already loaded
    if st.session_state["m_data"] is None:
        with st.spinner("🛰️ Fetching Landsat satellite imagery & computing spectral indices..."):
            try:
                lat, lon = st.session_state["map_center"]
                sat_data = fetch_satellite_imagery(lat, lon, selected_year)
                st.session_state["satellite_data"] = sat_data
                st.session_state["m_data"] = {
                    "Way_1_Iron_Oxide_Gossan": sat_data["Way_1_Iron_Oxide_Gossan"],
                    "Way_1_Clay_Phyllic": sat_data["Way_1_Clay_Phyllic"],
                    "Way_2_Fault_Density_Index": sat_data["Way_2_Fault_Density_Index"],
                    "Way_3_Silica_Flooding_Cap": sat_data["Way_3_Silica_Flooding_Cap"],
                    "Way_4_Geobotanical_Stress": sat_data["Way_4_Geobotanical_Stress"],
                    "Way_5_WLC_Score_Percent": sat_data["Way_5_WLC_Score_Percent"],
                    "Satellite_Used": sat_data["Satellite_Used"],
                }
            except Exception as e:
                st.warning(f"⚠️ Satellite fetch failed: {str(e)[:120]}. Using predictive model values.")
                st.session_state["m_data"] = fetch_and_calculate_spatz(
                    st.session_state["map_center"], st.session_state["map_center"], selected_year
                )
                st.session_state["satellite_data"] = None

    m_data = st.session_state["m_data"]

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

# ========================================================
# 5b. SATELLITE IMAGERY & SPECTRAL INDEX VISUALIZATION
# ========================================================
sat_data = st.session_state.get("satellite_data")

if sat_data is not None:
    st.markdown("---")
    st.markdown("## 🛰️ Satellite Imagery & Spectral Index Maps")
    st.caption(f"Scene Date: {sat_data['scene_date']} | Cloud Cover: {sat_data['cloud_cover']}% | Source: {sat_data['Satellite_Used']}")

    # --- True Color & False Color Composites ---
    img_col1, img_col2 = st.columns(2)

    with img_col1:
        st.markdown("### 🌍 True Color Composite (RGB)")
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(sat_data["rgb"])
        ax.set_title("Natural Color — Landsat", fontsize=11, fontweight="bold")
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with img_col2:
        st.markdown("### 🔴 False Color (SWIR-NIR-Red)")
        st.caption("Mineral alteration enhancement — red/magenta tones indicate hydrothermal alteration zones")
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(sat_data["false_color"])
        ax.set_title("Mineral Enhancement Composite", fontsize=11, fontweight="bold")
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # --- Spectral Index Maps ---
    st.markdown("---")
    st.markdown("### 📐 Spectral Index Maps (Computed from Landsat Bands)")

    idx1, idx2 = st.columns(2)

    with idx1:
        st.markdown("#### 🔶 Iron Oxide (Gossans) Index")
        st.caption("Red/Blue ratio — highlights ferric iron oxide zones")
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(sat_data["iron_oxide_map"], cmap="RdYlBu_r")
        ax.set_title("Iron Oxide Ratio (Band 4 / Band 2)", fontsize=10)
        ax.axis("off")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Fe-Oxide Ratio", fontsize=9)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with idx2:
        st.markdown("#### 🟡 Clay/Hydroxyl Index")
        st.caption("SWIR1/SWIR2 ratio — highlights hydrothermal clay alteration")
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(sat_data["clay_map"], cmap="YlOrBr")
        ax.set_title("Clay Minerals Ratio (Band 6 / Band 7)", fontsize=10)
        ax.axis("off")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Clay Ratio", fontsize=9)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    idx3, idx4 = st.columns(2)

    with idx3:
        st.markdown("#### 🌿 NDVI — Vegetation Stress")
        st.caption("Negative values = bare rock/mineral exposure; positive = healthy vegetation")
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(sat_data["ndvi_map"], cmap="RdYlGn", vmin=-0.3, vmax=0.8)
        ax.set_title("NDVI (Band 5 - Band 4) / (Band 5 + Band 4)", fontsize=10)
        ax.axis("off")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("NDVI", fontsize=9)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with idx4:
        st.markdown("#### ⬜ Silica Proxy Index")
        st.caption("SWIR2/SWIR1 ratio — highlights silicified alteration zones")
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(sat_data["silica_map"], cmap="bone")
        ax.set_title("Silica Proxy (Band 7 / Band 6)", fontsize=10)
        ax.axis("off")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Silica Ratio", fontsize=9)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")
    st.info("ℹ️ All spectral indices computed from real Landsat Collection 2 Level-2 surface reflectance data via Microsoft Planetary Computer. Higher values in Iron Oxide and Clay maps indicate stronger alteration signatures.")

# ========================================================
# 6. IBM WATSONX GEOLOGICAL REPORT GENERATION
# ========================================================
st.markdown("---")
if st.button("🚀 Generate 5-Way Geological Synthesis", use_container_width=True):
    with st.spinner("O watsonx.ai está correlacionando as matrizes geológicas..."):
        client = get_watsonx_client()
        meta = st.session_state["concession_metadata"]

        p1 = "[Role: Geólogo Sénior de Exploração Especialista em Metalogenia de Moçambique]\n"
        p2 = "Execute uma avaliação geológica detalhada para o alvo: " + str(target_commodity) + " nas coordenadas " + str(st.session_state['map_center']) + " para o ano de " + str(selected_year) + ".\n\n"
        p3 = "Dados do Cadastro Mineiro (Trimble Landfolio Moçambique):\n"
        p4 = "- Código da Licença: " + str(meta.get('Código da Licença (Code)', '11521')) + "\n- Nome da Concessão: " + str(meta.get('Nome da Concessão', '')) + "\n- Titular: " + str(meta.get('Titular (Holder Company)', '')) + "\n- Dimensão: " + str(meta.get('Área / Dimensão', '')) + "\n- Validade: " + str(meta.get('Data de Validade (Expiry)', '')) + "\n- Substâncias Registadas: " + str(meta.get('Substâncias', '')) + "\n\n"
        p5 = "Matriz de Telemetria de Detecção Remota (5-Way Model - Landsat Real Data):\n"
        p6 = "- Óxido de Ferro (Gossans): " + str(m_data.get('Way_1_Iron_Oxide_Gossan', 2.4)) + "\n- Índice de Argila/Hidroxilo: " + str(m_data.get('Way_1_Clay_Phyllic', 1.9)) + "\n- Densidade de Falhas Estruturais: " + str(m_data.get('Way_2_Fault_Density_Index', 0.8)) + "\n- Indicador de Silicification: " + str(m_data.get('Way_3_Silica_Flooding_Cap', 0.6)) + "\n- Estresse Geobotânico (NDVI): " + str(m_data.get('Way_4_Geobotanical_Stress', 0.34)) + "\n- Pontuação de Prospectivity Combinada (WLC): " + str(m_data.get('Way_5_WLC_Score_Percent', 88.5)) + "%\n\n"
        p7 = "Directrizes da Tarefa:\n"
        p8 = "Escreva um parecer técnico formal em português. Analise a associação entre o Ouro/Platina e os minerais pegmatíticos listados (Lítio, Turmalinas, Tantalite). Avalie o significado do estresse geobotânico observado e a densidade estrutural. Conclua com recomendações claras de campo (amostragem de solo ou abertura de trincheiras) e um parecer final de 'Perfurar / Não Perfurar' (Drill/No-Drill)."

        complete_prompt = p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8

        model = ModelInference(
            model_id="meta-llama/llama-3-3-70b-instruct",
            credentials=credentials,
            project_id=PROJECT_ID,
            params={
                "max_new_tokens": 1500,
                "temperature": 0.7
            }
        )
        result = model.generate_text(prompt=complete_prompt)
        st.markdown(result)
