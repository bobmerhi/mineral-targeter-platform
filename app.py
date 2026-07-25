import streamlit as st
st.set_page_config(page_title="SatIntel Moçambique Real-Time AI", layout="wide")

import folium
from streamlit_folium import st_folium
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from georemote import (
    fetch_and_calculate_spatz,
    get_real_mozambique_cadastre,
    fetch_satellite_imagery,
    polygon_to_bbox,
)
from fpdf import FPDF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
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
# PLATFORM CONFIGURATION
# ========================================================
try:
    IBM_API_KEY = st.secrets["WATSONX_APIKEY"]
    PROJECT_ID  = st.secrets["WATSONX_PROJECT_ID"]
except KeyError:
    st.error("🔒 Streamlit Secrets missing!")
    st.stop()

credentials = {"url": "https://us-south.ml.cloud.ibm.com", "apikey": IBM_API_KEY}


@st.cache_resource
def get_watsonx_client():
    client = APIClient(credentials=credentials)
    client.set.default_project(PROJECT_ID)
    return client


# ========================================================
# HELPER: draw polygon boundary on a matplotlib axis
# ========================================================
def draw_polygon_on_ax(ax, polygon_geojson, fetch_bbox, img_shape):
    """
    Draw the concession polygon on a matplotlib image axis.
    Converts geographic coordinates to pixel coordinates using fetch_bbox.
    fetch_bbox = [lon_min, lat_min, lon_max, lat_max]
    img_shape  = (height, width) of the numpy image array
    """
    if not polygon_geojson or fetch_bbox is None:
        return
    try:
        lon_min, lat_min, lon_max, lat_max = fetch_bbox
        h, w = img_shape[:2]

        def geo_to_px(lon, lat):
            # Map lon→x (left to right), lat→y (top to bottom, lat decreases downward in image)
            x = (lon - lon_min) / (lon_max - lon_min) * w
            y = (lat_max - lat) / (lat_max - lat_min) * h
            return x, y

        rings = polygon_geojson["geometry"]["coordinates"]
        for ring in rings:
            px_coords = [geo_to_px(p[0], p[1]) for p in ring]
            xs = [c[0] for c in px_coords]
            ys = [c[1] for c in px_coords]

            # Filled semi-transparent polygon
            patch = MplPolygon(
                list(zip(xs, ys)),
                closed=True,
                facecolor="cyan",
                alpha=0.15,
                edgecolor="yellow",
                linewidth=2.5,
                zorder=5,
                transform=ax.transData,
            )
            ax.add_patch(patch)
            # Bold boundary line
            ax.plot(xs + [xs[0]], ys + [ys[0]], color="#FFD700", linewidth=2.5, zorder=6)

        # Keep axes in pixel space (no axis labels)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)   # invert y so image top = lat_max
        ax.axis("off")

    except Exception:
        ax.axis("off")


# ========================================================
# SESSION STATE
# ========================================================
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
# LAYOUT
# ========================================================
st.title("🛰️ SatIntel: Mozambique Mining Cadastre Real-Time Platform")
st.caption("Live Production Database Synchronization with Landfolio MIREME REST API Servers")

st.sidebar.header("🎯 Portal de Seleção de Alvos")

selected_basemap = st.sidebar.selectbox(
    "🗺️ Select Map Layer View",
    ["Esri World Imagery (Satellite)", "Google Satellite Imagery", "OpenStreetMap (Standard)", "Esri Topographic Map"]
)
selected_year  = st.sidebar.slider("Select Analysis Year", 1990, 2026, 2024)
search_method  = st.sidebar.radio("Select Landfolio Lookup Method", ["(a) License # Search", "(c) Map Selection"])

if search_method == "(a) License # Search":
    license_num = st.sidebar.text_input("Enter License Number (Real Database Match)", placeholder="e.g., 11521")
    if license_num:
        with st.sidebar.spinner("Buscando dados no Cadastro Nacional (INAMI)..."):
            db_result = get_real_mozambique_cadastre(license_num)
            if db_result["found"]:
                st.session_state["map_center"]         = [db_result["lat"], db_result["lon"]]
                st.session_state["active_polygon"]      = db_result["polygon"]
                st.session_state["concession_metadata"] = db_result["metadata"]
                st.session_state["satellite_data"]      = None
                st.session_state["m_data"]              = None
                st.sidebar.success(f"✓ Concessão {license_num} carregada! Geometry from INAMI.")
            else:
                st.sidebar.error(f"❌ Licença '{license_num}' não encontrada.")
else:
    st.sidebar.info("👉 Clique no mapa para selecionar coordenadas.")

st.sidebar.divider()
target_commodity = st.sidebar.selectbox("Commodity Focus", ["Gold", "Copper", "Lithium", "Heavy Mineral Sands", "Emeralds"])

# ========================================================
# MAP + METADATA
# ========================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🗺️ Live Geographic Registry View")

    if selected_basemap == "Esri World Imagery (Satellite)":
        m = folium.Map(
            location=st.session_state["map_center"], zoom_start=10,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery"
        )
    elif selected_basemap == "Google Satellite Imagery":
        m = folium.Map(
            location=st.session_state["map_center"], zoom_start=10,
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google Satellite"
        )
    elif selected_basemap == "Esri Topographic Map":
        m = folium.Map(
            location=st.session_state["map_center"], zoom_start=10,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
            attr="Esri Topographic"
        )
    else:
        m = folium.Map(location=st.session_state["map_center"], zoom_start=10)

    if st.session_state["active_polygon"]:
        folium.GeoJson(
            st.session_state["active_polygon"],
            name="Concession Boundary",
            style_function=lambda x: {"fillColor": "#00E5FF", "color": "#FFD700", "weight": 4, "fillOpacity": 0.3},
            tooltip=folium.GeoJsonTooltip(
                fields=["name"], aliases=["Concession:"],
                style="background-color:#004D40;color:white;font-weight:bold;padding:5px;border-radius:3px;"
            )
        ).add_to(m)
        folium.Marker(
            location=st.session_state["map_center"],
            tooltip=st.session_state["concession_metadata"].get("Nome da Concessão", "Center"),
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)

    map_data = st_folium(m, width=550, height=380, key=f"map_{selected_basemap}_{st.session_state['map_center']}")

    if search_method == "(c) Map Selection" and map_data and map_data.get("last_clicked"):
        cp = map_data["last_clicked"]
        lat, lng = cp["lat"], cp["lng"]
        st.session_state["map_center"]         = [lat, lng]
        st.session_state["active_polygon"]      = None
        st.session_state["concession_metadata"] = {
            "Código da Licença (Code)": "Coordenadas Manuais",
            "Nome da Concessão": f"Ponto ({lat:.4f}, {lng:.4f})",
            "Titular (Holder Company)": "Campo Livre",
            "Área / Dimensão": "N/A",
            "Data de Emissão": "N/A",
            "Data de Validade (Expiry)": "N/A",
            "Tipo de Direito / Estado": "Área Livre",
            "Substâncias": "Seleção Manual"
        }
        st.session_state["satellite_data"] = None
        st.session_state["m_data"]         = None
        st.rerun()

    st.write("### 📋 Registo Oficial em Tempo Real (Trimble Landfolio / INAMI)")
    st.table(st.session_state["concession_metadata"])

# ========================================================
# 5-WAY REMOTE SENSING METRICS
# ========================================================
with col2:
    st.subheader("📊 5 Core Remote Sensing Target Frameworks")

    if st.session_state["m_data"] is None:
        with st.spinner("🛰️ Fetching Landsat satellite imagery..."):
            try:
                lat, lon   = st.session_state["map_center"]
                active_poly = st.session_state.get("active_polygon")

                # Use polygon bbox so the full concession is covered
                poly_bbox  = polygon_to_bbox(active_poly) if active_poly else None
                sat_data   = fetch_satellite_imagery(lat, lon, selected_year, bbox=poly_bbox)

                st.session_state["satellite_data"] = sat_data
                st.session_state["m_data"] = {
                    "Way_1_Iron_Oxide_Gossan":   sat_data["Way_1_Iron_Oxide_Gossan"],
                    "Way_1_Clay_Phyllic":         sat_data["Way_1_Clay_Phyllic"],
                    "Way_2_Fault_Density_Index":  sat_data["Way_2_Fault_Density_Index"],
                    "Way_3_Silica_Flooding_Cap":  sat_data["Way_3_Silica_Flooding_Cap"],
                    "Way_4_Geobotanical_Stress":  sat_data["Way_4_Geobotanical_Stress"],
                    "Way_5_WLC_Score_Percent":    sat_data["Way_5_WLC_Score_Percent"],
                    "Satellite_Used":             sat_data["Satellite_Used"],
                }
            except Exception as e:
                st.warning(f"⚠️ Satellite fetch failed: {str(e)[:120]}. Using predictive values.")
                st.session_state["m_data"] = fetch_and_calculate_spatz(
                    st.session_state["map_center"], None, selected_year
                )
                st.session_state["satellite_data"] = None

    m_data = st.session_state["m_data"]

    st.markdown("#### **WAY 1: Hydrothermal Alteration**")
    w1c1, w1c2 = st.columns(2)
    w1c1.metric("Iron Oxide (Gossans)",  m_data["Way_1_Iron_Oxide_Gossan"])
    w1c2.metric("Clay/Hydroxyl Index",   m_data["Way_1_Clay_Phyllic"])

    st.markdown("#### **WAY 2: Structural Lineaments**")
    st.metric("Fault Intersection Density", m_data["Way_2_Fault_Density_Index"])

    st.markdown("#### **WAY 3: Lithological Silicification**")
    st.metric("Quartz Veining Emissivity",  m_data["Way_3_Silica_Flooding_Cap"])

    st.markdown("#### **WAY 4: Geobotanical Stress**")
    st.metric("Vegetation Stress Proxy (NDVI)", m_data["Way_4_Geobotanical_Stress"])

    st.markdown("#### **WAY 5: GIS Predictive Synthesis**")
    st.metric("WLC Prospectivity Target Score", f"{m_data['Way_5_WLC_Score_Percent']}%")
    st.caption(f"🛰️ Source Pipeline ID: {m_data['Satellite_Used']}")
    st.divider()

# ========================================================
# SATELLITE IMAGES WITH POLYGON OVERLAY (pixel-accurate)
# ========================================================
sat_data = st.session_state.get("satellite_data")

if sat_data is not None:
    st.markdown("---")
    st.markdown("## 🛰️ Satellite Imagery & Spectral Index Maps")
    st.caption(
        f"Scene Date: {sat_data['scene_date']} | "
        f"Cloud Cover: {sat_data['cloud_cover']}% | "
        f"Source: {sat_data['Satellite_Used']}"
    )

    active_poly = st.session_state.get("active_polygon")
    fetch_bbox  = sat_data.get("fetch_bbox")   # [lon_min, lat_min, lon_max, lat_max]

    if active_poly:
        lon_min, lat_min, lon_max, lat_max = fetch_bbox
        st.success(
            f"📍 Concession polygon overlay active — "
            f"bbox: [{lon_min:.3f}°, {lat_min:.3f}°, {lon_max:.3f}°, {lat_max:.3f}°]"
        )

    def make_fig(img_array, cmap=None, vmin=None, vmax=None, title="", label=""):
        """Render one image with polygon overlay. Returns (fig, ax)."""
        fig, ax = plt.subplots(figsize=(7, 6))
        kw = {}
        if vmin is not None: kw["vmin"] = vmin
        if vmax is not None: kw["vmax"] = vmax
        if cmap:
            im = ax.imshow(img_array, cmap=cmap, **kw)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(label, fontsize=9)
        else:
            ax.imshow(img_array, **kw)
        ax.set_title(title, fontsize=10, fontweight="bold")
        if active_poly and fetch_bbox:
            draw_polygon_on_ax(ax, active_poly, fetch_bbox, img_array.shape)
        else:
            ax.axis("off")
        return fig

    # Row 1 – RGB & False Color
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.markdown("### 🌍 True Color Composite (RGB)")
        fig = make_fig(sat_data["rgb"], title="Natural Color — Landsat")
        st.pyplot(fig, use_container_width=True); plt.close()

    with img_col2:
        st.markdown("### 🔴 False Color (SWIR-NIR-Red)")
        st.caption("Red/magenta = hydrothermal alteration zones")
        fig = make_fig(sat_data["false_color"], title="Mineral Enhancement Composite")
        st.pyplot(fig, use_container_width=True); plt.close()

    # Row 2 – Spectral indices
    st.markdown("---")
    st.markdown("### 📐 Spectral Index Maps (Computed from Landsat Bands)")

    idx1, idx2 = st.columns(2)
    with idx1:
        st.markdown("#### 🔶 Iron Oxide (Gossans) Index")
        st.caption("Red/Blue ratio — ferric iron oxide zones")
        fig = make_fig(sat_data["iron_oxide_map"], cmap="RdYlBu_r",
                       title="Iron Oxide Ratio (Band 4 / Band 2)", label="Fe-Oxide Ratio")
        st.pyplot(fig, use_container_width=True); plt.close()

    with idx2:
        st.markdown("#### 🟡 Clay/Hydroxyl Index")
        st.caption("SWIR1/SWIR2 — hydrothermal clay alteration")
        fig = make_fig(sat_data["clay_map"], cmap="YlOrBr",
                       title="Clay Minerals Ratio (Band 6 / Band 7)", label="Clay Ratio")
        st.pyplot(fig, use_container_width=True); plt.close()

    idx3, idx4 = st.columns(2)
    with idx3:
        st.markdown("#### 🌿 NDVI — Vegetation Stress")
        st.caption("Negative = bare rock; positive = healthy vegetation")
        fig = make_fig(sat_data["ndvi_map"], cmap="RdYlGn", vmin=-0.3, vmax=0.8,
                       title="NDVI (Band 5 - Band 4) / (Band 5 + Band 4)", label="NDVI")
        st.pyplot(fig, use_container_width=True); plt.close()

    with idx4:
        st.markdown("#### ⬜ Silica Proxy Index")
        st.caption("SWIR2/SWIR1 — silicified alteration zones")
        fig = make_fig(sat_data["silica_map"], cmap="bone",
                       title="Silica Proxy (Band 7 / Band 6)", label="Silica Ratio")
        st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("---")
    st.info(
        "ℹ️ Yellow polygon = real concession boundary from INAMI cadastre. "
        "All spectral indices from Landsat Collection 2 Level-2 surface reflectance "
        "via Microsoft Planetary Computer."
    )

# ========================================================
# IBM WATSONX GEOLOGICAL REPORT
# ========================================================
st.markdown("---")
if st.button("🚀 Generate 5-Way Geological Synthesis", use_container_width=True):
    with st.spinner("O watsonx.ai está correlacionando as matrizes geológicas..."):
        client = get_watsonx_client()
        meta   = st.session_state["concession_metadata"]

        prompt = (
            "[Role: Geólogo Sénior de Exploração Especialista em Metalogenia de Moçambique]\n"
            f"Execute uma avaliação geológica detalhada para o alvo: {target_commodity} "
            f"nas coordenadas {st.session_state['map_center']} para o ano de {selected_year}.\n\n"
            "Dados do Cadastro Mineiro (Trimble Landfolio Moçambique):\n"
            f"- Código da Licença: {meta.get('Código da Licença (Code)', 'N/A')}\n"
            f"- Nome da Concessão: {meta.get('Nome da Concessão', '')}\n"
            f"- Titular: {meta.get('Titular (Holder Company)', '')}\n"
            f"- Dimensão: {meta.get('Área / Dimensão', '')}\n"
            f"- Validade: {meta.get('Data de Validade (Expiry)', '')}\n"
            f"- Substâncias: {meta.get('Substâncias', '')}\n\n"
            "Matriz de Telemetria de Detecção Remota (5-Way Model):\n"
            f"- Óxido de Ferro (Gossans): {m_data.get('Way_1_Iron_Oxide_Gossan', 2.4)}\n"
            f"- Índice de Argila/Hidroxilo: {m_data.get('Way_1_Clay_Phyllic', 1.9)}\n"
            f"- Densidade de Falhas: {m_data.get('Way_2_Fault_Density_Index', 0.8)}\n"
            f"- Silicification: {m_data.get('Way_3_Silica_Flooding_Cap', 0.6)}\n"
            f"- Estresse Geobotânico (NDVI): {m_data.get('Way_4_Geobotanical_Stress', 0.34)}\n"
            f"- WLC Prospectivity: {m_data.get('Way_5_WLC_Score_Percent', 88.5)}%\n\n"
            "Escreva um parecer técnico formal em português. Analise a associação mineralógica. "
            "Conclua com recomendações de campo e parecer 'Perfurar / Não Perfurar'."
        )

        model = ModelInference(
            model_id="meta-llama/llama-3-3-70b-instruct",
            credentials=credentials,
            project_id=PROJECT_ID,
            params={"max_new_tokens": 1500, "temperature": 0.7}
        )
        st.markdown(model.generate_text(prompt=prompt))
