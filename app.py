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
from export_utils import (
    polygon_to_kml,
    create_kmz_bundle,
    create_geotiff_bundle,
    create_png_bundle,
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
# HELPER: draw polygon on matplotlib axes (pixel coordinates)
# ========================================================
def draw_polygon_on_ax(ax, polygon_geojson, fetch_bbox, img_shape):
    if not polygon_geojson or fetch_bbox is None:
        return
    try:
        lon_min, lat_min, lon_max, lat_max = fetch_bbox
        h, w = img_shape[:2]

        def geo_to_px(lon, lat):
            x = (lon - lon_min) / (lon_max - lon_min) * w
            y = (lat_max - lat) / (lat_max - lat_min) * h
            return x, y

        rings = polygon_geojson["geometry"]["coordinates"]
        for ring in rings:
            px = [geo_to_px(p[0], p[1]) for p in ring]
            xs = [c[0] for c in px]
            ys = [c[1] for c in px]
            patch = MplPolygon(list(zip(xs, ys)), closed=True,
                              facecolor="cyan", alpha=0.15,
                              edgecolor="yellow", linewidth=2.5, zorder=5)
            ax.add_patch(patch)
            ax.plot(xs + [xs[0]], ys + [ys[0]], color="#FFD700", linewidth=2.5, zorder=6)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
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
    license_num = st.sidebar.text_input("Enter License Number", placeholder="e.g., 11521")
    if license_num:
        with st.sidebar.spinner("Buscando no Cadastro (INAMI)..."):
            db_result = get_real_mozambique_cadastre(license_num)
            if db_result["found"]:
                st.session_state["map_center"]         = [db_result["lat"], db_result["lon"]]
                st.session_state["active_polygon"]      = db_result["polygon"]
                st.session_state["concession_metadata"] = db_result["metadata"]
                st.session_state["satellite_data"]      = None
                st.session_state["m_data"]              = None
                st.sidebar.success(f"✓ Concessão {license_num} carregada!")
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
        m = folium.Map(location=st.session_state["map_center"], zoom_start=10,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="Esri")
    elif selected_basemap == "Google Satellite Imagery":
        m = folium.Map(location=st.session_state["map_center"], zoom_start=10,
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", attr="Google")
    elif selected_basemap == "Esri Topographic Map":
        m = folium.Map(location=st.session_state["map_center"], zoom_start=10,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}", attr="Esri Topo")
    else:
        m = folium.Map(location=st.session_state["map_center"], zoom_start=10)

    if st.session_state["active_polygon"]:
        folium.GeoJson(st.session_state["active_polygon"], name="Concession Boundary",
            style_function=lambda x: {"fillColor": "#00E5FF", "color": "#FFD700", "weight": 4, "fillOpacity": 0.3},
            tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=["Concession:"],
                style="background-color:#004D40;color:white;font-weight:bold;padding:5px;border-radius:3px;")
        ).add_to(m)
        folium.Marker(location=st.session_state["map_center"],
            tooltip=st.session_state["concession_metadata"].get("Nome da Concessão", "Center"),
            icon=folium.Icon(color="red", icon="info-sign")).add_to(m)

    map_data = st_folium(m, width=550, height=380, key=f"map_{selected_basemap}_{st.session_state['map_center']}")

    if search_method == "(c) Map Selection" and map_data and map_data.get("last_clicked"):
        cp = map_data["last_clicked"]
        lat, lng = cp["lat"], cp["lng"]
        st.session_state["map_center"] = [lat, lng]
        st.session_state["active_polygon"] = None
        st.session_state["concession_metadata"] = {
            "Código da Licença (Code)": "Manual", "Nome da Concessão": f"({lat:.4f}, {lng:.4f})",
            "Titular (Holder Company)": "Campo Livre", "Área / Dimensão": "N/A",
            "Data de Emissão": "N/A", "Data de Validade (Expiry)": "N/A",
            "Tipo de Direito / Estado": "Área Livre", "Substâncias": "Seleção Manual"
        }
        st.session_state["satellite_data"] = None
        st.session_state["m_data"] = None
        st.rerun()

    st.write("### 📋 Registo Oficial (Trimble Landfolio / INAMI)")
    st.table(st.session_state["concession_metadata"])

# ========================================================
# 5-WAY METRICS
# ========================================================
with col2:
    st.subheader("📊 5 Core Remote Sensing Target Frameworks")

    if st.session_state["m_data"] is None:
        with st.spinner("🛰️ Fetching Landsat imagery & computing spectral indices, PCA & lineaments..."):
            try:
                lat, lon = st.session_state["map_center"]
                active_poly = st.session_state.get("active_polygon")
                poly_bbox = polygon_to_bbox(active_poly) if active_poly else None
                sat_data = fetch_satellite_imagery(lat, lon, selected_year, bbox=poly_bbox)
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
                st.session_state["m_data"] = fetch_and_calculate_spatz(st.session_state["map_center"], None, selected_year)
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
    st.caption(f"🛰️ {m_data['Satellite_Used']}")
    st.divider()

# ========================================================
# SATELLITE IMAGERY + SPECTRAL INDEX MAPS
# ========================================================
sat_data = st.session_state.get("satellite_data")

if sat_data is not None:
    active_poly = st.session_state.get("active_polygon")
    fetch_bbox  = sat_data.get("fetch_bbox")

    # --- Standard 6 images ---
    st.markdown("---")
    st.markdown("## 🛰️ Satellite Imagery & Spectral Index Maps")
    st.caption(f"Scene: {sat_data['scene_date']} | Cloud: {sat_data['cloud_cover']}% | {sat_data['Satellite_Used']}")
    if active_poly:
        st.success("📍 Concession polygon overlay active on all images below.")

    def make_fig(img_array, cmap=None, vmin=None, vmax=None, title="", label=""):
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

    # Row 1
    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown("### 🌍 True Color (RGB)")
        st.pyplot(make_fig(sat_data["rgb"], title="Natural Color — Landsat"), use_container_width=True); plt.close()
    with ic2:
        st.markdown("### 🔴 False Color (SWIR-NIR-Red)")
        st.caption("Red/magenta = alteration zones")
        st.pyplot(make_fig(sat_data["false_color"], title="Mineral Enhancement Composite"), use_container_width=True); plt.close()

    # Row 2
    st.markdown("---")
    st.markdown("### 📐 Spectral Index Maps")
    ix1, ix2 = st.columns(2)
    with ix1:
        st.markdown("#### 🔶 Iron Oxide (Band Ratio)")
        st.pyplot(make_fig(sat_data["iron_oxide_map"], cmap="RdYlBu_r", title="Iron Oxide Ratio (B4/B2)", label="Fe-Oxide"), use_container_width=True); plt.close()
    with ix2:
        st.markdown("#### 🟡 Clay/Hydroxyl (Band Ratio)")
        st.pyplot(make_fig(sat_data["clay_map"], cmap="YlOrBr", title="Clay Ratio (B6/B7)", label="Clay"), use_container_width=True); plt.close()

    ix3, ix4 = st.columns(2)
    with ix3:
        st.markdown("#### 🌿 NDVI — Vegetation Stress")
        st.pyplot(make_fig(sat_data["ndvi_map"], cmap="RdYlGn", vmin=-0.3, vmax=0.8, title="NDVI", label="NDVI"), use_container_width=True); plt.close()
    with ix4:
        st.markdown("#### ⬜ Silica Proxy")
        st.pyplot(make_fig(sat_data["silica_map"], cmap="bone", title="Silica Proxy (B7/B6)", label="Silica"), use_container_width=True); plt.close()

    # ========================================================
    # CROSTA PCA ALTERATION ANALYSIS
    # ========================================================
    st.markdown("---")
    st.markdown("## 🔬 Crosta PCA — Hydrothermal Alteration Analysis")
    st.caption("Feature-Oriented Principal Component Analysis (Crosta Technique) — targeted PCA on Landsat band subsets to isolate alteration mineral signatures.")

    iron_load = sat_data.get("crosta_iron_loadings", {})
    clay_load = sat_data.get("crosta_clay_loadings", {})

    lc1, lc2 = st.columns(2)
    with lc1:
        st.markdown("#### Iron Oxide PCA Eigenvector Loadings")
        st.markdown(f"Selected **PC{sat_data.get('crosta_iron_pc', '?')+1}** (strongest Red vs Blue contrast)")
        st.dataframe({"Band": list(iron_load.keys()), "Loading": list(iron_load.values())}, use_container_width=True, hide_index=True)
        st.metric("Iron Oxide Anomaly Coverage", f"{sat_data.get('crosta_iron_anomaly_pct', 0)}%")
    with lc2:
        st.markdown("#### Clay/Hydroxyl PCA Eigenvector Loadings")
        st.markdown(f"Selected **PC{sat_data.get('crosta_clay_pc', '?')+1}** (strongest SWIR1 vs SWIR2 contrast)")
        st.dataframe({"Band": list(clay_load.keys()), "Loading": list(clay_load.values())}, use_container_width=True, hide_index=True)
        st.metric("Clay Alteration Anomaly Coverage", f"{sat_data.get('crosta_clay_anomaly_pct', 0)}%")

    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown("#### 🔶 Crosta Iron Oxide PCA")
        st.caption("Bright = gossan/iron-stained zones")
        st.pyplot(make_fig(sat_data["crosta_iron_pca"], cmap="RdYlBu_r", title=f"Crosta Iron Oxide (PC{sat_data.get('crosta_iron_pc', 0)+1})", label="PC Score"), use_container_width=True); plt.close()
    with pc2:
        st.markdown("#### 🟡 Crosta Clay/Hydroxyl PCA")
        st.caption("Bright = argillic/phyllic alteration zones")
        st.pyplot(make_fig(sat_data["crosta_clay_pca"], cmap="YlOrBr", title=f"Crosta Clay (PC{sat_data.get('crosta_clay_pc', 0)+1})", label="PC Score"), use_container_width=True); plt.close()

    st.info("ℹ️ The Crosta Technique identifies which Principal Component captures the spectral contrast between target mineral bands. Bright pixels = concentrated alteration minerals — direct indicators of hydrothermal gold systems.")

    # ========================================================
    # STRUCTURAL LINEAMENT & INTERSECTION ANALYSIS
    # ========================================================
    st.markdown("---")
    st.markdown("## 🏔️ Structural Lineament & Intersection Analysis")
    st.caption("Directional Sobel filtering on SWIR1 imagery to detect faults, fractures, and shear zones. Intersection points = highest-prospectivity structural nodes for gold mineralization.")

    lm1, lm2 = st.columns(2)
    with lm1:
        st.markdown("#### 📏 Lineament Density Map")
        st.pyplot(make_fig(sat_data["lineament_density_map"], cmap="hot", title="Structural Lineament Density", label="Density (0-4)"), use_container_width=True); plt.close()
    with lm2:
        st.markdown("#### ⚡ Lineament Intersection Map")
        st.pyplot(make_fig(sat_data["intersection_map"], cmap="magma", title="Lineament Intersection Density", label="Intersection Index"), use_container_width=True); plt.close()

    st.markdown("---")
    st.markdown("### 🧭 Per-Orientation Lineament Maps")
    ori1, ori2 = st.columns(2)
    with ori1:
        st.markdown("#### ↕️ N-S Lineaments")
        st.pyplot(make_fig(sat_data["lineament_ns_map"], cmap="gray", title="North-South Lineaments", label="Binary"), use_container_width=True); plt.close()
    with ori2:
        st.markdown("#### ↔️ E-W Lineaments")
        st.pyplot(make_fig(sat_data["lineament_ew_map"], cmap="gray", title="East-West Lineaments", label="Binary"), use_container_width=True); plt.close()

    ori3, ori4 = st.columns(2)
    with ori3:
        st.markdown("#### ↗️ NE-SW Lineaments")
        st.pyplot(make_fig(sat_data["lineament_nesw_map"], cmap="gray", title="NE-SW Lineaments", label="Binary"), use_container_width=True); plt.close()
    with ori4:
        st.markdown("#### ↖️ NW-SE Lineaments")
        st.pyplot(make_fig(sat_data["lineament_nwse_map"], cmap="gray", title="NW-SE Lineaments", label="Binary"), use_container_width=True); plt.close()

    st.markdown("---")
    lm_c1, lm_c2, lm_c3 = st.columns(3)
    lm_c1.metric("Lineament Density Index", sat_data.get("lineament_density_val", 0))
    lm_c2.metric("High-Confidence Intersections", sat_data.get("intersection_count", 0))
    lm_c3.metric("Intersection Density Index", sat_data.get("intersection_density_val", 0))

    st.info("ℹ️ Gold-bearing fluids travel along faults and fractures. When structures of different orientations intersect, they create zones of high permeability where gold precipitates. The intersection map highlights these critical target nodes within the concession.")

    # ========================================================
    # EXPORT & GOOGLE EARTH INTEGRATION
    # ========================================================
    st.markdown("---")
    st.markdown("## 📥 Export & Google Earth Integration")
    st.caption("Export the concession geometry and all satellite imagery / spectral index maps in various formats for use in Google Earth, QGIS, ArcGIS, or reports.")

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.markdown("### 📐 Concession Geometry")

        # KML export for polygon
        if active_poly:
            kml_str = polygon_to_kml(active_poly, st.session_state.get("concession_metadata"))
            if kml_str:
                st.download_button(
                    label="📐 Export Concession Boundary (KML)",
                    data=kml_str.encode("utf-8"),
                    file_name=f"concession_{st.session_state['concession_metadata'].get('Código da Licença (Code)', 'unknown')}.kml",
                    mime="application/vnd.google-earth.kml+xml",
                    use_container_width=True,
                )
                st.caption("Opens directly in Google Earth — polygon with concession metadata")
        else:
            st.info("Load a license to export the concession geometry.")

        # GeoJSON export
        if active_poly:
            import json
            geojson_bytes = json.dumps(active_poly, indent=2).encode("utf-8")
            st.download_button(
                label="🗺️ Export Concession Boundary (GeoJSON)",
                data=geojson_bytes,
                file_name=f"concession_{st.session_state['concession_metadata'].get('Código da Licença (Code)', 'unknown')}.geojson",
                mime="application/geo+json",
                use_container_width=True,
            )
            st.caption("For QGIS, ArcGIS, or any GeoJSON-compatible tool")

    with exp_col2:
        st.markdown("### 🛰️ Satellite Image Exports")

        # KMZ bundle (Google Earth)
        kmz_bytes = create_kmz_bundle(
            sat_data,
            polygon_geojson=active_poly,
            metadata=st.session_state.get("concession_metadata"),
            fetch_bbox=fetch_bbox,
        )
        if kmz_bytes:
            st.download_button(
                label="🌍 Export All Overlays (KMZ — Google Earth)",
                data=kmz_bytes,
                file_name=f"satintel_overlays_{sat_data.get('scene_date', '')}.kmz",
                mime="application/vnd.google-earth.kmz",
                use_container_width=True,
            )
            st.caption("Contains polygon + 10 georeferenced image overlays. Just open in Google Earth.")

    # GeoTIFF and PNG bundles
    st.markdown("---")
    exp2_c1, exp2_c2 = st.columns(2)

    with exp2_c1:
        geotiff_bytes = create_geotiff_bundle(sat_data, fetch_bbox=fetch_bbox)
        if geotiff_bytes:
            st.download_button(
                label="📊 Export All Rasters (GeoTIFF — QGIS/ArcGIS)",
                data=geotiff_bytes,
                file_name=f"satintel_geotiffs_{sat_data.get('scene_date', '')}.zip",
                mime="application/zip",
                use_container_width=True,
            )
            st.caption("10 GeoTIFF rasters (EPSG:4326) — georeferenced, ready for GIS analysis")
        else:
            st.warning("GeoTIFF export requires rasterio (already in requirements).")

    with exp2_c2:
        png_bytes = create_png_bundle(sat_data)
        if png_bytes:
            st.download_button(
                label="🖼️ Export All Images (PNG — Reports)",
                data=png_bytes,
                file_name=f"satintel_images_{sat_data.get('scene_date', '')}.zip",
                mime="application/zip",
                use_container_width=True,
            )
            st.caption("10 high-res PNGs — for presentations, reports, and documentation")

    st.markdown("---")
    st.info("ℹ️ **KMZ** = Google Earth (polygon + image overlays georeferenced automatically). **GeoTIFF** = QGIS/ArcGIS (rasters with coordinate system). **PNG** = reports & presentations. **KML** = concession boundary only (opens in Google Earth).")

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
            f"- WLC Prospectivity: {m_data.get('Way_5_WLC_Score_Percent', 88.5)}%\n"
        )

        if sat_data:
            prompt += (
                "\nAnálise Crosta PCA:\n"
                f"- Iron Oxide PCA (PC{sat_data.get('crosta_iron_pc',0)+1}): mean={sat_data.get('crosta_iron_mean',0)}, "
                f"anomaly coverage={sat_data.get('crosta_iron_anomaly_pct',0)}%\n"
                f"- Clay PCA (PC{sat_data.get('crosta_clay_pc',0)+1}): mean={sat_data.get('crosta_clay_mean',0)}, "
                f"anomaly coverage={sat_data.get('crosta_clay_anomaly_pct',0)}%\n"
                f"- Iron loadings: {sat_data.get('crosta_iron_loadings', {})}\n"
                f"- Clay loadings: {sat_data.get('crosta_clay_loadings', {})}\n\n"
                "Análise Estrutural (Lineamentos):\n"
                f"- Densidade de Lineamentos: {sat_data.get('lineament_density_val', 0)}\n"
                f"- Intersecções de Alta Confiança: {sat_data.get('intersection_count', 0)}\n"
                f"- Índice de Densidade de Intersecção: {sat_data.get('intersection_density_val', 0)}\n\n"
            )

        prompt += (
            "Directrizes da Tarefa:\n"
            "Escreva um parecer técnico formal em português. Analise a associação mineralógica. "
            "Integre os resultados do Crosta PCA (anomalias de alteração) com a análise de "
            "intersecções estruturais para identificar os alvos mais promissores dentro da concessão. "
            "Conclua com recomendações de campo (amostragem de solo, trincheiras, sondagens) "
            "e um parecer final de 'Perfurar / Não Perfurar' (Drill/No-Drill)."
        )

        model = ModelInference(
            model_id="meta-llama/llama-3-3-70b-instruct",
            credentials=credentials,
            project_id=PROJECT_ID,
            params={"max_new_tokens": 1500, "temperature": 0.7}
        )
        st.markdown(model.generate_text(prompt=prompt))
