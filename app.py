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
    generate_exploration_targets,
)
from export_utils import (
    polygon_to_kml,
    create_kmz_bundle,
    create_geotiff_bundle,
    create_png_bundle,
    create_targets_kmz,
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
        self.cell(0, 10, f'Pagina {self.page_no()}/{{nb}} - Gerado Automaticamente via SatIntel AI', 0, 0, 'C')


# ========================================================
# PLATFORM CONFIGURATION
# ========================================================
try:
    IBM_API_KEY = st.secrets["WATSONX_APIKEY"]
    PROJECT_ID  = st.secrets["WATSONX_PROJECT_ID"]
except KeyError:
    st.error("Streamlit Secrets missing!")
    st.stop()

credentials = {"url": "https://us-south.ml.cloud.ibm.com", "apikey": IBM_API_KEY}


@st.cache_resource
def get_watsonx_client():
    client = APIClient(credentials=credentials)
    client.set.default_project(PROJECT_ID)
    return client


# ========================================================
# HELPER: draw polygon on matplotlib axes
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
# HELPER: draw target polygons on matplotlib
# ========================================================
def draw_targets_on_ax(ax, targets, fetch_bbox, img_shape):
    if not targets or fetch_bbox is None:
        return
    try:
        lon_min, lat_min, lon_max, lat_max = fetch_bbox
        h, w = img_shape[:2]

        def geo_to_px(lon, lat):
            x = (lon - lon_min) / (lon_max - lon_min) * w
            y = (lat_max - lat) / (lat_max - lat_min) * h
            return x, y

        colors = {"HIGH": "#FF0000", "MEDIUM": "#FFAA00", "LOW": "#00FFAA"}
        for t in targets:
            ring = t["polygon"]
            px = [geo_to_px(p[0], p[1]) for p in ring]
            xs = [c[0] for c in px]
            ys = [c[1] for c in px]
            color = colors.get(t["priority"], "#FFFFFF")
            patch = MplPolygon(list(zip(xs, ys)), closed=True,
                              facecolor=color, alpha=0.2,
                              edgecolor=color, linewidth=2, zorder=7)
            ax.add_patch(patch)
            ax.text(xs[0], ys[0], t["id"], fontsize=7, fontweight="bold",
                   color=color, zorder=8, ha="center", va="center")
    except Exception:
        pass


# ========================================================
# SESSION STATE
# ========================================================
if "map_center" not in st.session_state:
    st.session_state["map_center"] = [-15.095314, 32.567917]
if "active_polygon" not in st.session_state:
    st.session_state["active_polygon"] = None
if "concession_metadata" not in st.session_state:
    st.session_state["concession_metadata"] = {
        "Codigo da Licenca (Code)": "Aguardando Consulta",
        "Nome da Concessao": "Aguardando Consulta",
        "Titular (Holder Company)": "Aguardando Banco de Dados Real",
        "Area / Dimensao": "0.00 Ha",
        "Data de Emissao": "N/A",
        "Data de Validade (Expiry)": "N/A",
        "Tipo de Direito / Estado": "N/A",
        "Substancias": "N/A"
    }
if "satellite_data" not in st.session_state:
    st.session_state["satellite_data"] = None
# Initialize with predictive values so the app doesn't auto-fetch on page load
if "m_data" not in st.session_state:
    st.session_state["m_data"] = fetch_and_calculate_spatz([-15.095314, 32.567917], None, 2024)
if "exploration_targets" not in st.session_state:
    st.session_state["exploration_targets"] = None
if "fetch_requested" not in st.session_state:
    st.session_state["fetch_requested"] = False

# ========================================================
# LAYOUT
# ========================================================
st.title("SatIntel: Mozambique Mining Cadastre Real-Time Platform")
st.caption("Live Production Database Synchronization with Landfolio MIREME REST API Servers")

st.sidebar.header("Portal de Selecao de Alvos")
selected_basemap = st.sidebar.selectbox(
    "Select Map Layer View",
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
                st.session_state["exploration_targets"]  = None
                st.session_state["fetch_requested"]      = True
                st.sidebar.success(f"Concessao {license_num} carregada!")
            else:
                st.sidebar.error(f"Licenca '{license_num}' nao encontrada.")
else:
    st.sidebar.info("Clique no mapa para selecionar coordenadas.")

st.sidebar.divider()
target_commodity = st.sidebar.selectbox("Commodity Focus", ["Gold", "Copper", "Lithium", "Heavy Mineral Sands", "Emeralds"])

# ========================================================
# MAP + METADATA
# ========================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Live Geographic Registry View")

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

    if st.session_state["exploration_targets"]:
        priority_colors = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}
        for t in st.session_state["exploration_targets"]:
            color = priority_colors.get(t["priority"], "gray")
            folium.CircleMarker(
                location=[t["lat"], t["lon"]],
                radius=max(5, t["radius_m"] / 50),
                popup=folium.Popup(
                    f"<b>{t['id']}</b> (Score: {t['score']})<br/>"
                    f"Priority: {t['priority']}<br/>"
                    f"Structural: {t['structural_control']}<br/>"
                    f"Lithology: {t['lithology']}<br/>"
                    f"Radius: ~{t['radius_m']}m",
                    max_width=300
                ),
                color=color, fill=True, fillOpacity=0.4, weight=2
            ).add_to(m)

    folium.Marker(location=st.session_state["map_center"],
        tooltip=st.session_state["concession_metadata"].get("Nome da Concessao", "Center"),
        icon=folium.Icon(color="red", icon="info-sign")).add_to(m)

    map_data = st_folium(m, width=550, height=380, key=f"map_{selected_basemap}_{st.session_state['map_center']}")

    if search_method == "(c) Map Selection" and map_data and map_data.get("last_clicked"):
        cp = map_data["last_clicked"]
        lat, lng = cp["lat"], cp["lng"]
        st.session_state["map_center"] = [lat, lng]
        st.session_state["active_polygon"] = None
        st.session_state["concession_metadata"] = {
            "Codigo da Licenca (Code)": "Manual", "Nome da Concessao": f"({lat:.4f}, {lng:.4f})",
            "Titular (Holder Company)": "Campo Livre", "Area / Dimensao": "N/A",
            "Data de Emissao": "N/A", "Data de Validade (Expiry)": "N/A",
            "Tipo de Direito / Estado": "Area Livre", "Substancias": "Selecao Manual"
        }
        st.session_state["satellite_data"] = None
        st.session_state["m_data"] = None
        st.session_state["exploration_targets"] = None
        st.session_state["fetch_requested"] = True
        st.rerun()

    st.write("### Registo Oficial (Trimble Landfolio / INAMI)")
    st.table(st.session_state["concession_metadata"])

# ========================================================
# 5-WAY METRICS
# ========================================================
with col2:
    st.subheader("5 Core Remote Sensing Target Frameworks")

    # Show fetch button when m_data is None (after license load or map click)
    if st.session_state["m_data"] is None:
        st.info("Click below to fetch real Landsat satellite imagery and compute spectral indices.")
        if st.button("Fetch Satellite Imagery", use_container_width=True, type="primary"):
            st.session_state["fetch_requested"] = True
            st.rerun()

        # Show predictive values as placeholder
        st.caption("Showing predictive values until satellite fetch is triggered.")
        placeholder_data = fetch_and_calculate_spatz(st.session_state["map_center"], None, selected_year)
        st.session_state["m_data"] = placeholder_data

    # If fetch was requested, do the actual fetch with LIVE progress
    if st.session_state.get("fetch_requested") and st.session_state.get("satellite_data") is None:
        st.session_state["fetch_requested"] = False
        with st.status("Fetching satellite data & computing spectral indices...", expanded=True) as status:
            try:
                lat, lon = st.session_state["map_center"]
                active_poly = st.session_state.get("active_polygon")
                poly_bbox = polygon_to_bbox(active_poly) if active_poly else None

                def progress_cb(msg):
                    st.write(f"  {msg}")

                sat_data = fetch_satellite_imagery(
                    lat, lon, selected_year, bbox=poly_bbox, progress_cb=progress_cb
                )

                st.write("  Generating exploration target zones...")
                st.session_state["satellite_data"] = sat_data
                st.session_state["m_data"] = {
                    "Way_1_Iron_Oxide_Gossan":   sat_data["Way_1_Iron_Oxide_Gossan"],
                    "Way_1_Clay_Phyllic":        sat_data["Way_1_Clay_Phyllic"],
                    "Way_2_Fault_Density_Index":  sat_data["Way_2_Fault_Density_Index"],
                    "Way_3_Silica_Flooding_Cap":  sat_data["Way_3_Silica_Flooding_Cap"],
                    "Way_4_Geobotanical_Stress":  sat_data["Way_4_Geobotanical_Stress"],
                    "Way_5_WLC_Score_Percent":    sat_data["Way_5_WLC_Score_Percent"],
                    "Satellite_Used":             sat_data["Satellite_Used"],
                }
                st.session_state["exploration_targets"] = generate_exploration_targets(sat_data)
                st.write("  Done! Loading results...")
                status.update(label="Satellite analysis complete!", state="complete", expanded=False)
            except Exception as e:
                st.write(f"  Error: {str(e)[:200]}")
                status.update(label="Satellite fetch failed - using predictive values", state="error")
                st.session_state["m_data"] = fetch_and_calculate_spatz(st.session_state["map_center"], None, selected_year)
                st.session_state["satellite_data"] = None
                st.session_state["exploration_targets"] = None
        st.rerun()

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
    st.caption(f"{m_data['Satellite_Used']}")
    st.divider()

# ========================================================
# SATELLITE IMAGERY + SPECTRAL INDEX MAPS
# ========================================================
sat_data = st.session_state.get("satellite_data")
targets  = st.session_state.get("exploration_targets")

if sat_data is not None:
    active_poly = st.session_state.get("active_polygon")
    fetch_bbox  = sat_data.get("fetch_bbox")

    st.markdown("---")
    st.markdown("## Satellite Imagery & Spectral Index Maps")
    st.caption(f"Scene: {sat_data['scene_date']} | Cloud: {sat_data['cloud_cover']}% | {sat_data['Satellite_Used']}")
    if active_poly:
        st.success("Concession polygon overlay active on all images below.")

    def make_fig(img_array, cmap=None, vmin=None, vmax=None, title="", label="", show_targets=False):
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
        if show_targets and targets:
            draw_targets_on_ax(ax, targets, fetch_bbox, img_array.shape)
        else:
            ax.axis("off")
        return fig

    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown("### True Color (RGB)")
        st.pyplot(make_fig(sat_data["rgb"], title="Natural Color - Landsat"), use_container_width=True); plt.close()
    with ic2:
        st.markdown("### False Color (SWIR-NIR-Red)")
        st.pyplot(make_fig(sat_data["false_color"], title="Mineral Enhancement Composite", show_targets=True), use_container_width=True); plt.close()

    st.markdown("---")
    st.markdown("### Spectral Index Maps")
    ix1, ix2 = st.columns(2)
    with ix1:
        st.markdown("#### Iron Oxide (Band Ratio)")
        st.pyplot(make_fig(sat_data["iron_oxide_map"], cmap="RdYlBu_r", title="Iron Oxide Ratio (B4/B2)", label="Fe-Oxide", show_targets=True), use_container_width=True); plt.close()
    with ix2:
        st.markdown("#### Clay/Hydroxyl (Band Ratio)")
        st.pyplot(make_fig(sat_data["clay_map"], cmap="YlOrBr", title="Clay Ratio (B6/B7)", label="Clay", show_targets=True), use_container_width=True); plt.close()

    ix3, ix4 = st.columns(2)
    with ix3:
        st.markdown("#### NDVI - Vegetation Stress")
        st.pyplot(make_fig(sat_data["ndvi_map"], cmap="RdYlGn", vmin=-0.3, vmax=0.8, title="NDVI", label="NDVI"), use_container_width=True); plt.close()
    with ix4:
        st.markdown("#### Silica Proxy")
        st.pyplot(make_fig(sat_data["silica_map"], cmap="bone", title="Silica Proxy (B7/B6)", label="Silica"), use_container_width=True); plt.close()

    # ========================================================
    # CROSTA PCA
    # ========================================================
    st.markdown("---")
    st.markdown("## Crosta PCA - Hydrothermal Alteration Analysis")
    st.caption("Feature-Oriented Principal Component Analysis - targeted PCA on Landsat band subsets.")

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
        st.markdown("#### Crosta Iron Oxide PCA")
        st.pyplot(make_fig(sat_data["crosta_iron_pca"], cmap="RdYlBu_r", title=f"Crosta Iron Oxide (PC{sat_data.get('crosta_iron_pc', 0)+1})", label="PC Score", show_targets=True), use_container_width=True); plt.close()
    with pc2:
        st.markdown("#### Crosta Clay/Hydroxyl PCA")
        st.pyplot(make_fig(sat_data["crosta_clay_pca"], cmap="YlOrBr", title=f"Crosta Clay (PC{sat_data.get('crosta_clay_pc', 0)+1})", label="PC Score", show_targets=True), use_container_width=True); plt.close()

    # ========================================================
    # LINEAMENT ANALYSIS
    # ========================================================
    st.markdown("---")
    st.markdown("## Structural Lineament & Intersection Analysis")
    st.caption("Directional Sobel filtering - faults, fractures, shear zones.")

    lm1, lm2 = st.columns(2)
    with lm1:
        st.markdown("#### Lineament Density Map")
        st.pyplot(make_fig(sat_data["lineament_density_map"], cmap="hot", title="Structural Lineament Density", label="Density (0-4)", show_targets=True), use_container_width=True); plt.close()
    with lm2:
        st.markdown("#### Lineament Intersection Map")
        st.pyplot(make_fig(sat_data["intersection_map"], cmap="magma", title="Lineament Intersection Density", label="Intersection Index", show_targets=True), use_container_width=True); plt.close()

    st.markdown("---")
    st.markdown("### Per-Orientation Lineament Maps")
    ori1, ori2 = st.columns(2)
    with ori1:
        st.markdown("#### N-S Lineaments")
        st.pyplot(make_fig(sat_data["lineament_ns_map"], cmap="gray", title="North-South Lineaments", label="Binary"), use_container_width=True); plt.close()
    with ori2:
        st.markdown("#### E-W Lineaments")
        st.pyplot(make_fig(sat_data["lineament_ew_map"], cmap="gray", title="East-West Lineaments", label="Binary"), use_container_width=True); plt.close()

    ori3, ori4 = st.columns(2)
    with ori3:
        st.markdown("#### NE-SW Lineaments")
        st.pyplot(make_fig(sat_data["lineament_nesw_map"], cmap="gray", title="NE-SW Lineaments", label="Binary"), use_container_width=True); plt.close()
    with ori4:
        st.markdown("#### NW-SE Lineaments")
        st.pyplot(make_fig(sat_data["lineament_nwse_map"], cmap="gray", title="NW-SE Lineaments", label="Binary"), use_container_width=True); plt.close()

    st.markdown("---")
    lm_c1, lm_c2, lm_c3 = st.columns(3)
    lm_c1.metric("Lineament Density Index", sat_data.get("lineament_density_val", 0))
    lm_c2.metric("High-Confidence Intersections", sat_data.get("intersection_count", 0))
    lm_c3.metric("Intersection Density Index", sat_data.get("intersection_density_val", 0))

    # ========================================================
    # EXPLORATION TARGETS TABLE
    # ========================================================
    if targets:
        st.markdown("---")
        st.markdown("## Exploration Target Zones")
        st.caption("Composite score: IO(0.20) + CLAY(0.20) + Structural(0.15) + Geomorphology(0.30) + Lineament(0.15)")

        high_count = sum(1 for t in targets if t["priority"] == "HIGH")
        med_count = sum(1 for t in targets if t["priority"] == "MEDIUM")
        low_count = sum(1 for t in targets if t["priority"] == "LOW")

        tc1, tc2, tc3, tc4 = st.columns(4)
        tc1.metric("Total Targets", len(targets))
        tc2.metric("High Priority", high_count)
        tc3.metric("Medium Priority", med_count)
        tc4.metric("Low Priority", low_count)

        target_rows = []
        for t in targets:
            target_rows.append({
                "ID": t["id"],
                "Score": t["score"],
                "Priority": t["priority"],
                "Structural Control": t["structural_control"],
                "Lithology": t["lithology"],
                "Radius (m)": t["radius_m"],
                "Lat": f"{t['lat']:.4f}",
                "Lon": f"{t['lon']:.4f}",
            })
        st.dataframe(target_rows, use_container_width=True, hide_index=True)

        st.markdown("### Target Descriptions")
        for t in targets:
            priority_emoji = {"HIGH": "Red", "MEDIUM": "Orange", "LOW": "Green"}
            emoji = priority_emoji.get(t["priority"], "White")
            with st.expander(f"[{emoji}] {t['id']} - Score: {t['score']} ({t['priority']})"):
                st.markdown(f"**Structural Control:** {t['structural_control']}")
                st.markdown(f"**Lithology:** {t['lithology']}")
                st.markdown(f"**Radius:** ~{t['radius_m']} m")
                st.markdown(f"**Coordinates:** {t['lat']:.6f}, {t['lon']:.6f}")
                st.markdown(f"**Component Scores:** IO={t['io_score']} | Clay={t['clay_score']} | Structural={t['struct_score']} | Geomorphology={t['geomorph_score']} | Lineament={t['line_score']}")
                st.markdown(f"**EN:** {t['description_en']}")
                st.markdown(f"**PT:** {t['description_pt']}")

    # ========================================================
    # EXPORT SECTION
    # ========================================================
    st.markdown("---")
    st.markdown("## Export & Google Earth Integration")

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.markdown("### Concession Geometry")
        if active_poly:
            kml_str = polygon_to_kml(active_poly, st.session_state.get("concession_metadata"))
            if kml_str:
                license_code = st.session_state['concession_metadata'].get('Codigo da Licenca (Code)', st.session_state['concession_metadata'].get('Código da Licença (Code)', 'unknown'))
                st.download_button(
                    label="Export Concession Boundary (KML)",
                    data=kml_str.encode("utf-8"),
                    file_name=f"concession_{license_code}.kml",
                    mime="application/vnd.google-earth.kml+xml",
                    use_container_width=True,
                )
            import json
            geojson_bytes = json.dumps(active_poly, indent=2).encode("utf-8")
            st.download_button(
                label="Export Concession Boundary (GeoJSON)",
                data=geojson_bytes,
                file_name=f"concession_{license_code}.geojson",
                mime="application/geo+json",
                use_container_width=True,
            )

    with exp_col2:
        st.markdown("### Image Exports")
        kmz_bytes = create_kmz_bundle(sat_data, polygon_geojson=active_poly,
                                      metadata=st.session_state.get("concession_metadata"), fetch_bbox=fetch_bbox)
        if kmz_bytes:
            st.download_button(
                label="Export All Overlays (KMZ)",
                data=kmz_bytes,
                file_name=f"satintel_overlays_{sat_data.get('scene_date', '')}.kmz",
                mime="application/vnd.google-earth.kmz",
                use_container_width=True,
            )
            st.caption("Polygon + 10 georeferenced image overlays")

    if targets:
        st.markdown("---")
        st.markdown("### Exploration Targets Export")
        targets_kmz = create_targets_kmz(
            targets, polygon_geojson=active_poly,
            metadata=st.session_state.get("concession_metadata"), sat_data=sat_data
        )
        if targets_kmz:
            license_code = st.session_state['concession_metadata'].get('Codigo da Licenca (Code)', st.session_state['concession_metadata'].get('Código da Licença (Code)', 'unknown'))
            st.download_button(
                label="Export Exploration Targets (KMZ)",
                data=targets_kmz,
                file_name=f"License{license_code}-GoldExplorationTargets.kmz",
                mime="application/vnd.google-earth.kmz",
                use_container_width=True,
            )
            st.caption("Priority-coded target polygons with scores - ready for Google Earth")

    st.markdown("---")
    exp2_c1, exp2_c2 = st.columns(2)
    with exp2_c1:
        geotiff_bytes = create_geotiff_bundle(sat_data, fetch_bbox=fetch_bbox)
        if geotiff_bytes:
            st.download_button(
                label="Export All Rasters (GeoTIFF)",
                data=geotiff_bytes,
                file_name=f"satintel_geotiffs_{sat_data.get('scene_date', '')}.zip",
                mime="application/zip",
                use_container_width=True,
            )
    with exp2_c2:
        png_bytes = create_png_bundle(sat_data)
        if png_bytes:
            st.download_button(
                label="Export All Images (PNG)",
                data=png_bytes,
                file_name=f"satintel_images_{sat_data.get('scene_date', '')}.zip",
                mime="application/zip",
                use_container_width=True,
            )

# ========================================================
# IBM WATSONX GEOLOGICAL REPORT
# ========================================================
st.markdown("---")
if st.button("Generate Comprehensive Geological Synthesis Report", use_container_width=True, type="primary"):
    with st.spinner("O watsonx.ai esta a processar a analise geologica completa..."):
        client = get_watsonx_client()
        meta   = st.session_state["concession_metadata"]

        target_summary = ""
        if targets:
            target_lines = []
            for t in targets:
                target_lines.append(
                    f"  {t['id']}: Score={t['score']}, Priority={t['priority']}, "
                    f"Structural={t['structural_control']}, Lithology={t['lithology']}, "
                    f"Radius=~{t['radius_m']}m, Lat={t['lat']:.4f}, Lon={t['lon']:.4f}\n"
                    f"    EN: {t['description_en']}\n"
                    f"    PT: {t['description_pt']}"
                )
            target_summary = "\n".join(target_lines)

        prompt = f"""[Role: Geologo Senior de Exploracao, Especialista em Metalogenia do Cinturao Moambicano (Pan-African Belt, 650-500 Ma)]

Voce esta a preparar um PARECER TECNICO FORMAL para uma concessao mineira em Moambique. O relatorio deve ser estruturado, detalhado e profissional.

=== DADOS DO CADASTRO MINEIRO ===
- Codigo da Licenca: {meta.get('Codigo da Licenca (Code)', meta.get('Código da Licença (Code)', 'N/A'))}
- Nome da Concessao: {meta.get('Nome da Concessao', meta.get('Nome da Concessão', ''))}
- Titular: {meta.get('Titular (Holder Company)', '')}
- Dimensao: {meta.get('Area / Dimensao', meta.get('Área / Dimensão', ''))}
- Validade: {meta.get('Data de Validade (Expiry)', '')}
- Substancias: {meta.get('Substancias', meta.get('Substâncias', ''))}
- Coordenadas: {st.session_state['map_center']}
- Ano de Analise: {selected_year}
- Commodity Focus: {target_commodity}

=== MATRIZ DE TELEMETRIA (5-WAY MODEL) ===
- Way 1 - Oxido de Ferro (Gossans): {m_data.get('Way_1_Iron_Oxide_Gossan', 2.4)}
- Way 1 - Indice de Argila/Hidroxilo: {m_data.get('Way_1_Clay_Phyllic', 1.9)}
- Way 2 - Densidade de Falhas: {m_data.get('Way_2_Fault_Density_Index', 0.8)}
- Way 3 - Silicificacao: {m_data.get('Way_3_Silica_Flooding_Cap', 0.6)}
- Way 4 - Estresse Geobotanico (NDVI): {m_data.get('Way_4_Geobotanical_Stress', 0.34)}
- Way 5 - WLC Prospectivity Score: {m_data.get('Way_5_WLC_Score_Percent', 88.5)}%
- Satelite: {m_data.get('Satellite_Used', 'Landsat')}"""

        if sat_data:
            prompt += f"""

=== ANALISE CROSTA PCA ===
- Iron Oxide PCA (PC{sat_data.get('crosta_iron_pc',0)+1}): mean={sat_data.get('crosta_iron_mean',0)}, anomaly coverage={sat_data.get('crosta_iron_anomaly_pct',0)}%
- Clay PCA (PC{sat_data.get('crosta_clay_pc',0)+1}): mean={sat_data.get('crosta_clay_mean',0)}, anomaly coverage={sat_data.get('crosta_clay_anomaly_pct',0)}%

=== ANALISE ESTRUTURAL ===
- Densidade de Lineamentos: {sat_data.get('lineament_density_val', 0)}
- Interseccoes de Alta Confianca: {sat_data.get('intersection_count', 0)}
- Indice de Densidade de Interseccao: {sat_data.get('intersection_density_val', 0)}"""

        if targets:
            high_count = sum(1 for t in targets if t["priority"] == "HIGH")
            med_count = sum(1 for t in targets if t["priority"] == "MEDIUM")
            low_count = sum(1 for t in targets if t["priority"] == "LOW")
            prompt += f"""

=== ALVOS DE EXPLORACAO ===
Total: {len(targets)} alvos ({high_count} Alta, {med_count} Media, {low_count} Baixa prioridade)

{target_summary}"""

        prompt += """

=== ESTRUTURA OBRIGATORIA DO RELATORIO ===

Escreva o parecer tecnico em PORTUGUES, seguindo EXATAMENTE esta estrutura:

**1. RESUMO EXECUTIVO**
- Sintese da concessao, commodity-alvo, e conclusao principal (2-3 paragrafos)

**2. CONTEXTO GEOLOGICO REGIONAL**
- Cinturao Moambicano (Pan-African Belt), idade, contexto tectonico
- Litologias predominantes e estilo de mineralizacao esperado
- Controles estruturais regionais

**3. ANALISE DE ALTERACAO HIDROTERMAL**
- Interpretar os resultados do Crosta PCA (iron oxide + clay)
- Correlacionar com os indices de band ratio (Way 1)

**4. ANALISE ESTRUTURAL**
- Interpretar a densidade de lineamentos e interseccoes
- Identificar orientacoes dominantes
- Discutir o controle estrutural sobre a mineralizacao aurifera

**5. ALVOS DE EXPLORACAO**
- Tabela resumo dos alvos gerados
- Discutir os alvos de ALTA prioridade em detalhe

**6. RECOMENDACOES DE CAMPO**
- Programa de amostragem de solo/rio
- Trincheiras / mapeamento geologico
- Sondagens (DD/RC)
- Cronograma sugerido

**7. PARECER FINAL: PERFURAR / NAO PERFURAR**
- Justificacao baseada em todos os dados
- Nivel de confianca
- Recomendacao final

Use terminologia geologica tecnica. Seja especifico e quantitativo."""

        model = ModelInference(
            model_id="meta-llama/llama-3-3-70b-instruct",
            credentials=credentials,
            project_id=PROJECT_ID,
            params={"max_new_tokens": 3000, "temperature": 0.5}
        )
        report_text = model.generate_text(prompt=prompt)
        st.markdown(report_text)

        st.markdown("---")
        st.markdown("### Export Report")
        try:
            pdf = TechnicalReportPDF()
            pdf.alias_nb_pages()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=25)

            pdf.set_font('Helvetica', 'B', 16)
            pdf.set_text_color(0, 77, 64)
            conces_name = meta.get('Nome da Concessao', meta.get('Nome da Concessão', 'Concessao'))
            pdf.multi_cell(0, 10, f'PARECER TECNICO - {conces_name}')
            pdf.ln(5)

            pdf.set_font('Helvetica', 'B', 10)
            pdf.set_text_color(0, 0, 0)
            lic_code = meta.get('Codigo da Licenca (Code)', meta.get('Código da Licença (Code)', 'N/A'))
            pdf.cell(0, 6, f'Licenca: {lic_code}', 0, 1)
            pdf.cell(0, 6, f'Titular: {meta.get("Titular (Holder Company)", "N/A")}', 0, 1)
            pdf.cell(0, 6, f'Data: {selected_year} | Satelite: {m_data.get("Satellite_Used", "N/A")}', 0, 1)
            pdf.ln(5)

            pdf.set_font('Helvetica', '', 10)
            clean_text = report_text.replace('**', '').replace('*', '').replace('#', '')
            pdf.multi_cell(0, 5, clean_text)

            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button(
                label="Download Report as PDF",
                data=pdf_bytes,
                file_name=f"parecer_tecnico_{lic_code}_{selected_year}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception:
            st.info("Relatorio gerado. Copie o texto acima para o seu documento.")
