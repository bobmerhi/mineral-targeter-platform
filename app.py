import warnings
warnings.filterwarnings("ignore", message="Unverified HTTPS request")
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import streamlit as st
from datetime import datetime
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
import json


# ========================================================
# PDF CLASS — Unicode-safe with professional cover page
# ========================================================
def _clean_pdf_text(text):
    """Replace Unicode chars that Helvetica can't render with ASCII equivalents."""
    if not text:
        return ""
    replacements = {
        "\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'", "\u201c": '"',
        "\u201d": '"', "\u2026": "...", "\u00e7": "c", "\u00e9": "e", "\u00ea": "e",
        "\u00e1": "a", "\u00ed": "i", "\u00f3": "o", "\u00f5": "o", "\u00fa": "u",
        "\u00e0": "a", "\u00e8": "e", "\u00ec": "i", "\u00f2": "o", "\u00f9": "u",
        "\u00c7": "C", "\u00c9": "E", "\u00ca": "E", "\u00c1": "A", "\u00cd": "I",
        "\u00d3": "O", "\u00d5": "O", "\u00da": "U", "\u00c0": "A", "\u00c8": "E",
        "\u00cc": "I", "\u00d2": "O", "\u00d9": "U", "\u00e3": "a", "\u00f1": "n",
        "\u00c3": "A", "\u00d1": "N", "\u00ba": "o", "\u00aa": "a", "\u00b2": "2",
        "\u00b3": "3", "\u00b0": " deg", "\u00b5": "u", "\u00d7": "x", "\u00f7": "/",
        "\u2192": "->", "\u2190": "<-", "\u2191": "^", "\u2193": "v",
        "\u2265": ">=", "\u2264": "<=", "\u2260": "!=", "\u221e": "inf",
        "\u00b1": "+/-", "\u00b7": ".", "\u25cf": "*", "\u2022": "-",
        "\u2013": "-", "\u2014": "-", "\u00a0": " ",
    }
    result = text
    for unicode_char, ascii_repl in replacements.items():
        result = result.replace(chr(int(unicode_char.replace("\\u", ""), 16)), ascii_repl)
    # Final fallback: encode to latin-1 with replacement for anything still unsupported
    return result.encode("ascii", "replace").decode("ascii")


class TechnicalReportPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return  # Cover page has its own header
        self.set_font('Helvetica', 'B', 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, _clean_pdf_text(self.report_title or 'SatIntel Report'), 0, 1, 'L')
        self.set_draw_color(180, 180, 180)
        self.line(10, 18, 200, 18)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        page = f'Pagina {self.page_no()}/{{nb}}'
        if hasattr(self, 'author_name') and self.author_name:
            page += f'  |  Autor: {_clean_pdf_text(self.author_name)}'
        page += '  |  SatIntel AI'
        self.cell(0, 10, page, 0, 0, 'C')


# ========================================================
# WATSONX
# ========================================================
try:
    IBM_API_KEY = st.secrets["WATSONX_APIKEY"]
    PROJECT_ID  = st.secrets["WATSONX_PROJECT_ID"]
except KeyError:
    st.error("Streamlit Secrets missing: WATSONX_APIKEY / WATSONX_PROJECT_ID")
    st.stop()

credentials = {"url": "https://us-south.ml.cloud.ibm.com", "apikey": IBM_API_KEY}

@st.cache_resource
def get_watsonx_client():
    client = APIClient(credentials=credentials)
    client.set.default_project(PROJECT_ID)
    return client


# ========================================================
# HELPERS: draw overlays on matplotlib axes
# ========================================================
def draw_polygon_on_ax(ax, polygon_geojson, fetch_bbox, img_shape):
    if not polygon_geojson or fetch_bbox is None:
        return
    try:
        lon_min, lat_min, lon_max, lat_max = fetch_bbox
        h, w = img_shape[:2]
        def geo_to_px(lon, lat):
            return (lon - lon_min) / (lon_max - lon_min) * w, (lat_max - lat) / (lat_max - lat_min) * h
        rings = polygon_geojson["geometry"]["coordinates"]
        for ring in rings:
            px = [geo_to_px(p[0], p[1]) for p in ring]
            xs, ys = [c[0] for c in px], [c[1] for c in px]
            ax.add_patch(MplPolygon(list(zip(xs, ys)), closed=True,
                facecolor="cyan", alpha=0.15, edgecolor="yellow", linewidth=2.5, zorder=5))
            ax.plot(xs + [xs[0]], ys + [ys[0]], color="#FFD700", linewidth=2.5, zorder=6)
        ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")
    except Exception:
        ax.axis("off")

def draw_targets_on_ax(ax, targets, fetch_bbox, img_shape):
    if not targets or fetch_bbox is None:
        return
    try:
        lon_min, lat_min, lon_max, lat_max = fetch_bbox
        h, w = img_shape[:2]
        def geo_to_px(lon, lat):
            return (lon - lon_min) / (lon_max - lon_min) * w, (lat_max - lat) / (lat_max - lat_min) * h
        colors = {"HIGH": "#FF0000", "MEDIUM": "#FFAA00", "LOW": "#00FFAA"}
        for t in targets:
            ring = t["polygon"]
            px = [geo_to_px(p[0], p[1]) for p in ring]
            xs, ys = [c[0] for c in px], [c[1] for c in px]
            color = colors.get(t["priority"], "#FFFFFF")
            ax.add_patch(MplPolygon(list(zip(xs, ys)), closed=True,
                facecolor=color, alpha=0.2, edgecolor=color, linewidth=2, zorder=7))
            cx = sum(xs) / len(xs); cy = sum(ys) / len(ys)
            ax.text(cx, cy, t["id"], fontsize=7, fontweight="bold",
                color=color, zorder=8, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5, lw=0))
    except Exception:
        pass


# ========================================================
# SESSION STATE — initialise once
# ========================================================
DEFAULTS = {
    "map_center": [-15.095314, 32.567917],
    "active_polygon": None,
    "concession_metadata": {
        "Codigo da Licenca (Code)": "Aguardando Consulta",
        "Nome da Concessao": "Aguardando Consulta",
        "Titular (Holder Company)": "—",
        "Area / Dimensao": "0.00 Ha",
        "Data de Emissao": "N/A",
        "Data de Validade (Expiry)": "N/A",
    },
    "satellite_data": None,
    "m_data": None,          # None = no data yet; dict = predictive or real
    "exploration_targets": None,
    "fetch_running": False,  # True only while the fetch st.status block is executing
    "last_license": "",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Compute predictive baseline only if m_data is None AND no fetch is running
if st.session_state["m_data"] is None and not st.session_state["fetch_running"]:
    st.session_state["m_data"] = fetch_and_calculate_spatz(
        st.session_state["map_center"], None, 2024
    )
    st.session_state["m_data"]["_is_predictive"] = True


# ========================================================
# TITLE
# ========================================================
st.title("SatIntel: Mozambique Mining Cadastre Real-Time Platform")
st.caption("Live Production Database Synchronization with Landfolio MIREME REST API Servers")


# ========================================================
# SIDEBAR
# ========================================================
st.sidebar.header("Portal de Selecao de Alvos")

selected_basemap = st.sidebar.selectbox(
    "Select Map Layer View",
    ["Esri World Imagery (Satellite)", "Google Satellite Imagery",
     "OpenStreetMap (Standard)", "Esri Topographic Map"]
)
selected_year = st.sidebar.slider("Select Analysis Year", 1990, 2026, 2024)
search_method = st.sidebar.radio(
    "Select Landfolio Lookup Method",
    ["(a) License # Search", "(c) Map Selection"]
)

if search_method == "(a) License # Search":
    license_num = st.sidebar.text_input(
        "Enter License Number",
        value=st.session_state.get("last_license", ""),
        placeholder="e.g., 11521"
    )
    search_clicked = st.sidebar.button("Search License", type="primary", use_container_width=True)

    if search_clicked and license_num.strip():
        with st.sidebar.status("Searching INAMI cadastre...", expanded=True) as cad_status:
            st.write("Connecting to Landfolio portal...")
            db_result = get_real_mozambique_cadastre(license_num.strip())
            if db_result["found"]:
                name = db_result.get("metadata", {}).get("Nome da Concessao", license_num)
                st.write(f"Found: {name}")
                st.session_state["map_center"]         = [db_result["lat"], db_result["lon"]]
                st.session_state["active_polygon"]      = db_result["polygon"]
                st.session_state["concession_metadata"] = db_result["metadata"]
                st.session_state["satellite_data"]      = None
                st.session_state["exploration_targets"] = None
                st.session_state["last_license"]        = license_num.strip()
                # Reset m_data to predictive for the new location
                st.session_state["m_data"] = fetch_and_calculate_spatz(
                    [db_result["lat"], db_result["lon"]], None, selected_year
                )
                st.session_state["m_data"]["_is_predictive"] = True
                cad_status.update(label=f"License {license_num} loaded!", state="complete", expanded=False)
            else:
                cad_status.update(label=f"License '{license_num}' not found", state="error")
else:
    st.sidebar.info("Clique no mapa para selecionar coordenadas.")

st.sidebar.divider()
target_commodity = st.sidebar.selectbox(
    "Commodity Focus",
    ["Gold", "Copper", "Lithium", "Heavy Mineral Sands", "Emeralds"]
)


# ========================================================
# MAP + METADATA
# ========================================================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Live Geographic Registry View")

    tile_kwargs = {}
    if selected_basemap == "Esri World Imagery (Satellite)":
        tile_kwargs = {"tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", "attr": "Esri"}
    elif selected_basemap == "Google Satellite Imagery":
        tile_kwargs = {"tiles": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", "attr": "Google"}
    elif selected_basemap == "Esri Topographic Map":
        tile_kwargs = {"tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}", "attr": "Esri Topo"}

    m = folium.Map(location=st.session_state["map_center"], zoom_start=10, **tile_kwargs)

    if st.session_state["active_polygon"]:
        folium.GeoJson(
            st.session_state["active_polygon"], name="Concession Boundary",
            style_function=lambda x: {"fillColor": "#00E5FF", "color": "#FFD700", "weight": 4, "fillOpacity": 0.3},
            tooltip=folium.GeoJsonTooltip(
                fields=["name"], aliases=["Concession:"],
                style="background-color:#004D40;color:white;font-weight:bold;padding:5px;border-radius:3px;"
            )
        ).add_to(m)

    targets = st.session_state.get("exploration_targets")
    if targets:
        priority_colors = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}
        for t in targets:
            folium.CircleMarker(
                location=[t["lat"], t["lon"]],
                radius=10,
                color=priority_colors.get(t["priority"], "gray"),
                fill=True, fill_opacity=0.7,
                tooltip=f"{t['id']} | Score: {t['score']} | {t['priority']}"
            ).add_to(m)

    folium.Marker(
        location=st.session_state["map_center"],
        tooltip=st.session_state["concession_metadata"].get("Nome da Concessao", "Center"),
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)

    map_data = st_folium(m, width=560, height=400, key=f"map_{selected_basemap}")

    # Map-click handling
    if map_data and map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lng = map_data["last_clicked"]["lng"]
        if [lat, lng] != st.session_state["map_center"]:
            st.session_state["map_center"]         = [lat, lng]
            st.session_state["active_polygon"]      = None
            st.session_state["concession_metadata"] = {k: "Map Selection" for k in st.session_state["concession_metadata"]}
            st.session_state["satellite_data"]      = None
            st.session_state["exploration_targets"] = None
            st.session_state["m_data"] = fetch_and_calculate_spatz([lat, lng], None, selected_year)
            st.session_state["m_data"]["_is_predictive"] = True
            st.rerun()

    st.write("### Registo Oficial (Trimble Landfolio / INAMI)")
    st.table(st.session_state["concession_metadata"])


# ========================================================
# 5-WAY METRICS + SATELLITE FETCH
# ========================================================
with col2:
    st.subheader("5 Core Remote Sensing Target Frameworks")

    m_data = st.session_state["m_data"]
    sat_data = st.session_state.get("satellite_data")
    is_predictive = m_data.get("_is_predictive", False) if m_data else True

    # ── FETCH BUTTON (always visible when no real sat data yet) ──────────
    if sat_data is None:
        if is_predictive:
            st.info("⚡ Showing **predictive values**. Click below to fetch real Landsat imagery.")
        st.button(
            "🛰️ Fetch Real Satellite Imagery",
            key="fetch_btn",
            type="primary",
            use_container_width=True,
            on_click=lambda: st.session_state.update({"fetch_running": True})
        )

    # ── ACTUAL FETCH EXECUTION ────────────────────────────────────────────
    if st.session_state["fetch_running"] and sat_data is None:
        st.session_state["fetch_running"] = False   # reset flag immediately

        with st.status("🛰️ Fetching satellite data & computing spectral indices...", expanded=True) as status:
            log = st.empty()
            steps = []

            def progress_cb(msg):
                steps.append(msg)
                log.markdown("\n".join(f"✅ {s}" for s in steps))

            def preview_cb(title, img, cmap=None):
                steps.append(title)
                log.markdown("\n".join(f"✅ {s}" for s in steps))
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.imshow(img, cmap=cmap, aspect="auto") if cmap else ax.imshow(img, aspect="auto")
                ax.set_title(title, fontsize=9, fontweight="bold"); ax.axis("off")
                st.pyplot(fig, use_container_width=True); plt.close(fig)

            try:
                lat, lon = st.session_state["map_center"]
                active_poly = st.session_state.get("active_polygon")
                poly_bbox = polygon_to_bbox(active_poly) if active_poly else None

                result = fetch_satellite_imagery(
                    lat, lon, selected_year,
                    bbox=poly_bbox,
                    progress_cb=progress_cb,
                    preview_cb=preview_cb
                )

                steps.append("Generating exploration target zones...")
                log.markdown("\n".join(f"✅ {s}" for s in steps))

                st.session_state["satellite_data"]      = result
                st.session_state["exploration_targets"] = generate_exploration_targets(result, polygon_geojson=st.session_state.get("active_polygon"))
                st.session_state["m_data"] = {
                    "Way_1_Iron_Oxide_Gossan":  result["Way_1_Iron_Oxide_Gossan"],
                    "Way_1_Clay_Phyllic":       result["Way_1_Clay_Phyllic"],
                    "Way_2_Fault_Density_Index": result["Way_2_Fault_Density_Index"],
                    "Way_3_Silica_Flooding_Cap": result["Way_3_Silica_Flooding_Cap"],
                    "Way_4_Geobotanical_Stress": result["Way_4_Geobotanical_Stress"],
                    "Way_5_WLC_Score_Percent":   result["Way_5_WLC_Score_Percent"],
                    "Satellite_Used":            result["Satellite_Used"],
                    "_is_predictive":            False,
                }
                status.update(label="✅ Satellite analysis complete!", state="complete", expanded=False)
                st.rerun()

            except Exception as e:
                status.update(label=f"❌ Fetch failed — showing predictive values", state="error")
                st.error(f"Error: {str(e)[:300]}")
                st.session_state["m_data"] = fetch_and_calculate_spatz(
                    st.session_state["map_center"], None, selected_year
                )
                st.session_state["m_data"]["_is_predictive"] = True
                st.session_state["satellite_data"]      = None
                st.session_state["exploration_targets"] = None

    # ── 5-WAY METRICS DISPLAY ─────────────────────────────────────────────
    if m_data:
        source_tag = "Predictive Model" if is_predictive else "Real Landsat Satellite"
        st.caption(f"Source: **{source_tag}** | {m_data.get('Satellite_Used', '')}")

        st.markdown("#### **WAY 1: Hydrothermal Alteration**")
        w1a, w1b = st.columns(2)
        w1a.metric("Iron Oxide (Gossans)",   m_data["Way_1_Iron_Oxide_Gossan"])
        w1b.metric("Clay/Hydroxyl Index",    m_data["Way_1_Clay_Phyllic"])

        st.markdown("#### **WAY 2: Structural Lineaments**")
        st.metric("Fault Intersection Density", m_data["Way_2_Fault_Density_Index"])

        st.markdown("#### **WAY 3: Lithological Silicification**")
        st.metric("Quartz Veining Emissivity",  m_data["Way_3_Silica_Flooding_Cap"])

        st.markdown("#### **WAY 4: Geobotanical Stress**")
        st.metric("Vegetation Stress Proxy (NDVI)", m_data["Way_4_Geobotanical_Stress"])

        st.markdown("#### **WAY 5: GIS Predictive Synthesis**")
        st.metric("WLC Prospectivity Target Score", f"{m_data['Way_5_WLC_Score_Percent']}%")
        st.divider()


# ========================================================
# SATELLITE IMAGERY + SPECTRAL MAPS (only when real data)
# ========================================================
sat_data = st.session_state.get("satellite_data")
targets  = st.session_state.get("exploration_targets")

if sat_data is not None:
    active_poly = st.session_state.get("active_polygon")
    fetch_bbox  = sat_data.get("fetch_bbox")

    def _polygon_pixel_mask(fetch_bbox, img_shape):
        """Build a boolean mask: True inside the concession polygon."""
        if not active_poly or fetch_bbox is None:
            return None
        try:
            from matplotlib.path import Path
            lon_min, lat_min, lon_max, lat_max = fetch_bbox
            h, w = img_shape[:2]
            ys, xs = np.mgrid[:h, :w]
            grid = np.column_stack([xs.ravel(), ys.ravel()])
            mask = np.zeros(h * w, dtype=bool)
            for ring in active_poly["geometry"]["coordinates"]:
                verts = []
                for p in ring:
                    px = (p[0] - lon_min) / (lon_max - lon_min) * w
                    py = (lat_max - p[1]) / (lat_max - lat_min) * h
                    verts.append((px, py))
                path = Path(verts)
                mask |= path.contains_points(grid)
            return mask.reshape(h, w)
        except Exception:
            return None

    def make_fig(img_array, cmap=None, vmin=None, vmax=None, title="", label="", show_targets=False):
        fig, ax = plt.subplots(figsize=(7, 6))
        # ── Clip image to polygon boundaries ──────────────────────────
        if active_poly and fetch_bbox:
            mask = _polygon_pixel_mask(fetch_bbox, img_array.shape)
            if mask is not None:
                if img_array.ndim == 3:
                    masked = img_array.copy()
                    masked[~mask] = 0  # black outside polygon
                    display = masked
                else:
                    masked = img_array.copy()
                    masked[~mask] = np.nan
                    display = masked
            else:
                display = img_array
        else:
            display = img_array

        kw = {}
        if vmin is not None: kw["vmin"] = vmin
        if vmax is not None: kw["vmax"] = vmax
        if cmap:
            im = ax.imshow(display, cmap=cmap, **kw)
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(label, fontsize=9)
        else:
            ax.imshow(display, **kw)
        ax.set_title(title, fontsize=10, fontweight="bold")
        if active_poly and fetch_bbox:
            draw_polygon_on_ax(ax, active_poly, fetch_bbox, img_array.shape)
        if show_targets and targets and fetch_bbox:
            draw_targets_on_ax(ax, targets, fetch_bbox, img_array.shape)
        else:
            ax.axis("off")
        return fig

    st.markdown("---")
    st.markdown("## 🛰️ Satellite Imagery & Spectral Index Maps")
    st.caption(f"Scene: {sat_data['scene_date']} | Cloud: {sat_data['cloud_cover']}% | {sat_data['Satellite_Used']}")
    if active_poly:
        st.success("✅ Concession polygon + target overlays active on all images.")

    # Row 1 — True Color + False Color
    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown("### True Color (RGB)")
        st.pyplot(make_fig(sat_data["rgb"], title="Natural Color — Landsat"), use_container_width=True)
        plt.close()
    with ic2:
        st.markdown("### False Color (SWIR-NIR-Red)")
        st.pyplot(make_fig(sat_data["false_color"], title="Mineral Enhancement Composite", show_targets=True), use_container_width=True)
        plt.close()

    # Row 2 — Spectral indices
    st.markdown("---")
    st.markdown("### Spectral Index Maps")
    ix1, ix2 = st.columns(2)
    with ix1:
        st.markdown("#### Iron Oxide (Band Ratio)")
        st.pyplot(make_fig(sat_data["iron_oxide_map"], cmap="RdYlBu_r", title="Iron Oxide Ratio (B4/B2)", label="Fe-Oxide", show_targets=True), use_container_width=True)
        plt.close()
    with ix2:
        st.markdown("#### Clay/Hydroxyl (Band Ratio)")
        st.pyplot(make_fig(sat_data["clay_map"], cmap="YlOrBr", title="Clay Ratio (B6/B7)", label="Clay", show_targets=True), use_container_width=True)
        plt.close()

    ix3, ix4 = st.columns(2)
    with ix3:
        st.markdown("#### NDVI — Vegetation Stress")
        st.pyplot(make_fig(sat_data["ndvi_map"], cmap="RdYlGn", vmin=-0.3, vmax=0.8, title="NDVI", label="NDVI"), use_container_width=True)
        plt.close()
    with ix4:
        st.markdown("#### Silica Proxy")
        st.pyplot(make_fig(sat_data["silica_map"], cmap="bone", title="Silica Proxy (B7/B6)", label="Silica"), use_container_width=True)
        plt.close()

    # Row 3 — Crosta PCA
    st.markdown("---")
    st.markdown("## Crosta PCA — Hydrothermal Alteration Analysis")
    st.caption("Feature-Oriented PCA on Landsat band subsets — isolates hydrothermal signatures.")

    iron_load = sat_data.get("crosta_iron_loadings", {})
    clay_load = sat_data.get("crosta_clay_loadings", {})

    lc1, lc2 = st.columns(2)
    with lc1:
        st.markdown(f"#### Iron Oxide PC{sat_data.get('crosta_iron_pc', 0)+1} Eigenvector Loadings")
        st.dataframe({"Band": list(iron_load.keys()), "Loading": list(iron_load.values())},
                     use_container_width=True, hide_index=True)
        st.metric("Iron Oxide Anomaly Coverage", f"{sat_data.get('crosta_iron_anomaly_pct', 0)}%")
    with lc2:
        st.markdown(f"#### Clay/Hydroxyl PC{sat_data.get('crosta_clay_pc', 0)+1} Eigenvector Loadings")
        st.dataframe({"Band": list(clay_load.keys()), "Loading": list(clay_load.values())},
                     use_container_width=True, hide_index=True)
        st.metric("Clay Alteration Anomaly Coverage", f"{sat_data.get('crosta_clay_anomaly_pct', 0)}%")

    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown(f"#### Crosta Iron Oxide PCA")
        st.pyplot(make_fig(sat_data["crosta_iron_pca"], cmap="RdYlBu_r",
            title=f"Crosta Iron Oxide (PC{sat_data.get('crosta_iron_pc', 0)+1})", label="PC Score", show_targets=True),
            use_container_width=True); plt.close()
    with pc2:
        st.markdown(f"#### Crosta Clay/Hydroxyl PCA")
        st.pyplot(make_fig(sat_data["crosta_clay_pca"], cmap="YlOrBr",
            title=f"Crosta Clay (PC{sat_data.get('crosta_clay_pc', 0)+1})", label="PC Score", show_targets=True),
            use_container_width=True); plt.close()

    # Row 4 — Lineaments
    st.markdown("---")
    st.markdown("## Structural Lineament & Intersection Analysis")
    st.caption("Directional Sobel filtering — faults, fractures, shear zones.")

    lm1, lm2 = st.columns(2)
    with lm1:
        st.markdown("#### Lineament Density Map")
        st.pyplot(make_fig(sat_data["lineament_density_map"], cmap="hot",
            title="Structural Lineament Density", label="Density", show_targets=True),
            use_container_width=True); plt.close()
    with lm2:
        st.markdown("#### Lineament Intersection Map")
        st.pyplot(make_fig(sat_data["intersection_map"], cmap="magma",
            title="Lineament Intersection Density", label="Index", show_targets=True),
            use_container_width=True); plt.close()

    st.markdown("### Per-Orientation Lineament Maps")
    ori1, ori2 = st.columns(2)
    with ori1:
        st.pyplot(make_fig(sat_data["lineament_ns_map"], cmap="gray", title="N-S Lineaments"), use_container_width=True); plt.close()
    with ori2:
        st.pyplot(make_fig(sat_data["lineament_ew_map"], cmap="gray", title="E-W Lineaments"), use_container_width=True); plt.close()
    ori3, ori4 = st.columns(2)
    with ori3:
        st.pyplot(make_fig(sat_data["lineament_nesw_map"], cmap="gray", title="NE-SW Lineaments"), use_container_width=True); plt.close()
    with ori4:
        st.pyplot(make_fig(sat_data["lineament_nwse_map"], cmap="gray", title="NW-SE Lineaments"), use_container_width=True); plt.close()

    st.markdown("---")
    lm_c1, lm_c2, lm_c3 = st.columns(3)
    lm_c1.metric("Lineament Density Index",     sat_data.get("lineament_density_val", 0))
    lm_c2.metric("High-Confidence Intersections", sat_data.get("intersection_count", 0))
    lm_c3.metric("Intersection Density Index",   sat_data.get("intersection_density_val", 0))

    # ── Exploration Targets ──────────────────────────────────────────────
    if targets:
        st.markdown("---")
        st.markdown("## Exploration Target Zones")
        st.caption("Composite score: IO(0.20) + CLAY(0.20) + Structural(0.15) + Geomorphology(0.30) + Lineament(0.15)")

        high_c = sum(1 for t in targets if t["priority"] == "HIGH")
        med_c  = sum(1 for t in targets if t["priority"] == "MEDIUM")
        low_c  = sum(1 for t in targets if t["priority"] == "LOW")
        tc1, tc2, tc3, tc4 = st.columns(4)
        tc1.metric("Total Targets", len(targets))
        tc2.metric("High Priority 🔴", high_c)
        tc3.metric("Medium Priority 🟠", med_c)
        tc4.metric("Low Priority 🟢", low_c)

        st.dataframe([{
            "ID": t["id"], "Score": t["score"], "Priority": t["priority"],
            "Structural": t["structural_control"], "Lithology": t["lithology"],
            "Radius (m)": t["radius_m"], "Lat": f"{t['lat']:.4f}", "Lon": f"{t['lon']:.4f}",
        } for t in targets], use_container_width=True, hide_index=True)

        st.markdown("### Target Descriptions")
        for t in targets:
            badge = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}.get(t["priority"], "⚪")
            with st.expander(f"{badge} {t['id']} — Score: {t['score']} ({t['priority']})"):
                c1, c2 = st.columns(2)
                c1.markdown(f"**Structural:** {t['structural_control']}")
                c1.markdown(f"**Lithology:** {t['lithology']}")
                c1.markdown(f"**Radius:** ~{t['radius_m']} m")
                c2.markdown(f"**Lat/Lon:** {t['lat']:.6f}, {t['lon']:.6f}")
                c2.markdown(f"**IO={t['io_score']} Clay={t['clay_score']} Struct={t['struct_score']} Geo={t['geomorph_score']} Line={t['line_score']}**")
                st.markdown(f"🇬🇧 {t['description_en']}")
                st.markdown(f"🇲🇿 {t['description_pt']}")

    # ── EXPORTS ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📥 Export & Google Earth Integration")
    license_code = st.session_state["concession_metadata"].get(
        "Codigo da Licenca (Code)",
        st.session_state["concession_metadata"].get("Código da Licença (Code)", "unknown")
    )

    exp1, exp2 = st.columns(2)
    with exp1:
        st.markdown("### Concession Geometry")
        if active_poly:
            kml_str = polygon_to_kml(active_poly, st.session_state.get("concession_metadata"))
            if kml_str:
                st.download_button("📌 Export Boundary (KML)", data=kml_str.encode("utf-8"),
                    file_name=f"concession_{license_code}.kml",
                    mime="application/vnd.google-earth.kml+xml",
                    use_container_width=True)
            geojson_bytes = json.dumps(active_poly, indent=2).encode("utf-8")
            st.download_button("🗺️ Export Boundary (GeoJSON)", data=geojson_bytes,
                file_name=f"concession_{license_code}.geojson",
                mime="application/geo+json",
                use_container_width=True)

    with exp2:
        st.markdown("### Satellite Image Overlays")
        kmz_bytes = create_kmz_bundle(sat_data, polygon_geojson=active_poly,
            metadata=st.session_state.get("concession_metadata"), fetch_bbox=fetch_bbox)
        if kmz_bytes:
            st.download_button("🌍 Export All Overlays (KMZ)", data=kmz_bytes,
                file_name=f"satintel_overlays_{sat_data.get('scene_date', '')}.kmz",
                mime="application/vnd.google-earth.kmz",
                use_container_width=True)
            st.caption("10 georeferenced image overlays + polygon boundary")

    if targets:
        st.markdown("---")
        st.markdown("### Exploration Targets Export")
        targets_kmz = create_targets_kmz(targets, polygon_geojson=active_poly,
            metadata=st.session_state.get("concession_metadata"), sat_data=sat_data)
        if targets_kmz:
            st.download_button(
                f"🎯 Export Targets (KMZ) — {len(targets)} zones",
                data=targets_kmz,
                file_name=f"License{license_code}-GoldExplorationTargets.kmz",
                mime="application/vnd.google-earth.kmz",
                use_container_width=True)
            st.caption("Priority-coded target polygons ready for Google Earth")

    st.markdown("---")
    exp2_c1, exp2_c2 = st.columns(2)
    with exp2_c1:
        geotiff_bytes = create_geotiff_bundle(sat_data, fetch_bbox=fetch_bbox)
        if geotiff_bytes:
            st.download_button("🗂️ Export Rasters (GeoTIFF ZIP)", data=geotiff_bytes,
                file_name=f"satintel_geotiffs_{sat_data.get('scene_date','')}.zip",
                mime="application/zip", use_container_width=True)
    with exp2_c2:
        png_bytes = create_png_bundle(sat_data)
        if png_bytes:
            st.download_button("🖼️ Export Images (PNG ZIP)", data=png_bytes,
                file_name=f"satintel_images_{sat_data.get('scene_date','')}.zip",
                mime="application/zip", use_container_width=True)


# ========================================================
# IBM WATSONX GEOLOGICAL REPORT
# ========================================================
st.markdown("---")
m_data = st.session_state["m_data"] or {}
sat_data = st.session_state.get("satellite_data")
targets  = st.session_state.get("exploration_targets")

# ── Professional report metadata form ──────────────────────
with st.expander("📋 Report Author & Professional Information", expanded=True):
    rc1, rc2 = st.columns(2)
    with rc1:
        report_author = st.text_input("Prepared by (Nome do Responsavel)*", value="",
            placeholder="e.g., Eng. Badr Merhi")
        report_title = st.selectbox("Professional Title (Cargo)", [
            "Geologo Senior de Exploracao",
            "Engenheiro de Minas",
            "Diretor Tecnico",
            "Consultor Geologico",
            "Geofisico",
            "Especialista em Sensoriamento Remoto",
            "Outro",
        ])
        if report_title == "Outro":
            report_title = st.text_input("Specify title", placeholder="Enter your title")
    with rc2:
        report_company = st.text_input("Company / Organization (Empresa)*", value="",
            placeholder="e.g., SatIntel Exploration Ltd.")
        report_license_no = st.text_input("Professional License No. (No. de Inscricao)",
            placeholder="e.g., CEA-1234/MZ")
        report_report_no = st.text_input("Report Reference No. (No. do Relatorio)",
            placeholder="e.g., SAT-2024-001")

    rc3, rc4 = st.columns(2)
    with rc3:
        report_date = st.date_input("Report Date (Data do Relatorio)",
            value=datetime.now())
    with rc4:
        report_classification = st.selectbox("Document Classification", [
            "Confidencial - Uso Interno",
            "Restrito - Cliente",
            "Tecnico - Informativo",
            "Preliminar - Nao Revisado",
        ])

    # Store in session for PDF generation
    st.session_state["report_author"] = report_author
    st.session_state["report_title"] = report_title
    st.session_state["report_company"] = report_company
    st.session_state["report_license_no"] = report_license_no
    st.session_state["report_ref_no"] = report_report_no
    st.session_state["report_date"] = report_date
    st.session_state["report_classification"] = report_classification

if st.button("📋 Generate Comprehensive Geological Synthesis Report",
             use_container_width=True, type="primary"):
    with st.spinner("watsonx.ai a processar analise geologica completa..."):
        client = get_watsonx_client()
        meta = st.session_state["concession_metadata"]

        target_summary = ""
        if targets:
            lines = []
            for t in targets:
                lines.append(
                    f"  {t['id']}: Score={t['score']}, Priority={t['priority']}, "
                    f"Structural={t['structural_control']}, Lithology={t['lithology']}, "
                    f"Radius=~{t['radius_m']}m\n"
                    f"    EN: {t['description_en']}\n    PT: {t['description_pt']}"
                )
            target_summary = "\n".join(lines)

        prompt = f"""[Role: Geologo Senior de Exploracao, Especialista em Metalogenia do Cinturao Moambicano (Pan-African Belt, 650-500 Ma)]

Voce esta a preparar um PARECER TECNICO FORMAL para uma concessao mineira em Mocambique.

=== DADOS DO CADASTRO MINEIRO ===
- Codigo: {meta.get('Codigo da Licenca (Code)', meta.get('Código da Licença (Code)', 'N/A'))}
- Nome: {meta.get('Nome da Concessao', meta.get('Nome da Concessão', ''))}
- Titular: {meta.get('Titular (Holder Company)', '')}
- Dimensao: {meta.get('Area / Dimensao', meta.get('Área / Dimensão', ''))}
- Validade: {meta.get('Data de Validade (Expiry)', '')}
- Substancias: {meta.get('Substancias', meta.get('Substâncias', ''))}
- Coordenadas: {st.session_state['map_center']}
- Ano: {selected_year} | Commodity: {target_commodity}

=== MATRIZ DE TELEMETRIA (5-WAY) ===
- Oxido de Ferro: {m_data.get('Way_1_Iron_Oxide_Gossan', 2.4)}
- Argila/Hidroxilo: {m_data.get('Way_1_Clay_Phyllic', 1.9)}
- Densidade de Falhas: {m_data.get('Way_2_Fault_Density_Index', 0.8)}
- Silicificacao: {m_data.get('Way_3_Silica_Flooding_Cap', 0.6)}
- Estresse Geobotanico: {m_data.get('Way_4_Geobotanical_Stress', 0.34)}
- WLC Score: {m_data.get('Way_5_WLC_Score_Percent', 88.5)}%
- Satelite: {m_data.get('Satellite_Used', 'Predictive')}"""

        if sat_data:
            prompt += f"""

=== CROSTA PCA ===
- Iron Oxide PC{sat_data.get('crosta_iron_pc',0)+1}: mean={sat_data.get('crosta_iron_mean',0)}, anomaly={sat_data.get('crosta_iron_anomaly_pct',0)}%
- Clay PC{sat_data.get('crosta_clay_pc',0)+1}: mean={sat_data.get('crosta_clay_mean',0)}, anomaly={sat_data.get('crosta_clay_anomaly_pct',0)}%

=== ESTRUTURAL ===
- Densidade Lineamentos: {sat_data.get('lineament_density_val', 0)}
- Interseccoes Alta Confianca: {sat_data.get('intersection_count', 0)}
- Indice Densidade Interseccao: {sat_data.get('intersection_density_val', 0)}"""

        if targets:
            high_c = sum(1 for t in targets if t["priority"] == "HIGH")
            med_c  = sum(1 for t in targets if t["priority"] == "MEDIUM")
            low_c  = sum(1 for t in targets if t["priority"] == "LOW")
            prompt += f"""

=== ALVOS DE EXPLORACAO ===
Total: {len(targets)} ({high_c} Alta, {med_c} Media, {low_c} Baixa prioridade)
{target_summary}"""

        prompt += """

=== ESTRUTURA OBRIGATORIA ===
Escreva em PORTUGUES. Siga EXATAMENTE:

**1. RESUMO EXECUTIVO** (2-3 paragrafos)
**2. CONTEXTO GEOLOGICO REGIONAL** (Cinturao Pan-Africano, litologias, controles)
**3. ANALISE DE ALTERACAO HIDROTERMAL** (Crosta PCA + band ratio Way 1)
**4. ANALISE ESTRUTURAL** (lineamentos, interseccoes, orientacoes dominantes)
**5. ALVOS DE EXPLORACAO** (tabela resumo + discutir alvos de ALTA prioridade)
**6. RECOMENDACOES DE CAMPO** (amostragem, trincheiras, sondagens, cronograma)
**7. PARECER FINAL: PERFURAR / NAO PERFURAR** (justificacao quantitativa)

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
        st.markdown("### 📄 Export Professional Report")

        # Build professional PDF
        try:
            pdf = TechnicalReportPDF()
            pdf.alias_nb_pages()
            pdf.author_name = st.session_state.get("report_author", "")
            conces_name = meta.get('Nome da Concessao', meta.get('Nome da Concessão', 'Concessao'))
            lic_code = meta.get('Codigo da Licenca (Code)', meta.get('Código da Licença (Code)', 'N/A'))
            pdf.report_title = f"Parecer Tecnico - {conces_name}"

            # ── COVER PAGE ──────────────────────────────────────
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=25)

            # Top banner
            pdf.set_fill_color(0, 77, 64)
            pdf.rect(0, 0, 210, 40, 'F')
            pdf.set_font('Helvetica', 'B', 22)
            pdf.set_text_color(255, 255, 255)
            pdf.set_xy(0, 12)
            pdf.cell(0, 12, _clean_pdf_text("SATINTEL"), 0, 1, 'C')
            pdf.set_font('Helvetica', '', 10)
            pdf.set_xy(0, 27)
            pdf.cell(0, 6, _clean_pdf_text("Geological & Mining Insights Platform"), 0, 1, 'C')

            # Classification badge
            classification = st.session_state.get("report_classification", "Confidencial")
            pdf.set_font('Helvetica', 'B', 9)
            pdf.set_text_color(180, 0, 0)
            pdf.set_xy(0, 44)
            pdf.cell(0, 6, _clean_pdf_text(classification), 0, 1, 'C')

            # Main title block
            pdf.ln(15)
            pdf.set_font('Helvetica', 'B', 18)
            pdf.set_text_color(0, 77, 64)
            pdf.multi_cell(190, 10, _clean_pdf_text(f"PARECER TECNICO DE EXPLORACAO"))
            pdf.ln(2)
            pdf.set_font('Helvetica', 'B', 14)
            pdf.set_text_color(40, 40, 40)
            pdf.multi_cell(190, 8, _clean_pdf_text(f"Concessao: {conces_name}"))
            pdf.ln(3)
            pdf.set_font('Helvetica', '', 11)
            pdf.set_text_color(80, 80, 80)
            pdf.cell(0, 6, _clean_pdf_text(f"Licenca: {lic_code}"), 0, 1)

            # Divider
            pdf.ln(5)
            pdf.set_draw_color(0, 77, 64)
            pdf.set_line_width(0.5)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(10)

            # Author / professional info box
            pdf.set_font('Helvetica', 'B', 11)
            pdf.set_text_color(0, 77, 64)
            pdf.cell(0, 7, _clean_pdf_text("INFORMACOES DO RESPONSABEL TECNICO"), 0, 1)
            pdf.ln(2)
            pdf.set_font('Helvetica', '', 10)
            pdf.set_text_color(40, 40, 40)

            author = st.session_state.get("report_author", "N/A")
            title = st.session_state.get("report_title", "N/A")
            company = st.session_state.get("report_company", "N/A")
            lic_no = st.session_state.get("report_license_no", "N/A")
            ref_no = st.session_state.get("report_ref_no", "N/A")
            r_date = st.session_state.get("report_date")
            date_str = r_date.strftime("%d/%m/%Y") if r_date else "N/A"

            info_lines = [
                f"Prepared by:    {author}",
                f"Cargo:          {title}",
                f"Empresa:        {company}",
                f"Licenca Prof.:  {lic_no}",
                f"No. Relatorio:  {ref_no}",
                f"Data:           {date_str}",
            ]
            for line in info_lines:
                pdf.cell(0, 6, _clean_pdf_text(line), 0, 1)

            # Concession summary box
            pdf.ln(8)
            pdf.set_font('Helvetica', 'B', 11)
            pdf.set_text_color(0, 77, 64)
            pdf.cell(0, 7, _clean_pdf_text("DADOS DA CONCESSAO"), 0, 1)
            pdf.ln(2)
            pdf.set_font('Helvetica', '', 10)
            pdf.set_text_color(40, 40, 40)

            concession_lines = [
                f"Nome:         {conces_name}",
                f"Titular:      {meta.get('Titular (Holder Company)', 'N/A')}",
                f"Area:         {meta.get('Area / Dimensao', meta.get('Area / Dimensao', 'N/A'))}",
                f"Substancias:  {meta.get('Substancias', meta.get('Substancias', 'N/A'))}",
                f"Status:       {meta.get('Estado (Status)', 'N/A')}",
                f"Validade:     {meta.get('Data de Validade (Expiry)', 'N/A')}",
            ]
            for line in concession_lines:
                pdf.cell(0, 6, _clean_pdf_text(line), 0, 1)

            # Technical summary box
            pdf.ln(8)
            pdf.set_font('Helvetica', 'B', 11)
            pdf.set_text_color(0, 77, 64)
            pdf.cell(0, 7, _clean_pdf_text("DADOS TECNICOS - SENSORIAMENTO REMOTO"), 0, 1)
            pdf.ln(2)
            pdf.set_font('Helvetica', '', 10)
            pdf.set_text_color(40, 40, 40)

            tech_lines = [
                f"Satelite:        {m_data.get('Satellite_Used', 'N/A')}",
                f"Cobertura Nuvens: {m_data.get('cloud_cover', 'N/A')}%",
                f"Cena:            {m_data.get('scene_date', 'N/A')}",
                f"WLC Score:       {m_data.get('Way_5_WLC_Score_Percent', 'N/A')}%",
                f"Alvos Gerados:    {len(targets) if targets else 0}",
            ]
            for line in tech_lines:
                pdf.cell(0, 6, _clean_pdf_text(line), 0, 1)

            # Bottom banner
            pdf.ln(15)
            pdf.set_font('Helvetica', 'I', 8)
            pdf.set_text_color(128, 128, 128)
            pdf.cell(0, 5, _clean_pdf_text(
                "Este documento foi gerado pelo SatIntel AI usando dados Landsat (USGS/NASA) e IBM watsonx.ai."), 0, 1, 'C')
            pdf.cell(0, 5, _clean_pdf_text(
                "O conteudo tecnico deve ser validado por trabalho de campo antes da tomada de decisao."), 0, 1, 'C')

            # ── REPORT CONTENT PAGE ──────────────────────────────
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=25)

            # Clean and format the Watsonx report text
            clean_text = report_text.replace('**', '').replace('*', '').replace('#', '')
            clean_text = clean_text.replace('\r\n', '\n').replace('\r', '\n')

            lines = clean_text.split('\n')
            in_table = False

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    pdf.ln(3)
                    continue

                # Detect section headers (numbered)
                is_header = False
                for prefix in ["1.", "2.", "3.", "4.", "5.", "6.", "7.", "8."]:
                    if stripped.startswith(prefix):
                        is_header = True
                        break

                if is_header and len(stripped) < 100:
                    pdf.ln(3)
                    pdf.set_font('Helvetica', 'B', 12)
                    pdf.set_text_color(0, 77, 64)
                    pdf.multi_cell(190, 6, _clean_pdf_text(stripped))
                    pdf.set_text_color(40, 40, 40)
                    pdf.ln(1)
                elif stripped.startswith("- ") or stripped.startswith("* "):
                    pdf.set_font('Helvetica', '', 10)
                    pdf.set_text_color(50, 50, 50)
                    pdf.cell(5, 5, "", 0, 0)
                    pdf.multi_cell(190, 5, _clean_pdf_text("  - " + stripped[2:]))
                elif "|" in stripped and stripped.count("|") >= 2:
                    # Table-like line
                    pdf.set_font('Courier', '', 9)
                    pdf.set_text_color(50, 50, 50)
                    pdf.multi_cell(190, 5, _clean_pdf_text(stripped))
                else:
                    pdf.set_font('Helvetica', '', 10)
                    pdf.set_text_color(40, 40, 40)
                    pdf.multi_cell(190, 5, _clean_pdf_text(stripped))

            # ── TARGETS SUMMARY PAGE (if targets exist) ──────────
            if targets:
                pdf.add_page()
                pdf.set_auto_page_break(auto=True, margin=25)
                pdf.set_font('Helvetica', 'B', 14)
                pdf.set_text_color(0, 77, 64)
                pdf.multi_cell(190, 8, _clean_pdf_text("RESUMO DE ALVOS DE EXPLORACAO"))
                pdf.ln(3)

                # Table header
                pdf.set_font('Helvetica', 'B', 9)
                pdf.set_fill_color(0, 77, 64)
                pdf.set_text_color(255, 255, 255)
                pdf.cell(15, 7, "ID", 1, 0, 'C', True)
                pdf.cell(20, 7, "Score", 1, 0, 'C', True)
                pdf.cell(25, 7, "Priority", 1, 0, 'C', True)
                pdf.cell(55, 7, "Structural Control", 1, 0, 'C', True)
                pdf.cell(75, 7, "Lithology", 1, 1, 'C', True)

                # Table rows
                pdf.set_text_color(40, 40, 40)
                for t in targets:
                    pdf.set_font('Helvetica', '', 8)
                    if t["priority"] == "HIGH":
                        pdf.set_fill_color(255, 230, 230)
                    elif t["priority"] == "MEDIUM":
                        pdf.set_fill_color(255, 240, 220)
                    else:
                        pdf.set_fill_color(230, 255, 230)
                    pdf.cell(15, 6, _clean_pdf_text(t["id"]), 1, 0, 'C', True)
                    pdf.cell(20, 6, f'{t["score"]}%', 1, 0, 'C', True)
                    pdf.cell(25, 6, _clean_pdf_text(t["priority"]), 1, 0, 'C', True)
                    pdf.cell(55, 6, _clean_pdf_text(t["structural_control"]), 1, 0, 'L', True)
                    pdf.cell(75, 6, _clean_pdf_text(t["lithology"]), 1, 1, 'L', True)

                pdf.ln(5)
                pdf.set_font('Helvetica', '', 9)
                pdf.set_text_color(60, 60, 60)
                for t in targets:
                    pdf.set_font('Helvetica', 'B', 9)
                    pdf.set_text_color(0, 77, 64)
                    pdf.multi_cell(190, 5, _clean_pdf_text(f"  {t['id']} - {t['lithology']} ({t['priority']})"))
                    pdf.set_font('Helvetica', '', 8)
                    pdf.set_text_color(80, 80, 80)
                    pdf.multi_cell(190, 4, _clean_pdf_text(f"    Lat: {t['lat']:.4f}, Lon: {t['lon']:.4f} | Radius: ~{t['radius_m']}m"))
                    pdf.multi_cell(190, 4, _clean_pdf_text(f"    EN: {t['description_en']}"))
                    pdf.multi_cell(190, 4, _clean_pdf_text(f"    PT: {t['description_pt']}"))
                    pdf.ln(2)

            # ── SIGNATURE BLOCK ──────────────────────────────────
            pdf.ln(15)
            pdf.set_draw_color(120, 120, 120)
            pdf.set_line_width(0.3)
            sig_y = pdf.get_y()
            pdf.line(60, sig_y, 140, sig_y)
            pdf.ln(2)
            pdf.set_font('Helvetica', 'B', 10)
            pdf.set_text_color(40, 40, 40)
            pdf.cell(0, 5, _clean_pdf_text(author), 0, 1, 'C')
            pdf.set_font('Helvetica', '', 9)
            pdf.set_text_color(80, 80, 80)
            pdf.cell(0, 5, _clean_pdf_text(title), 0, 1, 'C')
            pdf.cell(0, 5, _clean_pdf_text(company), 0, 1, 'C')
            if lic_no and lic_no != "N/A":
                pdf.cell(0, 5, _clean_pdf_text(f"Licenca Prof.: {lic_no}"), 0, 1, 'C')

            pdf_bytes = bytes(pdf.output())
            st.download_button("📥 Download Professional PDF Report", data=pdf_bytes,
                file_name=f"SatIntel_Report_{lic_code}_{selected_year}.pdf",
                mime="application/pdf", use_container_width=True)

        except Exception as e:
            st.warning(f"PDF export error: {e}")
            # Fallback: clean TXT with header
            header = f"""
SATINTEL - PARECER TECNICO DE EXPLORACAO
==========================================
Concessao: {conces_name}
Licenca: {lic_code}
Data: {date_str}

PREPARED BY:
  Author:  {author}
  Title:   {title}
  Company: {company}
  Lic.No:  {lic_no}
  Ref.No:  {ref_no}
==========================================

"""
            full_text = header + report_text
            if targets:
                full_text += "\n\n=== ALVOS DE EXPLORACAO ===\n"
                for t in targets:
                    full_text += f"\n{t['id']} | Score: {t['score']}% | {t['priority']}\n"
                    full_text += f"  Lithology: {t['lithology']}\n"
                    full_text += f"  Control: {t['structural_control']}\n"
                    full_text += f"  Lat: {t['lat']:.4f}, Lon: {t['lon']:.4f} | Radius: ~{t['radius_m']}m\n"
                    full_text += f"  EN: {t['description_en']}\n"
                    full_text += f"  PT: {t['description_pt']}\n"
            st.download_button("📥 Download Report (TXT)", data=full_text.encode("utf-8"),
                file_name=f"SatIntel_Report_{lic_code}_{selected_year}.txt",
                mime="text/plain", use_container_width=True)
