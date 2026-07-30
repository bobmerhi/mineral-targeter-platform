import warnings
warnings.filterwarnings("ignore", message="Unverified HTTPS request")
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import streamlit as st
from datetime import datetime
st.set_page_config(page_title="SatIntel Moçambique Real-Time AI", layout="wide")

import folium
from streamlit_folium import st_folium
try:
    from ibm_watsonx_ai import APIClient
    from ibm_watsonx_ai.foundation_models import ModelInference
except Exception as _ibm_err:
    import traceback as _tb
    st.error(f"Failed to import ibm_watsonx_ai: {type(_ibm_err).__name__}: {_ibm_err}")
    st.code(_tb.format_exc())
    st.stop()
try:
    from georemote import (
        fetch_and_calculate_spatz,
        get_real_mozambique_cadastre,
        search_cadastre_by_name,
        fetch_satellite_imagery,
        fetch_sentinel2_lithology,
        fetch_aster_tir_indices,
        polygon_to_bbox,
        generate_exploration_targets,
    )
except Exception as _georemote_err:
    import traceback as _tb
    st.error(f"Failed to import georemote: {type(_georemote_err).__name__}: {_georemote_err}")
    st.code(_tb.format_exc())
    st.stop()
try:
    from export_utils import (
        polygon_to_kml,
        create_kmz_bundle,
        create_geotiff_bundle,
        create_png_bundle,
        create_targets_kmz,
    )
except Exception as _export_err:
    import traceback as _tb
    st.error(f"Failed to import export_utils: {type(_export_err).__name__}: {_export_err}")
    st.code(_tb.format_exc())
    st.stop()

try:
    from pdf_report import generate_professional_report
except Exception as _pdf_err:
    import traceback as _tb
    st.error(f"Failed to import pdf_report: {type(_pdf_err).__name__}: {_pdf_err}")
    st.code(_tb.format_exc())
    st.stop()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import json


# ========================================================
# PDF generation handled by pdf_report.py module (DejaVuSans UTF-8)
# ========================================================


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
    "name_search_results": None,
    "name_search_term": "",
    "cached_report_text": "",
    "lithology_data": None,
    "aster_tir_data": None,
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
    ["(a) License # Search", "(b) License Name Search", "(c) Coordinates + Radius", "(d) Map Selection"]
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
elif search_method == "(b) License Name Search":
    name_input = st.sidebar.text_input(
        "Enter License Name or Holder",
        value=st.session_state.get("name_search_term", ""),
        placeholder="e.g., Tete Platinum, EXXON, or Ouro",
        help="Search by concession name or holder company. Multiple results may appear."
    )
    name_search_clicked = st.sidebar.button("Search by Name", type="primary", use_container_width=True)

    if name_search_clicked and name_input.strip():
        with st.sidebar.status("Scanning 14,800+ INAMI licenses...", expanded=True) as name_status:
            st.write("Getting all license IDs from Landfolio...")
            _progress_bar = st.progress(0)
            _progress_text = st.empty()

            def _name_progress_cb(i, total):
                pct = int(i / total * 100) if total > 0 else 0
                _progress_bar.progress(pct)
                _progress_text.caption(f"Scanned {i:,} / {total:,} records…")

            results = search_cadastre_by_name(name_input.strip(), progress_cb=_name_progress_cb)
            _progress_bar.progress(100)
            _progress_text.empty()
            if results:
                st.session_state["name_search_results"] = results
                st.session_state["name_search_term"] = name_input.strip()
                name_status.update(label=f"Found {len(results)} matching license(s)", state="complete", expanded=False)
            else:
                st.session_state["name_search_results"] = None
                name_status.update(label=f"No licenses found for '{name_input.strip()}'", state="error")

    # Show search results for selection
    if st.session_state.get("name_search_results"):
        results = st.session_state["name_search_results"]
        st.sidebar.markdown(f"**{len(results)} license(s) found** \u2014 select one:")

        labels = []
        for r in results:
            holder_short = r['holder'][:30] if r['holder'] != 'N/A' else 'N/A'
            labels.append(f"{r['code']} \u2014 {r['name']} ({holder_short})")

        selected_idx = st.sidebar.selectbox(
            "Select License",
            range(len(results)),
            format_func=lambda i: labels[i],
            key="name_search_select"
        )

        if selected_idx is not None:
            r = results[selected_idx]
            with st.sidebar.container():
                st.markdown(f"""
                **{r['name']}**
                - Code: {r['code']} | Status: {r['status']}
                - Holder: {r['holder']}
                - Area: {r['area']}
                - Commodities: {r['commodities']}
                - Expiry: {r['expiry']}
                """)

            if st.sidebar.button("Load Selected License", type="primary", use_container_width=True):
                with st.sidebar.status("Loading license...", expanded=True) as load_status:
                    st.write(f"Fetching {r['code']} from cadastre...")
                    db_result = get_real_mozambique_cadastre(r["code"])
                    if db_result["found"]:
                        name = db_result.get("metadata", {}).get("Nome da Concessao", r["code"])
                        st.write(f"Found: {name}")
                        st.session_state["map_center"]         = [db_result["lat"], db_result["lon"]]
                        st.session_state["active_polygon"]      = db_result["polygon"]
                        st.session_state["concession_metadata"] = db_result["metadata"]
                        st.session_state["satellite_data"]      = None
                        st.session_state["exploration_targets"] = None
                        st.session_state["last_license"]        = r["code"]
                        st.session_state["m_data"] = fetch_and_calculate_spatz(
                            [db_result["lat"], db_result["lon"]], None, selected_year
                        )
                        st.session_state["m_data"]["_is_predictive"] = True
                        load_status.update(label=f"License {r['code']} loaded!", state="complete", expanded=False)
                    else:
                        load_status.update(label=f"Failed to load license {r['code']}", state="error")

    elif st.session_state.get("name_search_term"):
        st.sidebar.info(f"No results for '{st.session_state['name_search_term']}'. Try a different name.")

elif search_method == "(c) Coordinates + Radius":
    st.sidebar.markdown("Paste coordinates (e.g. `-15.5451, 34.1422`)")
    coord_paste = st.sidebar.text_input(
        "Lat, Lon",
        value=st.session_state.get("coord_paste_val", ""),
        placeholder="-15.5451, 34.1422",
        help="Cola ou digita coordenadas no formato: latitude, longitude"
    )
    radius_m = st.sidebar.slider(
        "Detection Radius (meters)", min_value=500, max_value=50000,
        value=st.session_state.get("radius_m_val", 5000), step=500,
        help="Raio de deteccao em metros (ex: 5000m = 5km)"
    )
    search_clicked = st.sidebar.button("Search Coordinates", type="primary", use_container_width=True)

    if search_clicked:
        import re
        parsed = re.findall(r'-?\d+\.?\d*', coord_paste.strip())
        if len(parsed) < 2:
            st.sidebar.error("Formato invalido. Exemplo: -15.5451, 34.1422")
            st.stop()
        coord_lat = float(parsed[0])
        coord_lon = float(parsed[1])
        if not (-27.0 <= coord_lat <= -10.0) or not (30.0 <= coord_lon <= 42.0):
            st.sidebar.error("Coordenadas fora de Mocambique. Lat: -27 a -10 | Lon: 30 a 42")
            st.stop()

        st.session_state["coord_paste_val"] = coord_paste.strip()
        st.session_state["radius_m_val"] = radius_m

        with st.sidebar.status("Creating search area...", expanded=True) as cad_status:
            st.write(f"Center: {coord_lat:.6f}, {coord_lon:.6f}")
            st.write(f"Radius: {radius_m}m ({radius_m/1000:.1f} km)")

            # Generate circular polygon from coordinates + radius
            import math
            R_EARTH = 6378137.0  # meters
            num_points = 64
            coords_ring = []
            for i in range(num_points + 1):
                angle = 2 * math.pi * i / num_points
                d_lat = (radius_m / R_EARTH) * math.cos(angle) * (180 / math.pi)
                d_lon = (radius_m / R_EARTH) * math.sin(angle) * (180 / math.pi) / max(math.cos(math.radians(coord_lat)), 1e-6)
                coords_ring.append([coord_lon + d_lon, coord_lat + d_lat])

            circle_polygon = {
                "type": "Feature",
                "properties": {"name": f"Custom Area ({radius_m}m radius)"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords_ring]
                }
            }

            st.session_state["map_center"]         = [coord_lat, coord_lon]
            st.session_state["active_polygon"]      = circle_polygon
            st.session_state["concession_metadata"] = {
                "Codigo da Licenca (Code)": "CUSTOM",
                "Nome da Concessao": f"Custom Area ({radius_m/1000:.1f}km radius)",
                "Titular (Holder Company)": "Custom Search",
                "Area / Dimensao": f"~{math.pi * (radius_m/1000)**2:.1f} km2",
                "Data de Validade (Expiry)": "N/A",
                "Substancias": "Gold / Multi-mineral",
            }
            st.session_state["satellite_data"]      = None
            st.session_state["exploration_targets"] = None
            st.session_state["m_data"] = fetch_and_calculate_spatz(
                [coord_lat, coord_lon], None, selected_year
            )
            st.session_state["m_data"]["_is_predictive"] = True
            cad_status.update(label=f"Search area created ({radius_m/1000:.1f}km radius)", state="complete", expanded=False)
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
st.session_state["exploration_targets"] = generate_exploration_targets(result,polygon_geojson=st.session_state.get("active_polygon"),target_commodity=target_commodity  # ← Connects sidebar selection to WLC formula
)                st.session_state["m_data"] = {
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
                    masked = img_array.copy().astype(np.float64)
                    masked[~mask] = 0  # Use 0 instead of NaN so colormap renders properly
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
        lm_data = sat_data["lineament_density_map"]
        lm_vmax = float(np.nanmax(lm_data)) if np.nanmax(lm_data) > 0 else 1.0
        st.pyplot(make_fig(lm_data, cmap="hot", vmin=0, vmax=lm_vmax,
            title="Structural Lineament Density", label="Density", show_targets=True),
            use_container_width=True); plt.close()
    with lm2:
        st.markdown("#### Lineament Intersection Map")
        im_data = sat_data["intersection_map"]
        im_vmax = float(np.nanmax(im_data)) if np.nanmax(im_data) > 0 else 1.0
        st.pyplot(make_fig(im_data, cmap="magma", vmin=0, vmax=im_vmax,
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

        st.markdown("### Descrições Detalhadas dos Alvos")
        for t in targets:
            badge = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}.get(t["priority"], "⚪")
            priority_color = {"HIGH": "#dc3545", "MEDIUM": "#fd7e14", "LOW": "#198754"}.get(t["priority"], "#6c757d")
            with st.expander(f"{badge} {t['id']} — {t['lithology']} ({t['priority']}) | Score: {t['score']}", expanded=False):
                # Metadata row
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"**🧭 Controlo Estrutural**  \n{t['structural_control']}")
                c2.markdown(f"**🪨 Litologia**  \n{t['lithology']}")
                c3.markdown(f"**📍 Coordenadas**  \nLat: {t['lat']:.4f} | Lon: {t['lon']:.4f}  \n**Raio:** ~{t['radius_m']} m")

                # Score breakdown
                st.markdown(f"""
<div style="background:#f8f9fa;border-radius:6px;padding:8px 12px;margin:8px 0;font-size:0.85em;color:#333">
<b>Scores:</b> &nbsp;
IO = {t['io_score']} &nbsp;|&nbsp;
Clay = {t['clay_score']} &nbsp;|&nbsp;
Struct = {t['struct_score']} &nbsp;|&nbsp;
Geo = {t['geomorph_score']} &nbsp;|&nbsp;
Line = {t['line_score']} &nbsp;|&nbsp;
<b>Composto = {t['score']}</b>
</div>
""", unsafe_allow_html=True)

                # Bilingual descriptions — full width, word wrapped
                st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px">
  <div style="background:#e8f4fd;border-left:4px solid #0d6efd;border-radius:4px;padding:10px 14px">
    <div style="font-size:0.8em;font-weight:600;color:#0d6efd;margin-bottom:4px">🇬🇧 ENGLISH</div>
    <div style="font-size:0.9em;line-height:1.5;word-wrap:break-word;white-space:normal">{t['description_en']}</div>
  </div>
  <div style="background:#fff3cd;border-left:4px solid #ffc107;border-radius:4px;padding:10px 14px">
    <div style="font-size:0.8em;font-weight:600;color:#856404;margin-bottom:4px">🇲🇿 PORTUGUÊS</div>
    <div style="font-size:0.9em;line-height:1.5;word-wrap:break-word;white-space:normal">{t['description_pt']}</div>
  </div>
</div>
""", unsafe_allow_html=True)

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
        report_author = st.text_input(
            "Prepared by (Nome do Responsavel)*",
            value=st.session_state.get("report_author", ""),
            placeholder="e.g., Eng. Yasser Abu baker",
            key="form_author"
        )
        title_options = [
            "Geologo Senior de Exploracao",
            "Engenheiro de Minas",
            "Diretor Tecnico",
            "Consultor Geologico",
            "Geofisico",
            "Especialista em Sensoriamento Remoto",
            "Outro",
        ]
        saved_title = st.session_state.get("report_title", "Geologo Senior de Exploracao")
        title_idx = title_options.index(saved_title) if saved_title in title_options else 0
        report_title = st.selectbox(
            "Professional Title (Cargo)", title_options,
            index=title_idx, key="form_title"
        )
        if report_title == "Outro":
            report_title = st.text_input(
                "Specify title",
                value=st.session_state.get("report_title_custom", ""),
                placeholder="Enter your title", key="form_title_custom"
            )
            st.session_state["report_title_custom"] = report_title
    with rc2:
        report_company = st.text_input(
            "Company / Organization (Empresa)*",
            value=st.session_state.get("report_company", ""),
            placeholder="e.g., GIS & REMOTE SENSING POPULAR, Lda",
            key="form_company"
        )
        report_license_no = st.text_input(
            "Professional License No. (No. de Inscricao)",
            value=st.session_state.get("report_license_no", ""),
            placeholder="e.g., 10000",
            key="form_license_no"
        )
        report_report_no = st.text_input(
            "Report Reference No. (No. do Relatorio)",
            value=st.session_state.get("report_ref_no", ""),
            placeholder="e.g., POP002/07/26",
            key="form_report_no"
        )

    rc3, rc4 = st.columns(2)
    with rc3:
        report_date = st.date_input(
            "Report Date (Data do Relatorio)",
            value=st.session_state.get("report_date", datetime.now().date()),
            key="form_date"
        )
    with rc4:
        classif_options = [
            "Confidencial - Uso Interno",
            "Restrito - Cliente",
            "Tecnico - Informativo",
            "Preliminar - Nao Revisado",
        ]
        saved_classif = st.session_state.get("report_classification", "Confidencial - Uso Interno")
        classif_idx = classif_options.index(saved_classif) if saved_classif in classif_options else 0
        report_classification = st.selectbox(
            "Document Classification", classif_options,
            index=classif_idx, key="form_classification"
        )

    # Persist immediately so values survive reruns
    st.session_state["report_author"]         = report_author
    st.session_state["report_title"]          = report_title
    st.session_state["report_company"]        = report_company
    st.session_state["report_license_no"]     = report_license_no
    st.session_state["report_ref_no"]         = report_report_no
    st.session_state["report_date"]           = report_date
    st.session_state["report_classification"] = report_classification



# ── HOST ROCK LITHOLOGY SECTION (2024-2026 Methods) ─────────────
st.markdown("---")
st.markdown("### 🪨 Host Rock Lithology Identification")
st.caption("Sentinel-2 spectral indices (2024-2025) + ASTER Thermal Infrared (Ninomiya)")

col_lith1, col_lith2 = st.columns(2)

with col_lith1:
    if st.button("🔬 Run Sentinel-2 Lithology Analysis", use_container_width=True):
        with st.status("Fetching Sentinel-2 data & computing lithology indices...", expanded=True) as lith_status:
            st.write("Connecting to Earth Search (AWS STAC)...")
            try:
                lith_result = fetch_sentinel2_lithology(
                    st.session_state["map_center"][0],
                    st.session_state["map_center"][1],
                    selected_year,
                    bbox=st.session_state.get("satellite_data", {}).get("fetch_bbox") if st.session_state.get("satellite_data") else None,
                    progress_cb=lambda msg: st.write(msg),
                )
                if lith_result:
                    st.session_state["lithology_data"] = lith_result
                    lith_status.update(label="Lithology analysis complete!", state="complete", expanded=False)
                else:
                    lith_status.update(label="No Sentinel-2 data available for this area", state="error")
            except Exception as e:
                lith_status.update(label=f"Error: {e}", state="error")

with col_lith2:
    if st.button("🌋 Run ASTER TIR Silicate Analysis", use_container_width=True):
        with st.status("Fetching ASTER Thermal Infrared data...", expanded=True) as tir_status:
            st.write("Searching Planetary Computer ASTER L1T...")
            try:
                tir_result = fetch_aster_tir_indices(
                    st.session_state["map_center"][0],
                    st.session_state["map_center"][1],
                    selected_year,
                    bbox=st.session_state.get("satellite_data", {}).get("fetch_bbox") if st.session_state.get("satellite_data") else None,
                    progress_cb=lambda msg: st.write(msg),
                )
                if tir_result:
                    st.session_state["aster_tir_data"] = tir_result
                    tir_status.update(label="ASTER TIR analysis complete!", state="complete", expanded=False)
                else:
                    tir_status.update(label="No ASTER TIR data available", state="error")
            except Exception as e:
                tir_status.update(label=f"Error: {e}", state="error")

# Display lithology results
lith = st.session_state.get("lithology_data")
if lith:
    st.markdown("#### 🪨 Sentinel-2 Host Rock Classification")
    dom = lith["dominant_lithology"]
    pct = lith["lithology_percentages"][dom]
    if "Mafic" in dom:
        badge_color = "#228B22"
    elif "Felsic" in dom:
        badge_color = "#C71585"
    elif "Graphitic" in dom:
        badge_color = "#444444"
    else:
        badge_color = "#CC4400"
    st.markdown(f'<div style="background:{badge_color};color:white;padding:10px;border-radius:8px;text-align:center;font-size:16px;font-weight:bold;">Dominant Host Rock: {dom} ({pct}%)</div>', unsafe_allow_html=True)

    st.markdown("**Lithology Distribution:**")
    for name, pct_val in lith["lithology_percentages"].items():
        st.write(f"- {name}: **{pct_val}%**")

    st.markdown("**Spectral Indices (2024-2025):**")
    idx_col1, idx_col2 = st.columns(2)
    with idx_col1:
        st.metric("AGDI (Amphibolite-Gneiss)", lith["agdi_val"])
        st.metric("FSI (Ferrous Silicate)", lith["fsi_val"])
        st.metric("FEI (Ferrous Iron 2024)", lith["fei_val"])
    with idx_col2:
        st.metric("NDGI (Graphite 2024)", lith["ndgi_val"])
        st.metric("Clay/Felsic (B11/B12)", lith["clay_felsic_val"])
        st.metric("Iron Oxide (B4/B2)", lith["iron_oxide_val"])

    st.markdown("**Lithology Classification Scores:**")
    score_col1, score_col2 = st.columns(2)
    with score_col1:
        st.metric("Mafic Score", f"{lith['mafic_score']:.3f}")
        st.metric("Graphitic Score", f"{lith['graphite_score']:.3f}")
    with score_col2:
        st.metric("Felsic Score", f"{lith['felsic_score']:.3f}")
        st.metric("Gossan Score", f"{lith['gossan_score']:.3f}")

    st.markdown("**Lithology Maps:**")
    map_col1, map_col2 = st.columns(2)
    with map_col1:
        st.markdown("**Host Rock Classification Map**")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(lith["lithology_classified"])
        ax.set_title("Classified Lithology", fontsize=10)
        ax.axis("off")
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=[0.2,0.6,0.2], label="Mafic (Amphibolite)"),
            Patch(facecolor=[0.9,0.4,0.8], label="Felsic (Gneiss)"),
            Patch(facecolor=[0.3,0.3,0.3], label="Graphitic Schist"),
            Patch(facecolor=[0.9,0.3,0.1], label="Iron Oxide/Gossan"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=6, framealpha=0.8)
        st.pyplot(fig)
        plt.close()

        st.markdown("**AGDI (Amphibolite-Gneiss Discriminator, 2025)**")
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        im2 = ax2.imshow(lith["agdi_map"], cmap="RdYlGn_r")
        ax2.set_title("AGDI \u2014 Low = Amphibolite, High = Gneiss", fontsize=8)
        ax2.axis("off")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        st.pyplot(fig2)
        plt.close()

    with map_col2:
        st.markdown("**Lithology Composite (B12-B11-B02)**")
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        ax3.imshow(lith["lithology_rgb"])
        ax3.set_title("S2 Geology Composite", fontsize=10)
        ax3.axis("off")
        st.pyplot(fig3)
        plt.close()

        st.markdown("**Alteration Composite (Clay-Mafic-IronOxide)**")
        fig4, ax4 = plt.subplots(figsize=(6, 6))
        ax4.imshow(lith["alteration_rgb"])
        ax4.set_title("Alteration/Lithology RGB", fontsize=10)
        ax4.axis("off")
        st.pyplot(fig4)
        plt.close()

    with st.expander("\U0001f4ca Detailed Spectral Index Maps"):
        idx_map_col1, idx_map_col2 = st.columns(2)
        with idx_map_col1:
            st.markdown("**FSI (Ferrous Silicate \u2014 Mafic)**")
            fig5, ax5 = plt.subplots(figsize=(5, 5))
            im5 = ax5.imshow(lith["fsi_map"], cmap="YlOrRd")
            ax5.set_title("FSI \u2014 High = Mafic/Greenstone", fontsize=8)
            ax5.axis("off")
            plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
            st.pyplot(fig5)
            plt.close()

            st.markdown("**NDGI (Graphite Index, 2024)**")
            fig6, ax6 = plt.subplots(figsize=(5, 5))
            im6 = ax6.imshow(lith["ndgi_map"], cmap="gray")
            ax6.set_title("NDGI \u2014 High = Graphitic Schist", fontsize=8)
            ax6.axis("off")
            plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
            st.pyplot(fig6)
            plt.close()

        with idx_map_col2:
            st.markdown("**FEI (Ferrous Iron Index, 2024)**")
            fig7, ax7 = plt.subplots(figsize=(5, 5))
            im7 = ax7.imshow(lith["fei_map"], cmap="coolwarm")
            ax7.set_title("FEI \u2014 High = Mafic Dykes", fontsize=8)
            ax7.axis("off")
            plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
            st.pyplot(fig7)
            plt.close()

            st.markdown("**Clay/Felsic Index**")
            fig8, ax8 = plt.subplots(figsize=(5, 5))
            im8 = ax8.imshow(lith["clay_felsic_map"], cmap="YlOrBr")
            ax8.set_title("Clay/Felsic (B11/B12)", fontsize=8)
            ax8.axis("off")
            plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
            st.pyplot(fig8)
            plt.close()

# Display ASTER TIR results
tir = st.session_state.get("aster_tir_data")
if tir:
    st.markdown("#### \U0001f30b ASTER Thermal Infrared \u2014 Silicate Rock Indices")
    tir_col1, tir_col2 = st.columns(2)
    with tir_col1:
        st.metric("Quartz Index (QI)", tir["qi_val"], help="High = quartz-rich rocks, quartz veins, silicification")
        st.metric("Carbonate Index (CI)", tir["ci_val"], help="High = marble, limestone, carbonatite")
        st.metric("Mafic Index (MI)", tir["mi_val"], help="High = amphibolite, basalt, gabbro")

        st.markdown("**Interpretation:**")
        if tir["qi_val"] > 1.0:
            st.write("\U0001f7e1 High quartz content \u2014 silicification or quartz-rich host rocks")
        if tir["mi_val"] > 1.0:
            st.write("\U0001f7e2 High mafic content \u2014 amphibolites, greenstones, mafic dykes")
        if tir["ci_val"] > 1.0:
            st.write("\u26aa High carbonate content \u2014 marble, carbonatite, or carbonate alteration")

    with tir_col2:
        st.markdown("**TIR Composite (QI-CI-MI)**")
        fig_tir, ax_tir = plt.subplots(figsize=(6, 6))
        ax_tir.imshow(tir["tir_rgb"])
        ax_tir.set_title("ASTER TIR Index Composite", fontsize=10)
        ax_tir.axis("off")
        st.pyplot(fig_tir)
        plt.close()

st.markdown("---")
if st.button("📋 Generate Comprehensive Geological Synthesis Report",
         use_container_width=True, type="primary"):
    # Warn if author info is empty
    if not st.session_state.get("report_author", "").strip():
        st.warning("⚠️ Por favor preencha o campo **Prepared by** antes de gerar o relatório.", icon="⚠️")
        st.stop()
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

        prompt = f"""Redija agora um PARECER TECNICO GEOLOGICO FORMAL e completo em Portugues, com base exclusivamente nos dados abaixo. NAO faca perguntas. NAO aguarde instrucoes. Escreva o parecer directamente.

    ESPECIALIDADE: Geologo Senior, Metalogenia do Cinturao Moambicano (Pan-African Belt, 650-500 Ma)

    CONCESSAO MINEIRA:

    DADOS DO CADASTRO MINEIRO:
    - Codigo: {meta.get('Codigo da Licenca (Code)', meta.get('Código da Licença (Code)', 'N/A'))}
    - Nome: {meta.get('Nome da Concessao', meta.get('Nome da Concessão', ''))}
    - Titular: {meta.get('Titular (Holder Company)', '')}
    - Dimensao: {meta.get('Area / Dimensao', meta.get('Área / Dimensão', ''))}
    - Validade: {meta.get('Data de Validade (Expiry)', '')}
    - Substancias: {meta.get('Substancias', meta.get('Substâncias', ''))}
    - Coordenadas: {st.session_state['map_center']}
    - Ano: {selected_year} | Commodity: {target_commodity}

    MATRIZ DE TELEMETRIA (5-WAY):
    - Oxido de Ferro: {m_data.get('Way_1_Iron_Oxide_Gossan', 2.4)}
    - Argila/Hidroxilo: {m_data.get('Way_1_Clay_Phyllic', 1.9)}
    - Densidade de Falhas: {m_data.get('Way_2_Fault_Density_Index', 0.8)}
    - Silicificacao: {m_data.get('Way_3_Silica_Flooding_Cap', 0.6)}
    - Estresse Geobotanico: {m_data.get('Way_4_Geobotanical_Stress', 0.34)}
    - WLC Score: {m_data.get('Way_5_WLC_Score_Percent', 88.5)}%
    - Satelite: {m_data.get('Satellite_Used', 'Predictive')}"""

        if sat_data:
            prompt += f"""

    CROSTA PCA:
    - Iron Oxide PC{sat_data.get('crosta_iron_pc',0)+1}: mean={sat_data.get('crosta_iron_mean',0)}, anomaly={sat_data.get('crosta_iron_anomaly_pct',0)}%
    - Clay PC{sat_data.get('crosta_clay_pc',0)+1}: mean={sat_data.get('crosta_clay_mean',0)}, anomaly={sat_data.get('crosta_clay_anomaly_pct',0)}%

    ANALISE ESTRUTURAL:
    - Densidade Lineamentos: {sat_data.get('lineament_density_val', 0)}
    - Interseccoes Alta Confianca: {sat_data.get('intersection_count', 0)}
    - Indice Densidade Interseccao: {sat_data.get('intersection_density_val', 0)}"""

    
        # Add lithology data to prompt if available
        litho = st.session_state.get("lithology_data")
        if litho:
            prompt += f"""

    IDENTIFICACAO DA ROCHA HOSPEDEIRA (Sentinel-2, 2024-2025):
    - Rocha Dominante: {litho['dominant_lithology']}
    - Distribuicao Litologica: {litho['lithology_percentages']}
    - AGDI (Anfibolito-Gnaisse, 2025): {litho['agdi_val']}
    - FSI (Silicato Ferroso): {litho['fsi_val']}
    - FEI (Ferro Ferroso, 2024): {litho['fei_val']}
    - NDGI (Grafite, 2024): {litho['ndgi_val']}
    - Clay/Felsic (B11/B12): {litho['clay_felsic_val']}
    - Iron Oxide (B4/B2): {litho['iron_oxide_val']}
    - Score Mafico: {litho['mafic_score']} | Score Felsico: {litho['felsic_score']}
    - Score Grafite: {litho['graphite_score']} | Score Gossan: {litho['gossan_score']}"""

        tir_data = st.session_state.get("aster_tir_data")
        if tir_data:
            prompt += f"""

    INDICES TIR ASTER (Silicatos, Ninomiya):
    - Quartz Index: {tir_data['qi_val']}
    - Carbonate Index: {tir_data['ci_val']}
    - Mafic Index: {tir_data['mi_val']}"""

        if targets:
            high_c = sum(1 for t in targets if t["priority"] == "HIGH")
            med_c  = sum(1 for t in targets if t["priority"] == "MEDIUM")
            low_c  = sum(1 for t in targets if t["priority"] == "LOW")
            prompt += f"""

    ALVOS DE EXPLORACAO:
    Total: {len(targets)} ({high_c} Alta, {med_c} Media, {low_c} Baixa prioridade)
    {target_summary}"""

        prompt += """

    INSTRUCOES DE REDACCAO:
    - Escreva o parecer tecnico COMPLETO em Portugues AGORA. Cada seccao deve ter pelo menos 2-3 paragrafos de conteudo real.
    - PROIBIDO escrever "FIM", "Nota:", "Aguardo", ou qualquer comentario sobre o proprio texto.
    - PROIBIDO dizer que o parecer ja foi escrito ou que nao precisa de resposta adicional.
    - PROIBIDO repetir a mesma frase ou paragrafo. Cada seccao deve ter conteudo unico e distinto.
    - Escreva APENAS o conteudo tecnico das 7 seccoes. Nada de conversacao, meta-comentarios, ou frases de espera.
    - Use terminologia geologica tecnica e seja quantitativo (cite os valores numericos fornecidos acima).
    - Estruture OBRIGATORIAMENTE em 7 seccoes, cada uma com texto substantivo:

    1. RESUMO EXECUTIVO
    Escreva 3 paragrafos: sintese dos dados telemetricos, prospectividade geral, e recomendacao principal. Mencione a concessao, titulares, e o WLC Score.

    2. CONTEXTO GEOLOGICO REGIONAL
    Escreva 3 paragrafos: enquadramento no Cinturao Pan-Africano Mocambicano (650-500 Ma), descricao litologica, e controles estruturais regionais.

    3. ANALISE DE ALTERACAO HIDROTERMAL
    Escreva 3 paragrafos: interpretacao dos indices de oxido de ferro e argila, resultados do Crosta PCA, e classificacao da intensidade de alteracao. Cite os valores numericos.

    3.5 IDENTIFICACAO DA ROCHA HOSPEDEIRA (se dados Sentinel-2/ASTER disponiveis)
    Escreva 2-3 paragrafos: interpretacao dos indices AGDI, FSI, FEI, NDGI, e classificacao litologica dominante. Identifique o tipo de rocha hospedeira (anfibolito, gnaisse granitico, xisto grafítico, etc.) e sua relacao com a mineralizacao aurifera. Se disponivel, interprete os indices TIR do ASTER (Quartz, Carbonate, Mafic).

    4. ANALISE ESTRUTURAL
    Escreva 3 paragrafos: interpretacao de lineamentos, interseccoes de alta confianca, orientacoes dominantes, e o papel no controle da mineralizacao.

    5. ALVOS DE EXPLORACAO PRIORIZADOS
    Escreva uma tabela e 2 paragrafos: liste os alvos de ALTA prioridade com localizacao, score, e justificacao geologica para cada um.

    6. RECOMENDACOES DE CAMPO (PROGRAMA DE TRABALHO)
    Escreva 3 paragrafos: Fase 1 (prospeccao geoquimica), Fase 2 (trincheiras), Fase 3 (sondagens). Inclua cronograma e custos estimados.

    7. PARECER FINAL
    Escreva 2 paragrafos: veredicto claro (PERFURAR / NAO PERFURAR / RECLASSIFICAR) com justificacao quantitativa baseada no WLC Score e indices telemetricos.

    Comece imediatamente a escrever a Secao 1. NAO escreva nada antes da Secao 1. NAO escreva "FIM" no final."""

        model = ModelInference(
            model_id="meta-llama/llama-3-3-70b-instruct",
            credentials=credentials,
            project_id=PROJECT_ID,
            params={"max_new_tokens": 4000, "temperature": 0.3, "decoding_method": "greedy", "stop_sequences": ["FIM", "(NOTA"]}
        )
        raw_report = model.generate_text(prompt=prompt)
        # ── Anti-loop post-processing ──────────────────────────────────────
        import re as _re

        def _clean_report(text):
            # 1. Cut at first occurrence of loop triggers (case-insensitive)
            triggers = [
                r'\nFIM DO PARECER',
                r'\nFIM\.?\s*\n',
                r'\nAguardo sua resposta',
                r'\nA resposta é o parecer',
                r'\n\(Nota:',
                r'\n\(NOTA:',
            ]
            for t in triggers:
                match = _re.search(t, text, flags=_re.IGNORECASE)
                if match:
                    text = text[:match.start()].strip()
            # 2. Remove any remaining isolated FIM lines
            text = _re.sub(r'(?m)^FIM\.?\s*$', '', text)
            # 3. Remove (NOTA...) blocks anywhere
            text = _re.sub(r'\(NOTA[^)]*\)', '', text, flags=_re.IGNORECASE)
            # 4. Remove (Nota...) blocks
            text = _re.sub(r'\(Nota[^)]*\)', '', text, flags=_re.IGNORECASE)
            # 5. Collapse excessive blank lines
            text = _re.sub(r'\n{3,}', '\n\n', text)
            return text.strip()

        report_text = _clean_report(raw_report)
        st.session_state["cached_report_text"] = report_text
        st.markdown(report_text)

        st.markdown("---")
        st.markdown("### 📄 Export Professional Report")

        # Pre-define variables
        conces_name = meta.get('Nome da Concessão', meta.get('Nome da Concessao', 'Concessão'))
        lic_code = meta.get('Código da Licença (Code)', meta.get('Codigo da Licenca (Code)', 'N/A'))
        author = st.session_state.get("report_author", "N/A")
        title = st.session_state.get("report_title", "N/A")
        company = st.session_state.get("report_company", "N/A")
        lic_no = st.session_state.get("report_license_no", "N/A")
        ref_no = st.session_state.get("report_ref_no", "N/A")
        r_date = st.session_state.get("report_date")
        date_str = r_date.strftime("%d/%m/%Y") if r_date else "N/A"

        author_info = {
            "author": author,
            "title": title,
            "company": company,
            "license_no": lic_no,
            "ref_no": ref_no,
            "date": date_str,
            "classification": st.session_state.get("report_classification", "Confidencial — Uso Interno"),
        }

        active_poly = st.session_state.get("active_polygon")
        fbbox = st.session_state.get("satellite_data", {}).get("fetch_bbox") if st.session_state.get("satellite_data") else None

        try:
            pdf_bytes = generate_professional_report(
                report_text=report_text,
                sat_data=sat_data,
                m_data=m_data,
                meta=meta,
                targets=targets,
                polygon=active_poly,
                fetch_bbox=fbbox,
                author_info=author_info,
                selected_year=selected_year,
                target_commodity=target_commodity,
                map_center=st.session_state["map_center"],
            )
            st.download_button("📥 Download Professional PDF Report", data=pdf_bytes,
                file_name=f"SatIntel_Report_{lic_code}_{selected_year}.pdf",
                mime="application/pdf", use_container_width=True)
        except Exception as e:
            st.warning(f"PDF export error: {e}")
            # Fallback: TXT with header
            header = f"""
    SATINTEL — PARECER TÉCNICO DE EXPLORAÇÃO
    ==========================================
    Concessão: {conces_name}
    Licença: {lic_code}
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
                full_text += "\n\n=== ALVOS DE EXPLORAÇÃO ===\n"
                for t in targets:
                    full_text += f"\n{t['id']} | Score: {t['score']}% | {t['priority']}\n"
                    full_text += f"  Lithology: {t['lithology']}\n"
                    full_text += f"  Lat: {t['lat']:.4f}, Lon: {t['lon']:.4f}\n"
            st.download_button("📥 Download Report (TXT)", data=full_text.encode("utf-8"),
                file_name=f"SatIntel_Report_{lic_code}_{selected_year}.txt",
                mime="text/plain", use_container_width=True)
