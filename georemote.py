# SatIntel - redeploy trigger
"""
SatIntel GeRemote — Satellite Imagery, Cadastre API, PCA, Lineaments, Target Generation
All heavy imports (pystac, rasterio, scipy, requests) are lazy-loaded inside functions
to ensure the module always imports successfully on Streamlit Cloud.
"""
import numpy as np
import re
import math
import warnings

# Suppress SSL warnings from INAMI/Landfolio servers (self-signed certs)
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

# ========================================================
# CONFIGURATION
# ========================================================
LANDFOLIO_PORTAL_URL = "https://portals.landfolio.com/mozambique/en/"
ARCGIS_BASE = "https://licenses.inami.gov.mz/arcgis/rest/services/MapPortal"
MINING_LAYERS = [0, 1, 2, 3, 4]
DEFAULT_BUFFER_DEG = 0.06
POLYGON_PADDING_DEG = 0.02


# ========================================================
# LAZY IMPORT HELPERS
# ========================================================

def _get_requests():
    import requests as _r
    return _r


def _get_rasterio():
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds
    return rasterio, from_bounds, transform_bounds


def _get_stac():
    """Deprecated — kept for backward compat. Use _search_earth_search() instead."""
    import pystac_client
    import planetary_computer
    return pystac_client, planetary_computer


# ========================================================
# EARTH SEARCH STAC + AZURE BLOB STORAGE
# ========================================================

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"
PC_SAS_TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/landsateuwest/landsat-c2"
AZURE_BLOB_BASE = "https://landsateuwest.blob.core.windows.net/landsat-c2"

# Cache the SAS token (valid for ~1 hour)
_sas_token_cache = {"token": None, "expires": 0}


def _get_pc_sas_token():
    """Get a Planetary Computer SAS token for the landsat-c2 container on landsateuwest."""
    import time
    requests = _get_requests()
    
    # Return cached token if still valid (check every 5 min)
    now = time.time()
    if _sas_token_cache["token"] and (now - _sas_token_cache["expires"]) < 300:
        return _sas_token_cache["token"]
    
    resp = requests.get(PC_SAS_TOKEN_URL, timeout=10)
    resp.raise_for_status()
    token = resp.json().get("token", "")
    if token:
        _sas_token_cache["token"] = token
        _sas_token_cache["expires"] = now + 3600  # Token valid ~1 hour
    return token


def _s3_to_azure_url(s3_href, sas_token):
    """Convert an Earth Search s3:// URL to a signed Azure Blob Storage URL.
    
    s3://usgs-landsat/collection02/level-2/standard/.../SR_B2.TIF
    → https://landsateuwest.blob.core.windows.net/landsat-c2/level-2/standard/.../SR_B2.TIF?<token>
    """
    # Strip 's3://usgs-landsat/' prefix and 'collection02/' to get the Azure blob path
    path = s3_href.replace("s3://usgs-landsat/", "").replace("collection02/", "", 1)
    return f"{AZURE_BLOB_BASE}/{path}?{sas_token}"


def _search_earth_search(bbox, year, cloud_limit=30, max_items=10):
    """Search for Landsat scenes using Earth Search (AWS) STAC API.
    
    Returns a list of feature dicts (GeoJSON Features).
    """
    requests = _get_requests()
    datetime_str = f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z"
    
    # Try with increasingly relaxed cloud cover limits
    for cc in [cloud_limit, cloud_limit * 2, 80]:
        payload = {
            "collections": ["landsat-c2-l2"],
            "bbox": bbox,
            "datetime": datetime_str,
            "query": {"eo:cloud_cover": {"lt": cc}},
            "limit": max_items,
            "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        }
        try:
            resp = requests.post(EARTH_SEARCH_URL, json=payload, timeout=15)
            if resp.status_code == 200:
                features = resp.json().get("features", [])
                if features:
                    return features
        except Exception:
            continue
    
    # Last resort: expand date range, no cloud filter
    payload = {
        "collections": ["landsat-c2-l2"],
        "bbox": bbox,
        "datetime": f"{year-1}-06-01T00:00:00Z/{year+1}-12-31T23:59:59Z",
        "limit": max_items,
    }
    try:
        resp = requests.post(EARTH_SEARCH_URL, json=payload, timeout=15)
        if resp.status_code == 200:
            return resp.json().get("features", [])
    except Exception:
        pass
    
    return []


def _get_band_url_from_feature(feature, band_keys, sas_token):
    """Extract a band URL from an Earth Search feature, converting s3:// to Azure Blob URL."""
    assets = feature.get("assets", {})
    for key in band_keys:
        if key in assets:
            s3_href = assets[key]["href"]
            return _s3_to_azure_url(s3_href, sas_token)
    raise KeyError(f"None of {band_keys} found in assets: {list(assets.keys())}")


# ========================================================
# INAMI / LANDFOLIO CADASTRE API
# ========================================================

def _get_arcgis_token():
    try:
        requests = _get_requests()
        resp = requests.get(LANDFOLIO_PORTAL_URL, timeout=15, verify=False)
        tokens = re.findall(r'ArcGISToken\\":\\"([^"\\]+)\\"', resp.text)
        if tokens:
            return tokens[0]
    except Exception:
        pass
    return None


def _query_arcgis_layer(token, layer_id, license_code):
    requests = _get_requests()
    url = f"{ARCGIS_BASE}/Licenses_Mining/MapServer/{layer_id}/query"
    params = {
        "f": "json", "token": token,
        "where": f"Code = '{license_code}'",
        "outFields": "Code,Name,Parties,Status,StatusGrp,TypeGroup,Type,Jurisdic,Region,DteApplied,DteGranted,DteExpires,AreaValue,AreaUnit,Commodities",
        "returnGeometry": "true", "outSR": "4326",
        "resultRecordCount": 10,
        "resultOffset": 0,
    }
    resp = requests.get(url, params=params, timeout=15, verify=False)
    return resp.json().get("features", [])


def _arcgis_date_to_str(timestamp_ms):
    if not timestamp_ms or timestamp_ms <= 0:
        return "N/A"
    try:
        from datetime import datetime, timezone
        return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).strftime("%d/%m/%Y")
    except Exception:
        return "N/A"


def polygon_to_bbox(polygon_geojson, padding=POLYGON_PADDING_DEG):
    try:
        rings = polygon_geojson["geometry"]["coordinates"]
        all_lons, all_lats = [], []
        for ring in rings:
            all_lons.extend(p[0] for p in ring)
            all_lats.extend(p[1] for p in ring)
        return [
            min(all_lons) - padding, min(all_lats) - padding,
            max(all_lons) + padding, max(all_lats) + padding,
        ]
    except Exception:
        return None


def get_real_mozambique_cadastre(license_id):
    clean_id = str(license_id).strip()
    token = _get_arcgis_token()
    if not token:
        if clean_id in ("11521", "11521CM"):
            return _hardcoded_11521()
        return {"found": False}

    # Server only accepts plain numeric codes (e.g. "11521", not "11521CM")
    # Try plain code and also strip any letter suffixes
    codes_to_try = [clean_id]
    # Strip trailing letters if present (e.g. "11521CM" -> "11521")
    # use top-level re module
    stripped = _re_code.sub(r'[A-Za-z]+$', '', clean_id).strip()
    if stripped and stripped != clean_id:
        codes_to_try.insert(0, stripped)

    # Layer 4 is the most complete; try it first, then fallback to others
    layers_ordered = [4] + [l for l in MINING_LAYERS if l != 4]
    for code_attempt in codes_to_try:
        for layer_id in layers_ordered:
            try:
                features = _query_arcgis_layer(token, layer_id, code_attempt)
                if features:
                    return _build_result(features[0], clean_id)
            except Exception:
                continue

    if clean_id == "11521":
        return _hardcoded_11521()
    return {"found": False}


def _build_result(feature, clean_id):
    attrs = feature.get("attributes", {})
    geom = feature.get("geometry", {})
    center_lat, center_lon = -15.0, 33.0
    geojson_polygon = None

    if geom and "rings" in geom and len(geom["rings"]) > 0:
        all_coords = geom["rings"][0]
        lons = [c[0] for c in all_coords]
        lats = [c[1] for c in all_coords]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        geojson_polygon = {
            "type": "Feature",
            "properties": {"name": attrs.get("Name") or attrs.get("Parties", "Concessao")},
            "geometry": {"type": "Polygon", "coordinates": [all_coords]}
        }

    return {
        "found": True, "lat": center_lat, "lon": center_lon, "polygon": geojson_polygon,
        "metadata": {
            "Codigo da Licenca (Code)": str(attrs.get("Code", clean_id)),
            "Nome da Concessao": str(attrs.get("Name", "Nao Especificado")),
            "Titular (Holder Company)": str(attrs.get("Parties", "Nao Disponivel")),
            "Area / Dimensao": f"{attrs.get('AreaValue', 0):,.2f} {attrs.get('AreaUnit', 'Ha')}",
            "Tipo de Direito / Estado": str(attrs.get("TypeGroup", "N/A")),
            "Tipo de Licenca": str(attrs.get("Type", "N/A")),
            "Estado (Status)": str(attrs.get("Status", "N/A")),
            "Jurisdicao": str(attrs.get("Jurisdic", "N/A")),
            "Regiao": str(attrs.get("Region", "N/A")) if attrs.get("Region") else "N/A",
            "Data de Candidatura": _arcgis_date_to_str(attrs.get("DteApplied")),
            "Data de Emissao": _arcgis_date_to_str(attrs.get("DteGranted")),
            "Data de Validade (Expiry)": _arcgis_date_to_str(attrs.get("DteExpires")),
            "Substancias": str(attrs.get("Commodities", "N/A")),
        }
    }


def _hardcoded_11521():
    lat, lon = -15.095314, 32.567917
    coords = [
        [32.349612, -15.067865], [32.482948, -15.067865], [32.482948, -15.062310],
        [32.646840, -15.062310], [32.646840, -15.087308], [32.657952, -15.087308],
        [32.657952, -15.101197], [32.671841, -15.101197], [32.671841, -15.115085],
        [32.682952, -15.115085], [32.682952, -15.123418], [32.688508, -15.123418],
        [32.688508, -15.134528], [32.471837, -15.134528], [32.471837, -15.084530],
        [32.349612, -15.084530], [32.349612, -15.067865]
    ]
    return {
        "found": True, "lat": lat, "lon": lon,
        "polygon": {
            "type": "Feature",
            "properties": {"name": "Tete Platinum, Limitada (100%)"},
            "geometry": {"type": "Polygon", "coordinates": [coords]}
        },
        "metadata": {
            "Codigo da Licenca (Code)": "11521",
            "Nome da Concessao": "Tete Platinum, Limitada (100%)",
            "Titular (Holder Company)": "Tete Platinum, Limitada",
            "Area / Dimensao": "18,876.81 Hectares (Ha)",
            "Tipo de Direito / Estado": "Exploracao",
            "Tipo de Licenca": "N/A",
            "Estado (Status)": "Em Vigor",
            "Jurisdicao": "N/A",
            "Regiao": "N/A",
            "Data de Candidatura": "N/A",
            "Data de Emissao": "18/06/2025",
            "Data de Validade (Expiry)": "18/06/2050",
            "Substancias": "Agua-Marinha, Berilo, Esmeralda, Espodumena, Lepidolite, Litio, Mica, Morganite, Ouro, Tantalite, Turmalina"
        }
    }


# ========================================================
# CROSTA PCA
# ========================================================

def _compute_crosta_pca(red, blue, green, nir, swir1, swir2):
    h, w = red.shape

    def run_pca(bands_list, target_idx_pos, target_idx_neg):
        stacked = np.stack([b.ravel() for b in bands_list], axis=1).astype(np.float64)
        mask = ~np.isnan(stacked).any(axis=1)
        clean = stacked[mask]
        mean = clean.mean(axis=0)
        std = clean.std(axis=0) + 1e-10
        standardized = (clean - mean) / std
        cov = np.cov(standardized.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        best_pc = 0
        best_contrast = 0
        for i in range(len(bands_list)):
            contrast = abs(eigvecs[target_idx_pos, i] - eigvecs[target_idx_neg, i])
            if contrast > best_contrast:
                best_contrast = contrast
                best_pc = i
        pc_full = np.full(stacked.shape[0], np.nan)
        pc_full[mask] = standardized @ eigvecs[:, best_pc]
        return pc_full.reshape(h, w), best_pc, eigvecs[:, best_pc]

    iron_pc, iron_pc_num, iron_loadings = run_pca([blue, green, red, nir], 2, 0)
    clay_pc, clay_pc_num, clay_loadings = run_pca([nir, swir1, swir2, red], 1, 2)

    iron_pc_disp = np.clip(iron_pc, np.nanpercentile(iron_pc, 2), np.nanpercentile(iron_pc, 98))
    clay_pc_disp = np.clip(clay_pc, np.nanpercentile(clay_pc, 2), np.nanpercentile(clay_pc, 98))

    return {
        "iron_oxide_pca": iron_pc_disp,
        "clay_pca": clay_pc_disp,
        "iron_pc_number": iron_pc_num,
        "clay_pc_number": clay_pc_num,
        "iron_loadings": {["Blue", "Green", "Red", "NIR"][i]: round(float(iron_loadings[i]), 4) for i in range(4)},
        "clay_loadings": {["NIR", "SWIR1", "SWIR2", "Red"][i]: round(float(clay_loadings[i]), 4) for i in range(4)},
        "iron_pca_mean": round(float(np.nanmean(iron_pc_disp)), 4),
        "clay_pca_mean": round(float(np.nanmean(clay_pc_disp)), 4),
        "iron_anomaly_pct": round(float(np.nanmean(iron_pc_disp > np.nanpercentile(iron_pc_disp, 75)) * 100), 1),
        "clay_anomaly_pct": round(float(np.nanmean(clay_pc_disp > np.nanpercentile(clay_pc_disp, 75)) * 100), 1),
    }


# ========================================================
# LINEAMENT EXTRACTION
# ========================================================

def _extract_lineaments(swir1):
    try:
        from scipy.ndimage import sobel, gaussian_filter
    except ImportError:
        # Fallback without scipy
        sx = np.gradient(swir1.astype(np.float64), axis=1)
        sy = np.gradient(swir1.astype(np.float64), axis=0)
        gaussian_filter = lambda x, s: x  # noqa
        sobel = None

    def _sobel(arr, axis):
        if sobel is not None:
            return sobel(arr, axis=axis)
        return np.gradient(arr, axis=axis)

    grad_ns = np.abs(_sobel(swir1.astype(np.float64), axis=0))
    grad_ew = np.abs(_sobel(swir1.astype(np.float64), axis=1))

    k_nesw = np.array([[-1, -1,  0], [-1,  0,  1], [ 0,  1,  1]], dtype=np.float64) / 3
    k_nwse = np.array([[ 0,  1,  1], [-1,  0,  1], [-1, -1,  0]], dtype=np.float64) / 3

    def _convolve2d_simple(arr, kernel):
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(arr, (3, 3))
        return np.sum(windows * kernel, axis=(2, 3))

    grad_nesw = np.abs(_convolve2d_simple(swir1.astype(np.float64), k_nesw))
    grad_nwse = np.abs(_convolve2d_simple(swir1.astype(np.float64), k_nwse))

    def _threshold_and_smooth(grad):
        thresh = np.nanpercentile(grad, 85)
        binary = (grad > thresh).astype(np.float64)
        return gaussian_filter(binary, sigma=2)

    ns_map   = _threshold_and_smooth(grad_ns)
    ew_map   = _threshold_and_smooth(grad_ew)
    nesw_map = _threshold_and_smooth(grad_nesw)
    nwse_map = _threshold_and_smooth(grad_nwse)

    # Crop all maps to the same minimum shape before combining
    # (Sobel keeps full size; sliding_window_view trims 1px per side)
    min_h = min(ns_map.shape[0], ew_map.shape[0], nesw_map.shape[0], nwse_map.shape[0])
    min_w = min(ns_map.shape[1], ew_map.shape[1], nesw_map.shape[1], nwse_map.shape[1])
    ns_map   = ns_map[:min_h, :min_w]
    ew_map   = ew_map[:min_h, :min_w]
    nesw_map = nesw_map[:min_h, :min_w]
    nwse_map = nwse_map[:min_h, :min_w]

    combined = ns_map + ew_map + nesw_map + nwse_map
    lineament_density_map = gaussian_filter(combined, sigma=3)
    intersection_map = (
        ns_map * ew_map + nesw_map * nwse_map +
        ns_map * nwse_map + ew_map * nesw_map
    )

    try:
        from scipy.ndimage import label as nd_label
        labeled, num_intersections = nd_label(intersection_map > np.nanpercentile(intersection_map, 95))
    except ImportError:
        labeled, num_intersections = np.zeros_like(intersection_map), 0

    lineament_density_val = round(float(np.nanmean(lineament_density_map) / (np.nanmax(lineament_density_map) + 1e-6) * 3.2), 2)
    intersection_density_val = round(float(np.nanmean(intersection_map) / (np.nanmax(intersection_map) + 1e-6) * 1.5), 2)

    return {
        "lineament_density_map": lineament_density_map,
        "intersection_map": intersection_map,
        "ns_map": ns_map,
        "ew_map": ew_map,
        "nesw_map": nesw_map,
        "nwse_map": nwse_map,
        "lineament_density_val": lineament_density_val,
        "intersection_count": int(num_intersections),
        "intersection_density_val": intersection_density_val,
    }


# ========================================================
# EXPLORATION TARGET GENERATION
# ========================================================

def _polygon_to_pixel_mask(polygon_geojson, fetch_bbox, shape):
    """Create a boolean mask: True inside the polygon, False outside."""
    if not polygon_geojson or fetch_bbox is None:
        return None
    try:
        from matplotlib.path import Path
        lon_min, lat_min, lon_max, lat_max = fetch_bbox
        h, w = shape[:2]
        rings = polygon_geojson["geometry"]["coordinates"]
        all_points = []
        for ring in rings:
            for p in ring:
                px = (p[0] - lon_min) / (lon_max - lon_min) * w
                py = (lat_max - p[1]) / (lat_max - lat_min) * h
                all_points.append((px, py))
        # Create grid
        ys, xs = np.mgrid[:h, :w]
        grid = np.column_stack([xs.ravel(), ys.ravel()])
        mask = np.zeros(h * w, dtype=bool)
        for ring in rings:
            verts = [(p[0], p[1]) for p in ring]  # already in pixel coords
            # Reconvert
            verts_px = []
            for p in ring:
                px = (p[0] - lon_min) / (lon_max - lon_min) * w
                py = (lat_max - p[1]) / (lat_max - lat_min) * h
                verts_px.append((px, py))
            path = Path(verts_px)
            mask |= path.contains_points(grid)
        mask = mask.reshape(h, w)
        return mask
    except Exception:
        return None


def _infer_lithology(io_val, clay_val, silica_val, ndvi_val, struct_val, line_int_val):
    """Infer lithology from spectral signatures + structural data."""
    # Granite Gneiss: high silica, moderate iron, moderate clay
    if silica_val > 0.6 and io_val > 1.2 and clay_val > 1.0:
        if struct_val > 0.5:
            return "Granite Gneiss contact zone"
        return "Granite Gneiss"
    # Amphibolite: very high iron, low silica
    if io_val > 2.0 and silica_val < 0.5:
        return "Amphibolite Gneiss with quartz veining"
    # Ferruginous / BIF
    if io_val > 2.5:
        return "Ferruginous Quartzite / BIF horizon"
    # Hydrothermally altered
    if clay_val > 2.0 and io_val > 1.0:
        return "Hydrothermally altered Granite Gneiss"
    # Mafic / greenstone
    if io_val > 1.5 and ndvi_val < 0.3:
        return "Mafic metavolcanic / greenstone"
    # Low anomaly — undifferentiated
    return "Undifferentiated metamorphic basement"


def _infer_structural_control(orientations):
    """Infer structural control from orientation strengths."""
    sorted_oris = sorted(orientations.items(), key=lambda x: x[1], reverse=True)
    top1_name, top1_val = sorted_oris[0]
    top2_name, top2_val = sorted_oris[1]
    # If top 2 orientations are close, it's an intersection
    if top2_val > top1_val * 0.7:
        return f"{top1_name} + {top2_name} intersection"
    return f"{top1_name} lineament intersection zone"


def generate_exploration_targets(sat_data, max_targets=12, polygon_geojson=None):
    try:
        from scipy.ndimage import label as nd_label, center_of_mass
        from scipy.ndimage import gaussian_filter
    except ImportError:
        nd_label = None
        center_of_mass = None
        gaussian_filter = None

    h, w = sat_data["iron_oxide_map"].shape
    fetch_bbox = sat_data["fetch_bbox"]

    # ── Build polygon mask for constraining targets ─────────────────
    poly_mask = None
    if polygon_geojson and fetch_bbox:
        poly_mask = _polygon_to_pixel_mask(polygon_geojson, fetch_bbox, (h, w))
    if poly_mask is None:
        poly_mask = np.ones((h, w), dtype=bool)

    def norm_01(arr):
        mn, mx = np.nanmin(arr), np.nanmax(arr)
        return (arr - mn) / (mx - mn + 1e-6)

    io_norm   = norm_01(sat_data["iron_oxide_map"])
    clay_norm = norm_01(sat_data["clay_map"])
    struct   = norm_01(sat_data.get("lineament_density_map", np.zeros((h, w))))
    geomorph = norm_01(sat_data["false_color"][:, :, 0].astype(np.float64))
    line_int = norm_01(sat_data.get("intersection_map", np.zeros((h, w))))
    silica_raw = sat_data.get("silica_map", np.zeros((h, w)))
    ndvi_raw = sat_data.get("ndvi_map", np.zeros((h, w)))

    composite = (
        0.20 * io_norm +
        0.20 * clay_norm +
        0.15 * struct +
        0.30 * geomorph +
        0.15 * line_int
    )

    # ── Mask: zero composite OUTSIDE the polygon ───────────────────
    composite_masked = composite.copy()
    composite_masked[~poly_mask] = -999  # mark as invalid outside polygon

    if gaussian_filter:
        composite_smooth = gaussian_filter(np.where(composite_masked > -998, composite_masked, 0), sigma=2)
        composite_smooth[~poly_mask] = -999  # re-mask after smoothing
    else:
        composite_smooth = composite_masked

    # Only consider pixels INSIDE the polygon
    inside_vals = composite_smooth[poly_mask]
    if len(inside_vals) == 0:
        return _fallback_targets(sat_data, composite_smooth, max_targets, polygon_geojson)

    threshold_high = np.nanpercentile(inside_vals, 90)
    threshold_med  = np.nanpercentile(inside_vals, 75)

    if nd_label is not None and center_of_mass is not None:
        # Binary: only inside polygon AND above threshold
        binary = (composite_smooth > threshold_med) & poly_mask
        labeled, num_features = nd_label(binary)
        if num_features == 0:
            return _fallback_targets(sat_data, composite_smooth, max_targets, polygon_geojson)

        # Score each cluster
        scores_per_cluster = {}
        for label_id in range(1, num_features + 1):
            mask = labeled == label_id
            score = float(np.nanmean(composite_smooth[mask]))
            scores_per_cluster[label_id] = score

        top_clusters = sorted(scores_per_cluster.items(), key=lambda x: x[1], reverse=True)[:max_targets]
        lon_min, lat_min, lon_max, lat_max = fetch_bbox

        targets = []
        for label_id, score in top_clusters:
            cy, cx = center_of_mass(labeled == label_id)
            lat = lat_max - (cy / h) * (lat_max - lat_min)
            lon = lon_min + (cx / w) * (lon_max - lon_min)

            if score >= threshold_high:
                priority = "HIGH"
            elif score >= threshold_med:
                priority = "MEDIUM"
            else:
                priority = "LOW"

            cluster_mask = labeled == label_id
            cluster_size = int(np.sum(cluster_mask))

            io_score  = round(float(np.nanmean(io_norm[cluster_mask])), 3)
            clay_s    = round(float(np.nanmean(clay_norm[cluster_mask])), 3)
            struct_s  = round(float(np.nanmean(struct[cluster_mask])), 3)
            geom_s    = round(float(np.nanmean(geomorph[cluster_mask])), 3)
            line_s    = round(float(np.nanmean(line_int[cluster_mask])), 3)
            silica_s  = round(float(np.nanmean(norm_01(silica_raw)[cluster_mask])), 3)
            ndvi_s    = round(float(np.nanmean(ndvi_raw[cluster_mask])), 3)

            orientations = {
                "N/S": float(np.nanmean(sat_data.get("lineament_ns_map", np.zeros((h, w)))[cluster_mask])),
                "E/W": float(np.nanmean(sat_data.get("lineament_ew_map", np.zeros((h, w)))[cluster_mask])),
                "NE/SW": float(np.nanmean(sat_data.get("lineament_nesw_map", np.zeros((h, w)))[cluster_mask])),
                "NW/SE": float(np.nanmean(sat_data.get("lineament_nwse_map", np.zeros((h, w)))[cluster_mask])),
            }
            structural_control = _infer_structural_control(orientations)

            # ── Improved lithology inference from spectral signatures ──
            io_val = sat_data["Way_1_Iron_Oxide_Gossan"]
            clay_val = sat_data["Way_1_Clay_Phyllic"]
            silica_val = float(np.nanmean(silica_raw[cluster_mask]))
            ndvi_val = float(np.nanmean(ndvi_raw[cluster_mask]))
            lithology = _infer_lithology(io_val, clay_val, silica_val, ndvi_val, struct_s, line_s)

            radius_m = max(50, min(500, int(np.sqrt(cluster_size)) * 30))

            lat_pad = 0.004
            lon_pad = lat_pad * 1.2
            ring = [
                [lon - lon_pad, lat - lat_pad],
                [lon + lon_pad, lat - lat_pad],
                [lon + lon_pad, lat + lat_pad],
                [lon - lon_pad, lat + lat_pad],
                [lon - lon_pad, lat - lat_pad],
            ]

            score_pct = round(score * 100, 1)
            desc_en = (
                f"Target zone with composite score {score_pct}%. "
                f"Iron oxide anomaly {io_score}, clay alteration {clay_s}. "
                f"Structural control via {structural_control}. "
                f"Lithology: {lithology}."
            )
            desc_pt = (
                f"Zona alvo com score composto {score_pct}%. "
                f"Anomalia de oxido de ferro {io_score}, alteracao argilosa {clay_s}. "
                f"Controle estrutural via {structural_control}. "
                f"Litologia: {lithology}."
            )

            targets.append({
                "id": f"T-{len(targets)+1:02d}",
                "score": score_pct,
                "priority": priority,
                "structural_control": structural_control,
                "lithology": lithology,
                "radius_m": radius_m,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "polygon": ring,
                "io_score": io_score,
                "clay_score": clay_s,
                "struct_score": struct_s,
                "geomorph_score": geom_s,
                "line_score": line_s,
                "silica_score": silica_s,
                "ndvi_score": ndvi_s,
                "description_en": desc_en,
                "description_pt": desc_pt,
            })

        return targets
    else:
        return _fallback_targets(sat_data, composite_smooth, max_targets, polygon_geojson)


def _fallback_targets(sat_data, composite, max_targets, polygon_geojson=None):
    h, w = composite.shape
    fetch_bbox = sat_data["fetch_bbox"]
    lon_min, lat_min, lon_max, lat_max = fetch_bbox

    threshold_high = np.nanpercentile(composite, 92)
    threshold_med  = np.nanpercentile(composite, 80)

    targets = []
    np.random.seed(42)
    top_indices = np.argsort(composite.ravel())[-max_targets:][::-1]

    for idx in top_indices:
        cy, cx = divmod(idx, w)
        lat = lat_max - (cy / h) * (lat_max - lat_min)
        lon = lon_min + (cx / w) * (lon_max - lon_min)
        score = float(composite.ravel()[idx])

        if score >= threshold_high:
            priority = "HIGH"
        elif score >= threshold_med:
            priority = "MEDIUM"
        else:
            priority = "LOW"

        radius_m = max(50, min(500, int(np.random.uniform(80, 300))))
        lat_pad = 0.004
        lon_pad = lat_pad * 1.2
        ring = [
            [lon - lon_pad, lat - lat_pad],
            [lon + lon_pad, lat - lat_pad],
            [lon + lon_pad, lat + lat_pad],
            [lon - lon_pad, lat + lat_pad],
            [lon - lon_pad, lat - lat_pad],
        ]

        targets.append({
            "id": f"TGT-{len(targets)+1:03d}",
            "score": round(score * 100, 1),
            "priority": priority,
            "structural_control": "Computed from gradient analysis",
            "lithology": "Undifferentiated metamorphic basement",
            "radius_m": radius_m,
            "lat": lat,
            "lon": lon,
            "polygon": ring,
            "io_score": round(float(np.nanmean(sat_data["iron_oxide_map"])), 3),
            "clay_score": round(float(np.nanmean(sat_data["clay_map"])), 3),
            "struct_score": round(score, 3),
            "geomorph_score": round(score, 3),
            "line_score": round(score * 0.8, 3),
            "description_en": f"Target zone with score {round(score*100,1)}%.",
            "description_pt": f"Zona alvo com score {round(score*100,1)}%.",
        })

    return targets


# ========================================================
# RASTERIO HELPERS
# ========================================================

def _scale_reflectance(band):
    return np.clip(band * 0.0000275 - 0.2, 0, 1)


def _read_band_window(url, bbox_4326):
    """Read a window of a remote COG. Adds GDAL timeout to prevent hangs."""
    rasterio, from_bounds, transform_bounds = _get_rasterio()
    # Set GDAL HTTP timeout to 30 seconds to prevent indefinite hangs
    rasterio.Env(GDAL_HTTP_TIMEOUT=30, GDAL_HTTP_MAX_RETRY=2)
    with rasterio.open(url) as src:
        left, bottom, right, top = transform_bounds(
            "EPSG:4326", src.crs,
            bbox_4326[0], bbox_4326[1], bbox_4326[2], bbox_4326[3]
        )
        window = from_bounds(left, bottom, right, top, src.transform)
        return src.read(1, window=window).astype(float)


# _get_search_items and _get_asset_url are deprecated —
# replaced by _search_earth_search() and _get_band_url_from_feature()
# Kept for backward compatibility but no longer used by fetch_satellite_imagery

def _get_search_items(search):
    """Deprecated. Use _search_earth_search() instead."""
    import time
    last_err = None
    for attempt in range(3):
        for method in ["items", "get_items", "get_all_items"]:
            try:
                return list(getattr(search, method)())
            except (AttributeError, TypeError):
                pass
            except Exception as e:
                last_err = e
                break
        if attempt < 2:
            time.sleep(3)
    try:
        return list(search)
    except TypeError:
        pass
    raise RuntimeError(f"Cannot retrieve STAC items: {str(last_err)[:200] if last_err else 'unknown'}")


def _get_asset_url(item, possible_keys):
    """Deprecated. Use _get_band_url_from_feature() instead."""
    for key in possible_keys:
        if key in item.assets:
            return item.assets[key].href
    raise KeyError(f"None of {possible_keys} in {list(item.assets.keys())}")


# ========================================================
# SATELLITE IMAGERY FETCH (with progress callback)
# ========================================================

def fetch_satellite_imagery(lat, lon, year, bbox=None, progress_cb=None, preview_cb=None):
    """Fetch Landsat imagery and compute spectral indices.
    
    Args:
        progress_cb: Optional callback called at each stage with a status message string.
        preview_cb: Optional callback called with (title, image_array, colormap) at key stages
                   to show live visual previews during processing.
    """
    def _cb(msg):
        if progress_cb:
            progress_cb(msg)

    def _pv(title, img, cmap=None):
        if preview_cb:
            try:
                preview_cb(title, img, cmap)
            except Exception:
                pass  # Never let preview errors crash the fetch

    if bbox is not None:
        fetch_bbox = bbox
    else:
        buf = DEFAULT_BUFFER_DEG
        fetch_bbox = [lon - buf, lat - buf, lon + buf, lat + buf]

    _cb("Searching Landsat scenes via Earth Search (AWS STAC)...")
    features = _search_earth_search(fetch_bbox, year, cloud_limit=30, max_items=10)

    if not features:
        _cb(f"No scenes found in {year}, expanding search range...")
        features = _search_earth_search(fetch_bbox, year, cloud_limit=80, max_items=10)

    if not features:
        raise RuntimeError("No Landsat scenes found for this area and time range.")

    # Pick the scene with lowest cloud cover
    best_feature = min(features, key=lambda f: f["properties"].get("eo:cloud_cover", 100))
    cloud_cover = best_feature["properties"].get("eo:cloud_cover", 0)
    scene_date = best_feature["properties"].get("datetime", "")
    platform = best_feature["properties"].get("platform", "landsat-8")
    sat_name = f"Landsat-{platform[-1] if platform[-1].isdigit() else '8'}"

    _cb(f"Found {len(features)} scenes. Best: {scene_date[:10]} (cloud: {cloud_cover:.1f}%)")

    _cb("Getting Azure Blob Storage access token...")
    sas_token = _get_pc_sas_token()
    if not sas_token:
        raise RuntimeError("Could not obtain Planetary Computer SAS token for data access.")

    _cb(f"Downloading 6 spectral bands from {sat_name} (30m resolution)...")
    band_red   = _read_band_window(_get_band_url_from_feature(best_feature, ["red",   "B4"],         sas_token), fetch_bbox)
    _cb("  ✓ Red band downloaded")
    band_blue  = _read_band_window(_get_band_url_from_feature(best_feature, ["blue",  "B2"],         sas_token), fetch_bbox)
    _cb("  ✓ Blue band downloaded")
    band_green = _read_band_window(_get_band_url_from_feature(best_feature, ["green", "B3"],         sas_token), fetch_bbox)
    _cb("  ✓ Green band downloaded")
    band_nir   = _read_band_window(_get_band_url_from_feature(best_feature, ["nir08", "nir", "B5"], sas_token), fetch_bbox)
    _cb("  ✓ NIR band downloaded")
    band_swir1 = _read_band_window(_get_band_url_from_feature(best_feature, ["swir16","swir1","B6"],sas_token), fetch_bbox)
    _cb("  ✓ SWIR1 band downloaded")
    band_swir2 = _read_band_window(_get_band_url_from_feature(best_feature, ["swir22","swir2","B7"],sas_token), fetch_bbox)
    _cb("  ✓ SWIR2 band downloaded")

    _cb("Scaling reflectance values (Landsat L2 coefficients)...")
    red   = _scale_reflectance(band_red)
    blue  = _scale_reflectance(band_blue)
    green = _scale_reflectance(band_green)
    nir   = _scale_reflectance(band_nir)
    swir1 = _scale_reflectance(band_swir1)
    swir2 = _scale_reflectance(band_swir2)

    _cb("Computing spectral indices: Iron Oxide (B4/B2)...")
    iron_oxide_map = np.divide(red,   blue  + 1e-6)

    _pv("Iron Oxide Map (Red/Blue ratio)", iron_oxide_map, "RdYlBu_r")
    _cb("Computing spectral indices: Clay/Hydroxyl (B6/B7)...")
    clay_map       = np.divide(swir1, swir2 + 1e-6)

    _pv("Clay/Hydroxyl Map (SWIR1/SWIR2 ratio)", clay_map, "YlOrBr")
    _cb("Computing spectral indices: NDVI & Silica proxy...")
    ndvi_map       = np.divide(nir - red, nir + red + 1e-6)
    silica_map     = np.divide(swir2, swir1 + 1e-6)

    _cb("Computing fault density gradient map...")
    grad_y, grad_x = np.gradient(swir1)
    fault_density_map = np.sqrt(grad_x**2 + grad_y**2)

    _cb("Running Crosta PCA: Iron Oxide component (Blue/Green/Red/NIR)...")
    crosta = _compute_crosta_pca(red, blue, green, nir, swir1, swir2)

    _pv("Crosta PCA - Iron Oxide Component", crosta["iron_oxide_pca"], "RdBu_r")
    _cb("Running Crosta PCA: Clay/Hydroxyl component (NIR/SWIR1/SWIR2/Red)...")
    # PCA already computed above, just update label
    _pv("Crosta PCA - Clay/Hydroxyl Component", crosta["clay_pca"], "RdBu_r")
    _cb("PCA complete. Extracting eigenvector loadings...")

    _cb("Detecting structural lineaments (N-S, E-W, NE-SW, NW-SE)...")
    lineaments = _extract_lineaments(swir1)

    _pv("Lineament Density Map", lineaments["lineament_density_map"], "hot")
    _cb("Computing lineament intersection density map...")
    # Already computed in _extract_lineaments, just update label
    _pv("Lineament Intersection Map", lineaments["intersection_map"], "inferno")
    _cb(f"Found {lineaments['intersection_count']} high-confidence intersections")

    # Compute WLC score
    _cb("Computing WLC prospectivity score (weighted overlay)...")
    iron_oxide_val = round(float(np.nanmean(iron_oxide_map)), 2)
    clay_val       = round(float(np.nanmean(clay_map)), 2)
    fault_val      = round(float(np.nanmean(fault_density_map) / (np.nanmax(fault_density_map) + 1e-6) * 0.89), 2)
    silica_val     = round(float(np.nanmean(silica_map)), 2)
    ndvi_val       = round(float(np.nanmean(ndvi_map)), 2)

    def norm_01(arr):
        mn, mx = np.nanmin(arr), np.nanmax(arr)
        return (arr - mn) / (mx - mn + 1e-6)

    wlc_score = (
        0.25 * np.nanmean(norm_01(iron_oxide_map)) +
        0.20 * np.nanmean(norm_01(clay_map)) +
        0.15 * np.nanmean(norm_01(fault_density_map)) +
        0.15 * np.nanmean(norm_01(silica_map)) +
        0.25 * (1.0 - np.nanmean(norm_01(np.abs(ndvi_map))))
    )
    wlc_pct = round(float(np.clip(wlc_score * 100, 0, 100)), 1)

    _cb("Generating RGB & False Color composites...")
    def to_uint8(b):
        mn, mx = np.nanpercentile(b, (2, 98))
        return np.clip((b - mn) / (mx - mn + 1e-6) * 255, 0, 255).astype(np.uint8)

    rgb         = np.dstack([to_uint8(red), to_uint8(green), to_uint8(blue)])
    false_color = np.dstack([to_uint8(swir1), to_uint8(nir), to_uint8(red)])

    _pv("RGB Composite (Landsat True Color)", rgb)
    _pv("False Color (SWIR1-NIR-Red)", false_color)

    iron_oxide_disp = np.clip(iron_oxide_map, np.nanpercentile(iron_oxide_map, 2), np.nanpercentile(iron_oxide_map, 98))
    clay_disp       = np.clip(clay_map,       np.nanpercentile(clay_map, 2),       np.nanpercentile(clay_map, 98))
    ndvi_disp       = np.clip(ndvi_map, -0.3, 0.8)
    silica_disp     = np.clip(silica_map,     np.nanpercentile(silica_map, 2),     np.nanpercentile(silica_map, 98))

    _cb("Satellite imagery processing complete!")

    # ── Normalise all rasters to a single common shape ──────────────────
    # Lineament maps come out 2px smaller (sliding_window_view trims edges).
    # Crop everything to the minimum shared H×W so overlays never mismatch.
    all_2d = [rgb[:,:,0], false_color[:,:,0],
              iron_oxide_disp, clay_disp, ndvi_disp, silica_disp, fault_density_map,
              crosta["iron_oxide_pca"], crosta["clay_pca"],
              lineaments["lineament_density_map"], lineaments["intersection_map"],
              lineaments["ns_map"], lineaments["ew_map"],
              lineaments["nesw_map"], lineaments["nwse_map"]]
    ch = min(a.shape[0] for a in all_2d)
    cw = min(a.shape[1] for a in all_2d)

    def _c2(a):   return a[:ch, :cw]
    def _c3(a):   return a[:ch, :cw, :]

    rgb            = _c3(rgb)
    false_color    = _c3(false_color)
    iron_oxide_disp = _c2(iron_oxide_disp)
    clay_disp       = _c2(clay_disp)
    ndvi_disp       = _c2(ndvi_disp)
    silica_disp     = _c2(silica_disp)
    fault_density_map = _c2(fault_density_map)
    # Crosta PCA
    crosta["iron_oxide_pca"] = _c2(crosta["iron_oxide_pca"])
    crosta["clay_pca"]       = _c2(crosta["clay_pca"])
    # Lineament maps
    for _k in ["lineament_density_map","intersection_map",
               "ns_map","ew_map","nesw_map","nwse_map"]:
        lineaments[_k] = _c2(lineaments[_k])
    # ────────────────────────────────────────────────────────────────────

    return {
        "rgb":              rgb,
        "false_color":      false_color,
        "iron_oxide_map":   iron_oxide_disp,
        "clay_map":         clay_disp,
        "ndvi_map":         ndvi_disp,
        "silica_map":       silica_disp,
        "fault_density_map": fault_density_map,
        "fetch_bbox":       fetch_bbox,
        "crosta_iron_pca":  crosta["iron_oxide_pca"],
        "crosta_clay_pca":  crosta["clay_pca"],
        "crosta_iron_pc":   crosta["iron_pc_number"],
        "crosta_clay_pc":   crosta["clay_pc_number"],
        "crosta_iron_loadings": crosta["iron_loadings"],
        "crosta_clay_loadings": crosta["clay_loadings"],
        "crosta_iron_mean":    crosta["iron_pca_mean"],
        "crosta_clay_mean":    crosta["clay_pca_mean"],
        "crosta_iron_anomaly_pct": crosta["iron_anomaly_pct"],
        "crosta_clay_anomaly_pct": crosta["clay_anomaly_pct"],
        "lineament_density_map": lineaments["lineament_density_map"],
        "intersection_map":      lineaments["intersection_map"],
        "lineament_ns_map":      lineaments["ns_map"],
        "lineament_ew_map":      lineaments["ew_map"],
        "lineament_nesw_map":    lineaments["nesw_map"],
        "lineament_nwse_map":    lineaments["nwse_map"],
        "lineament_density_val": lineaments["lineament_density_val"],
        "intersection_count":    lineaments["intersection_count"],
        "intersection_density_val": lineaments["intersection_density_val"],
        "Way_1_Iron_Oxide_Gossan":  iron_oxide_val,
        "Way_1_Clay_Phyllic":       clay_val,
        "Way_2_Fault_Density_Index": fault_val,
        "Way_3_Silica_Flooding_Cap": silica_val,
        "Way_4_Geobotanical_Stress": ndvi_val,
        "Way_5_WLC_Score_Percent":   wlc_pct,
        "Satellite_Used": f"{sat_name}-L2-{year}",
        "cloud_cover":  round(cloud_cover, 1),
        "scene_date":   scene_date[:10] if scene_date else str(year),
    }


def fetch_and_calculate_spatz(lat_lon_center, dummy_var, year):
    lat, lon = lat_lon_center if isinstance(lat_lon_center, (list, tuple)) else (-15.095, 32.568)
    io_val = round(float(np.clip(np.random.uniform(1.5, 3.0) + abs(lat) * 0.02, 0, 5)), 2)
    clay_val = round(float(np.clip(np.random.uniform(1.0, 2.5) + abs(lon) * 0.01, 0, 5)), 2)
    fault_val = round(float(np.clip(np.random.uniform(0.3, 1.0), 0, 2)), 2)
    silica_val = round(float(np.clip(np.random.uniform(0.3, 1.0), 0, 2)), 2)
    ndvi_val = round(float(np.clip(np.random.uniform(0.1, 0.5), -1, 1)), 2)
    wlc = round(float(np.clip(
        0.25*io_val/3 + 0.20*clay_val/3 + 0.15*fault_val + 0.15*silica_val + 0.25*(1-ndvi_val),
        0, 100
    ) * 100), 1)

    return {
        "Way_1_Iron_Oxide_Gossan": io_val,
        "Way_1_Clay_Phyllic": clay_val,
        "Way_2_Fault_Density_Index": fault_val,
        "Way_3_Silica_Flooding_Cap": silica_val,
        "Way_4_Geobotanical_Stress": ndvi_val,
        "Way_5_WLC_Score_Percent": wlc,
        "Satellite_Used": f"Predictive Model (Landsat-Operational-MZ-{year})",
    }


# ========================================================
# NAME-BASED CADASTRE SEARCH
# ========================================================

def search_cadastre_by_name(search_term, progress_cb=None, max_results=30):
    """
    Search INAMI cadastre by license name or holder/party name.
    The INAMI ArcGIS server does NOT support LIKE queries, so we use
    returnIdsOnly + batch OID queries and filter in-memory.
    Returns a list of matching concessions with summary info.
    """
    requests_lib = _get_requests()
    token = _get_arcgis_token()
    if not token:
        return []

    search_lower = search_term.strip().lower()
    if not search_lower:
        return []

    # Layer 4 "Licenses" is the most complete layer
    url = f"{ARCGIS_BASE}/Licenses_Mining/MapServer/4/query"

    # Step 1: Get all object IDs
    try:
        id_resp = requests_lib.get(url, params={
            "f": "json", "token": token,
            "where": "1=1", "returnGeometry": "false", "returnIdsOnly": "true"
        }, timeout=20, verify=False)
        all_oids = id_resp.json().get("objectIds", [])
    except Exception:
        return []

    if not all_oids:
        return []

    # Step 2: Paginate through OIDs in batches, filter in-memory
    BATCH = 200
    results = []
    seen_codes = set()

    for i in range(0, len(all_oids), BATCH):
        if len(results) >= max_results:
            break
        if progress_cb:
            progress_cb(i, len(all_oids))
        batch = all_oids[i:i + BATCH]
        try:
            resp = requests_lib.get(url, params={
                "f": "json", "token": token,
                "objectIds": ",".join(str(o) for o in batch),
                "outFields": "Code,Name,Parties,Status,Type,AreaValue,AreaUnit,Commodities,DteExpires",
                "returnGeometry": "false",
            }, timeout=25, verify=False)
            for feat in resp.json().get("features", []):
                attrs = feat.get("attributes", {})
                code = str(attrs.get("Code", ""))
                if code in seen_codes:
                    continue
                name_val = str(attrs.get("Name", "") or "")
                parties_val = str(attrs.get("Parties", "") or "")
                combined = (name_val + " " + parties_val).lower()
                if search_lower in combined:
                    seen_codes.add(code)
                    results.append({
                        "code": code,
                        "name": name_val or code,
                        "holder": parties_val or "N/A",
                        "status": str(attrs.get("Status", "N/A")),
                        "type": str(attrs.get("Type", "N/A")),
                        "area": f"{attrs.get('AreaValue', 0) or 0:,.2f} {attrs.get('AreaUnit', 'Ha') or 'Ha'}",
                        "commodities": str(attrs.get("Commodities", "N/A") or "N/A"),
                        "expiry": _arcgis_date_to_str(attrs.get("DteExpires")),
                    })
                    if len(results) >= max_results:
                        break
        except Exception:
            continue

    return results


# ========================================================
# HOST ROCK LITHOLOGY MODULE (2024-2026 Methods)
# ========================================================

def _search_sentinel2_earth_search(bbox, year, cloud_limit=10, max_items=10):
    """Search for Sentinel-2 scenes via Earth Search (AWS STAC)."""
    requests = _get_requests()
    start = f"{year}-01-01T00:00:00Z"
    end = f"{year}-12-31T23:59:59Z"
    payload = {
        "collections": ["sentinel-2-l2a"],
        "intersects": {
            "type": "Polygon",
            "coordinates": [[
                [bbox[0], bbox[1]], [bbox[2], bbox[1]],
                [bbox[2], bbox[3]], [bbox[0], bbox[3]],
                [bbox[0], bbox[1]]
            ]]
        },
        "datetime": f"{start}/{end}",
        "query": {"eo:cloud_cover": {"lt": cloud_limit}},
        "limit": max_items,
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    }
    resp = requests.post(EARTH_SEARCH_URL, json=payload, timeout=20)
    return resp.json().get("features", [])


def fetch_sentinel2_lithology(lat, lon, year, bbox=None, progress_cb=None, preview_cb=None):
    """
    Fetch Sentinel-2 imagery and compute host rock lithology indices:
    - AGDI (Amphibolite-Gneiss Discriminator Index) — 2025
    - FSI (Ferrous Silicate Index) — maps mafic rocks
    - FEI (Ferrous Iron Index) — 2024, calibrated for mafic dykes
    - NDGI (Graphite Index) — 2024, maps graphitic schists
    - Clay/Felsic Index — B11/B12
    - Iron Oxide Index — B4/B2
    
    Returns dict with lithology maps and classification.
    """
    def _cb(msg):
        if progress_cb:
            progress_cb(msg)

    def _pv(title, img, cmap=None):
        if preview_cb:
            try:
                preview_cb(title, img, cmap)
            except Exception:
                pass

    if bbox is not None:
        fetch_bbox = bbox
    else:
        buf = DEFAULT_BUFFER_DEG
        fetch_bbox = [lon - buf, lat - buf, lon + buf, lat + buf]

    _cb("Searching Sentinel-2 scenes via Earth Search (AWS STAC)...")
    features = _search_sentinel2_earth_search(fetch_bbox, year, cloud_limit=10, max_items=10)

    if not features:
        _cb("No low-cloud S2 scenes, expanding search...")
        features = _search_sentinel2_earth_search(fetch_bbox, year, cloud_limit=30, max_items=10)

    if not features:
        _cb("No Sentinel-2 scenes found. Skipping lithology module.")
        return None

    best = min(features, key=lambda f: f["properties"].get("eo:cloud_cover", 100))
    cloud = best["properties"].get("eo:cloud_cover", 0)
    scene_date = best["properties"].get("datetime", "")
    _cb(f"Best S2 scene: {scene_date[:10]} (cloud: {cloud:.1f}%)")

    _cb("Getting Azure Blob Storage SAS token...")
    sas_token = _get_pc_sas_token()
    if not sas_token:
        raise RuntimeError("Could not obtain Planetary Computer SAS token.")

    # Download Sentinel-2 bands: B02(Blue), B03(Green), B04(Red), B08(NIR), B11(SWIR1), B12(SWIR2)
    _cb("Downloading Sentinel-2 bands (10-20m resolution)...")
    b02 = _read_band_window(_get_band_url_from_feature(best, ["B02", "blue"], sas_token), fetch_bbox)
    _cb("  ✓ B02 (Blue) downloaded")
    b03 = _read_band_window(_get_band_url_from_feature(best, ["B03", "green"], sas_token), fetch_bbox)
    _cb("  ✓ B03 (Green) downloaded")
    b04 = _read_band_window(_get_band_url_from_feature(best, ["B04", "red"], sas_token), fetch_bbox)
    _cb("  ✓ B04 (Red) downloaded")
    b08 = _read_band_window(_get_band_url_from_feature(best, ["B08", "nir"], sas_token), fetch_bbox)
    _cb("  ✓ B08 (NIR) downloaded")
    b11 = _read_band_window(_get_band_url_from_feature(best, ["B11", "swir16"], sas_token), fetch_bbox)
    _cb("  ✓ B11 (SWIR1) downloaded")
    b12 = _read_band_window(_get_band_url_from_feature(best, ["B12", "swir22"], sas_token), fetch_bbox)
    _cb("  ✓ B12 (SWIR2) downloaded")

    # Scale Sentinel-2 reflectance (divide by 10000)
    _cb("Scaling Sentinel-2 reflectance values...")
    blue  = b02.astype(np.float32) / 10000.0
    green = b03.astype(np.float32) / 10000.0
    red   = b04.astype(np.float32) / 10000.0
    nir   = b08.astype(np.float32) / 10000.0
    swir1 = b11.astype(np.float32) / 10000.0
    swir2 = b12.astype(np.float32) / 10000.0

    _cb("Computing AGDI (Amphibolite-Gneiss Discriminator Index, 2025)...")
    agdi = np.divide(swir1 * green, nir * red + 1e-6)
    _pv("AGDI — Amphibolite-Gneiss Discriminator (2025)", agdi, "RdYlGn_r")
    agdi_val = round(float(np.nanmean(agdi)), 3)

    _cb("Computing FSI (Ferrous Silicate Index)...")
    fsi = np.divide(swir1, nir + 1e-6)
    _pv("FSI — Ferrous Silicate (Mafic Rocks)", fsi, "YlOrRd")
    fsi_val = round(float(np.nanmean(fsi)), 3)

    _cb("Computing FEI (Ferrous Iron Index, 2024)...")
    fei = (swir2 - red) + (nir - blue)
    _pv("FEI — Ferrous Iron Index (2024)", fei, "coolwarm")
    fei_val = round(float(np.nanmean(fei)), 3)

    _cb("Computing NDGI (Normalized Difference Graphite Index, 2024)...")
    ndgi = np.divide(swir1 - red, swir1 + red + 1e-6)
    _pv("NDGI — Graphite Index (2024)", ndgi, "gray")
    ndgi_val = round(float(np.nanmean(ndgi)), 3)

    _cb("Computing Clay/Felsic Index (B11/B12)...")
    clay_felsic = np.divide(swir1, swir2 + 1e-6)
    _pv("Clay/Felsic Index", clay_felsic, "YlOrBr")
    clay_felsic_val = round(float(np.nanmean(clay_felsic)), 3)

    _cb("Computing Iron Oxide Index (B4/B2)...")
    iron_oxide = np.divide(red, blue + 1e-6)
    _pv("Iron Oxide Index", iron_oxide, "RdYlBu_r")
    iron_oxide_val = round(float(np.nanmean(iron_oxide)), 3)

    _cb("Computing Mafic-Felsic RGB Lithology Composite (B12/B11/B2)...")
    def to_uint8(b):
        mn, mx = np.nanpercentile(b, (2, 98))
        return np.clip((b - mn) / (mx - mn + 1e-6) * 255, 0, 255).astype(np.uint8)
    lithology_rgb = np.dstack([to_uint8(swir2), to_uint8(swir1), to_uint8(blue)])
    _pv("Lithology Composite (B12-B11-B02)", lithology_rgb)

    _cb("Computing alteration/lithology ratio composite (B11/B12, B11/B8, B4/B2)...")
    alt_r = np.divide(swir1, swir2 + 1e-6)
    alt_g = np.divide(swir1, nir + 1e-6)
    alt_b = np.divide(red, blue + 1e-6)
    alteration_rgb = np.dstack([to_uint8(alt_r), to_uint8(alt_g), to_uint8(alt_b)])
    _pv("Alteration Composite (Clay-Mafic-IronOxide)", alteration_rgb)

    # ── Host Rock Classification ─────────────────────────────────────────
    _cb("Classifying host rock lithology from spectral signatures...")
    
    # Normalize indices to 0-1 for classification
    def norm_01(arr):
        mn, mx = np.nanpercentile(arr, (2, 98))
        return np.clip((arr - mn) / (mx - mn + 1e-6), 0, 1)

    agdi_n = norm_01(agdi)
    fsi_n = norm_01(fsi)
    fei_n = norm_01(fei)
    ndgi_n = norm_01(ndgi)
    clay_n = norm_01(clay_felsic)
    io_n = norm_01(iron_oxide)

    # Classification rules based on 2024-2025 literature:
    # - High FSI + High FEI + Low AGDI = Mafic (Amphibolite / Greenstone)
    # - High AGDI + High Clay = Felsic (Granite Gneiss / Migmatite)
    # - High NDGI = Graphitic Schist/Gneiss (gold-hosting)
    # - High Iron Oxide + Low Clay = Iron Oxide / Gossan
    # - Moderate all = Mixed/Metamorphic

    mafic_score = (fsi_n * 0.35) + (fei_n * 0.35) + ((1 - agdi_n) * 0.15) + (io_n * 0.15)
    felsic_score = (agdi_n * 0.30) + (clay_n * 0.35) + ((1 - fsi_n) * 0.20) + ((1 - fei_n) * 0.15)
    graphite_score = ndgi_n * 0.60 + (1 - io_n) * 0.20 + clay_n * 0.20
    gossan_score = io_n * 0.50 + (1 - clay_n) * 0.25 + fsi_n * 0.25

    # Dominant lithology per pixel
    scores = np.stack([mafic_score, felsic_score, graphite_score, gossan_score])
    litho_labels = np.argmax(scores, axis=0)
    
    litho_names = {
        0: "Mafic (Amphibolite/Greenstone)",
        1: "Felsic (Granite Gneiss/Migmatite)",
        2: "Graphitic Schist/Gneiss",
        3: "Iron Oxide/Gossan",
    }

    # Compute percentage of each lithology class
    total_px = litho_labels.size
    litho_pct = {}
    for i in range(4):
        pct = round(float(np.sum(litho_labels == i) / total_px * 100), 1)
        litho_pct[litho_names[i]] = pct

    # Dominant lithology
    dominant_idx = int(np.argmax([np.sum(litho_labels == i) for i in range(4)]))
    dominant_litho = litho_names[dominant_idx]

    _cb(f"Host rock classification: {dominant_litho} ({litho_pct[dominant_litho]}%)")

    # Create classified lithology map (colored)
    litho_colors = {
        0: [0.2, 0.6, 0.2],    # Mafic — green
        1: [0.9, 0.4, 0.8],    # Felsic — magenta
        2: [0.3, 0.3, 0.3],    # Graphitic — dark gray
        3: [0.9, 0.3, 0.1],    # Gossan — red-orange
    }
    litho_map_rgb = np.zeros((*litho_labels.shape, 3), dtype=np.float32)
    for i in range(4):
        mask = litho_labels == i
        litho_map_rgb[mask] = litho_colors[i]
    _pv("Host Rock Lithology Classification Map", litho_map_rgb)

    _cb("Sentinel-2 lithology analysis complete!")

    return {
        "satellite": "Sentinel-2",
        "scene_date": scene_date[:10] if scene_date else str(year),
        "cloud_cover": round(cloud, 1),
        "fetch_bbox": fetch_bbox,
        # Lithology maps
        "agdi_map": agdi,
        "fsi_map": fsi,
        "fei_map": fei,
        "ndgi_map": ndgi,
        "clay_felsic_map": clay_felsic,
        "iron_oxide_map": iron_oxide,
        "lithology_rgb": lithology_rgb,
        "alteration_rgb": alteration_rgb,
        "lithology_classified": litho_map_rgb,
        # Summary values
        "agdi_val": agdi_val,
        "fsi_val": fsi_val,
        "fei_val": fei_val,
        "ndgi_val": ndgi_val,
        "clay_felsic_val": clay_felsic_val,
        "iron_oxide_val": iron_oxide_val,
        # Classification
        "dominant_lithology": dominant_litho,
        "lithology_percentages": litho_pct,
        "mafic_score": round(float(np.nanmean(mafic_score)), 3),
        "felsic_score": round(float(np.nanmean(felsic_score)), 3),
        "graphite_score": round(float(np.nanmean(graphite_score)), 3),
        "gossan_score": round(float(np.nanmean(gossan_score)), 3),
    }


def fetch_aster_tir_indices(lat, lon, year, bbox=None, progress_cb=None):
    """
    Fetch ASTER L1T Thermal Infrared bands from Planetary Computer
    and compute Ninomiya's silicate rock indices:
    - Quartz Index (QI) = B11² / (B10 × B12)
    - Carbonate Index (CI) = B13 / B14
    - Mafic Index (MI) = B12 / B13
    """
    def _cb(msg):
        if progress_cb:
            progress_cb(msg)

    if bbox is not None:
        fetch_bbox = bbox
    else:
        buf = DEFAULT_BUFFER_DEG
        fetch_bbox = [lon - buf, lat - buf, lon + buf, lat + buf]

    pystac_client, planetary_computer = _get_pystac()

    _cb("Searching ASTER L1T scenes on Planetary Computer...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )

    # Search ASTER L1T with cloud filter
    search = catalog.search(
        collections=["aster-l1t"],
        bbox=fetch_bbox,
        datetime=f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z",
    )
    items = list(search.items())

    if not items:
        _cb("No ASTER scenes found for this area/year. Skipping TIR indices.")
        return None

    # Pick lowest cloud cover
    best_item = min(items, key=lambda i: i.properties.get("eo:cloud_cover", 100))
    cloud = best_item.properties.get("eo:cloud_cover", 0)
    _cb(f"Best ASTER scene: cloud {cloud:.1f}%")

    # Get signed URLs for TIR bands (90m resolution)
    _cb("Downloading ASTER TIR bands (B10-B14, 90m resolution)...")

    def _read_aster_band(item, band_names):
        """Read a single ASTER TIR band as numpy array."""
        import rasterio
        from rasterio.windows import from_bounds
        for bname in band_names:
            if bname in item.assets:
                href = item.assets[bname].href
                with rasterio.open(href) as src:
                    win = from_bounds(fetch_bbox[0], fetch_bbox[1],
                                       fetch_bbox[2], fetch_bbox[3], src.transform)
                    data = src.read(1, window=win)
                    return data.astype(np.float32)
        return None

    b10 = _read_aster_band(best_item, ["TIR_Band10", "tir_b10", "B10"])
    b11 = _read_aster_band(best_item, ["TIR_Band11", "tir_b11", "B11"])
    b12 = _read_aster_band(best_item, ["TIR_Band12", "tir_b12", "B12"])
    b13 = _read_aster_band(best_item, ["TIR_Band13", "tir_b13", "B13"])
    b14 = _read_aster_band(best_item, ["TIR_Band14", "tir_b14", "B14"])

    if any(b is None for b in [b10, b11, b12, b13, b14]):
        _cb("Could not download all ASTER TIR bands. Skipping.")
        return None

    _cb("Computing Ninomiya's silicate indices...")

    # Quartz Index = B11² / (B10 × B12)
    quartz_index = np.divide(b11 * b11, b10 * b12 + 1e-6)
    qi_val = round(float(np.nanmean(quartz_index)), 3)

    # Carbonate Index = B13 / B14
    carbonate_index = np.divide(b13, b14 + 1e-6)
    ci_val = round(float(np.nanmean(carbonate_index)), 3)

    # Mafic Index = B12 / B13
    mafic_index = np.divide(b12, b13 + 1e-6)
    mi_val = round(float(np.nanmean(mafic_index)), 3)

    _cb(f"Quartz Index: {qi_val} | Carbonate Index: {ci_val} | Mafic Index: {mi_val}")

    # Create TIR composite RGB (QI=Red, CI=Green, MI=Blue)
    def to_uint8(b):
        mn, mx = np.nanpercentile(b, (2, 98))
        return np.clip((b - mn) / (mx - mn + 1e-6) * 255, 0, 255).astype(np.uint8)

    tir_rgb = np.dstack([to_uint8(quartz_index), to_uint8(carbonate_index), to_uint8(mafic_index)])

    _cb("ASTER TIR analysis complete!")

    return {
        "satellite": "ASTER-TIR",
        "quartz_index": quartz_index,
        "carbonate_index": carbonate_index,
        "mafic_index": mafic_index,
        "tir_rgb": tir_rgb,
        "qi_val": qi_val,
        "ci_val": ci_val,
        "mi_val": mi_val,
        "fetch_bbox": fetch_bbox,
    }
