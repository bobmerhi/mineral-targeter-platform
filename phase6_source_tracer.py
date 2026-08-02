# phase6_source_tracer.py - COMPLETE INTEGRATED MODULE
# D8 Routing (pure numpy) + AW3D30 Georeferencing + KML Stream Export
# ==============================================================================

import numpy as np
from scipy.ndimage import label
from collections import deque
import warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# GEOREFERENCING FUNCTIONS (AW3D30 WGS84 / EPSG:4326)
# ==============================================================================

def lat_lon_to_pixel(lat, lon, dem_data, sat_data=None):
    """Converts WGS84 lat/lon to pixel row/col."""
    if hasattr(dem_data, 'transform'):
        try:
            inv_t = ~dem_data.transform
            col, row = inv_t * (lon, lat)
            return int(row), int(col)
        except Exception:
            pass
    
    h, w = dem_data.shape
    if sat_data and sat_data.get("fetch_bbox"):
        min_lon, min_lat, max_lon, max_lat = sat_data["fetch_bbox"]
        col = int((lon - min_lon) / (max_lon - min_lon) * w)
        row = int((max_lat - lat) / (max_lat - min_lat) * h)
        return max(0, min(h - 1, row)), max(0, min(w - 1, col))
    return int(h / 2), int(w / 2)


def pixel_to_lat_lon(row, col, dem_data, sat_data=None):
    """Converts pixel row/col to WGS84 lat/lon (pixel center)."""
    if hasattr(dem_data, 'transform'):
        try:
            t = dem_data.transform
            lon, lat = t * (col + 0.5, row + 0.5)
            return lat, lon
        except Exception:
            pass
    
    h, w = dem_data.shape
    if sat_data and sat_data.get("fetch_bbox"):
        min_lon, min_lat, max_lon, max_lat = sat_data["fetch_bbox"]
        lon = min_lon + ((col + 0.5) / max(w, 1)) * (max_lon - min_lon)
        lat = max_lat - ((row + 0.5) / max(h, 1)) * (max_lat - min_lat)
        return lat, lon
    return None, None


# ==============================================================================
# D8 FLOW ROUTING (Pure NumPy — no external C++ dependency)
# ==============================================================================

D8_OFFSETS = {
    1: (0, 1),     # E
    2: (1, 1),     # SE
    4: (1, 0),     # S
    8: (1, -1),    # SW
    16: (0, -1),   # W
    32: (-1, -1),  # NW
    64: (-1, 0),   # N
    128: (-1, 1),  # NE
}

DIRECTIONS = [
    (0, 1, 2.0),     # E
    (1, 1, 1.414),   # SE
    (1, 0, 2.0),     # S
    (1, -1, 1.414),  # SW
    (0, -1, 2.0),    # W
    (-1, -1, 1.414), # NW
    (-1, 0, 2.0),    # N
    (-1, 1, 1.414),  # NE
]
D8_CODES = [1, 2, 4, 8, 16, 32, 64, 128]


def _compute_d8_flow_direction(dem):
    """D8 flow direction: steepest downhill neighbor for each pixel."""
    h, w = dem.shape
    dem_pad = np.full((h + 2, w + 2), np.nan)
    dem_pad[1:-1, 1:-1] = dem
    
    flow_dir = np.zeros((h, w), dtype=np.uint8)
    max_slope = np.full((h, w), 0.0)
    
    for (dr, dc, dist), code in zip(DIRECTIONS, D8_CODES):
        neighbor = dem_pad[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        slope = (dem - neighbor) / dist
        mask = slope > max_slope
        max_slope[mask] = slope[mask]
        flow_dir[mask] = code
    
    return flow_dir


def _compute_flow_accumulation(flow_dir):
    """Flow accumulation via topological sort (queue-based, no recursion)."""
    h, w = flow_dir.shape
    flow_acc = np.ones((h, w), dtype=np.float64)
    upstream_count = np.zeros((h, w), dtype=np.int32)
    
    # Count upstream contributors (vectorized — clean 2D shift)
    for code, (dr, dc) in D8_OFFSETS.items():
        mask = (flow_dir == code)
        # For each cell with this flow direction, it drains to (r+dr, c+dc)
        # So upstream_count[r+dr, c+dc] += mask[r, c]
        if dr >= 0 and dc >= 0:
            upstream_count[dr:, dc:] += mask[:h-dr, :w-dc]
        elif dr >= 0 and dc < 0:
            upstream_count[dr:, :w+dc] += mask[:h-dr, -dc:]
        elif dr < 0 and dc >= 0:
            upstream_count[:h+dr, dc:] += mask[-dr:, :w-dc]
        else:  # dr < 0 and dc < 0
            upstream_count[:h+dr, :w+dc] += mask[-dr:, -dc:]
    
    # Topological sort: process cells with 0 upstream first
    queue = deque()
    sources = np.where(upstream_count == 0)
    for r, c in zip(sources[0], sources[1]):
        queue.append((int(r), int(c)))
    
    processed = np.zeros((h, w), dtype=bool)
    
    while queue:
        r, c = queue.popleft()
        if processed[r, c]:
            continue
        processed[r, c] = True
        
        code = flow_dir[r, c]
        if code == 0:
            continue
        
        dr, dc = D8_OFFSETS.get(code, (0, 0))
        tr, tc = r + dr, c + dc
        
        if 0 <= tr < h and 0 <= tc < w and not processed[tr, tc]:
            flow_acc[tr, tc] += flow_acc[r, c]
            upstream_count[tr, tc] -= 1
            if upstream_count[tr, tc] == 0:
                queue.append((tr, tc))
    
    return flow_acc


def _compute_hydro_routing(dem_data, sat_data=None):
    """Computes D8 flow direction and accumulation."""
    arr = np.array(dem_data, dtype=np.float64)
    if np.any(np.isnan(arr)):
        valid_med = np.nanmedian(arr) if np.any(~np.isnan(arr)) else 300.0
        arr = np.where(np.isnan(arr), valid_med, arr)
    
    flow_dir = _compute_d8_flow_direction(arr)
    flow_acc = _compute_flow_accumulation(flow_dir)
    return flow_dir, flow_acc


# ==============================================================================
# STREAM NETWORK EXTRACTION
# ==============================================================================

def _extract_watershed_streams(flow_acc, dem_data, sat_data=None, min_upstream_cells=50):
    """Extracts stream network polylines from flow accumulation."""
    stream_mask = flow_acc >= min_upstream_cells
    labeled, num_segments = label(stream_mask)
    polylines = []
    
    for i in range(1, num_segments + 1):
        segment_pixels = np.where(labeled == i)
        if len(segment_pixels[0]) < 5:
            continue
        
        rows, cols = segment_pixels
        sorted_idx = np.lexsort((cols, rows))
        
        polyline = []
        for idx in sorted_idx:
            r, c = rows[idx], cols[idx]
            lat, lon = pixel_to_lat_lon(r, c, dem_data, sat_data)
            if lat is not None and lon is not None:
                polyline.append([lon, lat])
        
        if len(polyline) >= 3:
            polylines.append(polyline)
    
    return polylines


# ==============================================================================
# GEOMETRIC TRAP DETECTION
# ==============================================================================

def _calculate_geometric_traps(dem_data):
    """TWI and Planform Curvature for pocket detection."""
    arr = np.array(dem_data, dtype=np.float64)
    grad_y, grad_x = np.gradient(arr)
    slope = np.arctan(np.sqrt(grad_x ** 2 + grad_y ** 2))
    flow_acc_proxy = 1.0 / (slope + 1e-6)
    twi = np.log(flow_acc_proxy / (np.tan(slope) + 1e-6))
    
    d2z_dx2 = np.gradient(grad_x, axis=1)
    d2z_dy2 = np.gradient(grad_y, axis=0)
    curvature = -(d2z_dx2 + d2z_dy2)
    
    return twi, curvature


def _identify_pockets(twi, curvature, catchment_mask):
    """High TWI + convergent curvature = deposition pockets."""
    if not np.any(catchment_mask):
        return np.zeros_like(twi, dtype=bool)
    twi_thresh = np.nanpercentile(twi[catchment_mask], 90)
    curv_thresh = np.nanpercentile(curvature[catchment_mask], 90)
    return (twi > twi_thresh) & (curvature > curv_thresh) & catchment_mask


# ==============================================================================
# MAIN TRACER FUNCTION
# ==============================================================================

def trace_alluvial_source(confirmed_point_lat, confirmed_point_lon,
                          sat_data, dem_data=None, progress_cb=None,
                          search_radius=1000):
    """
    Traces upstream/downstream from alluvial point using D8 routing + geometry.
    Returns targets + stream polylines for KML export.
    """
    def _cb(msg):
        if progress_cb:
            progress_cb(msg)
    
    if dem_data is None or (isinstance(dem_data, np.ndarray) and (dem_data.size == 0 or np.max(dem_data) == 0)):
        _cb("⚠️ NO DEM DATA AVAILABLE")
        _cb("OPTION B (PAID UPGRADE): LiDAR + Drone Magnetometry")
        return {
            "status": "option_b_required",
            "upgrade_path": "lidar_drone_mag",
            "accuracy_free": "<40%",
            "accuracy_paid": ">85%",
            "cost_estimate_usd_per_km2": 12,
            "delivery_weeks": 2
        }
    
    _cb("Starting Alluvial Source Tracing (D8 Watershed Mode)...")
    
    # Validate georeferencing
    try:
        test_r, test_c = lat_lon_to_pixel(confirmed_point_lat, confirmed_point_lon, dem_data, sat_data)
        check_lat, check_lon = pixel_to_lat_lon(test_r, test_c, dem_data, sat_data)
        if check_lat is not None and check_lon is not None:
            if abs(check_lat - confirmed_point_lat) > 0.001 or abs(check_lon - confirmed_point_lon) > 0.001:
                _cb(f"⚠️ GEOREF: {confirmed_point_lat},{confirmed_point_lon} → {check_lat:.5f},{check_lon:.5f}")
            else:
                _cb("✅ Georeferencing validated")
    except Exception as e:
        _cb(f"⚠️ Georef check skipped: {e}")
    
    # Define catchment radius (for clipping stream extraction, NOT for DEM routing)
    h, w = dem_data.shape
    center_r, center_c = lat_lon_to_pixel(confirmed_point_lat, confirmed_point_lon, dem_data, sat_data)
    
    if hasattr(dem_data, 'transform'):
        px_size_m = abs(dem_data.transform.a) * 111000
    elif sat_data and sat_data.get("fetch_bbox"):
        min_lon, _, max_lon, _ = sat_data["fetch_bbox"]
        px_size_m = ((max_lon - min_lon) / w) * 111000
    else:
        px_size_m = 30.0
    
    radius_px = max(5, int(search_radius / px_size_m))
    _cb(f"Radius: {search_radius}m = {radius_px}px (px: {px_size_m:.1f}m)")
    
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - center_r) ** 2 + (xx - center_c) ** 2)
    catchment_mask = dist <= radius_px
    
    # Use FULL DEM for D8 routing (don't clip — prevents edge artifacts)
    _cb("Step 1: D8 flow direction & accumulation (full DEM)...")
    flow_dir, flow_acc = _compute_hydro_routing(dem_data, sat_data)
    
    # Stream extraction — use percentile threshold within catchment area
    _cb("Step 2: Extracting stream network...")
    # Get flow_acc values within the catchment to set a dynamic threshold
    acc_in_catchment = flow_acc[catchment_mask]
    if len(acc_in_catchment) > 0 and np.max(acc_in_catchment) > 1:
        # Top 10% of flow accumulation within the catchment = streams
        threshold = max(5, np.percentile(acc_in_catchment[acc_in_catchment > 1], 90))
    else:
        threshold = 5
    
    stream_polylines = _extract_watershed_streams(
        flow_acc, dem_data, sat_data, min_upstream_cells=int(threshold))
    _cb(f"   Threshold: {threshold:.0f} cells | {len(stream_polylines)} stream segments found")
    
    # Geometric traps (computed on full DEM, masked to catchment)
    _cb("Step 3: TWI & curvature...")
    twi, curvature = _calculate_geometric_traps(dem_data)
    
    _cb("Step 4: Identifying deposition pockets...")
    pockets = _identify_pockets(twi, curvature, catchment_mask)
    
    # Spectral data (optional)
    hmi_raw = sat_data.get("hmi_map", sat_data.get("iron_oxide_map", None)) if sat_data else None
    fsi_raw = sat_data.get("fsi_map", None) if sat_data else None
    
    _cb("Step 5: Generating targets...")
    targets = []
    labeled_pockets, num_pockets = label(pockets)
    
    for i in range(1, num_pockets + 1):
        pocket_pixels = np.where(labeled_pockets == i)
        if len(pocket_pixels[0]) < 5:
            continue
        
        mean_twi = float(np.nanmean(twi[pocket_pixels]))
        mean_curv = float(np.nanmean(curvature[pocket_pixels]))
        
        hmi_score = 0.0
        fsi_score = 0.0
        if hmi_raw is not None and isinstance(hmi_raw, np.ndarray) and hmi_raw.size > 0:
            try: hmi_score = round(float(np.nanmean(hmi_raw[pocket_pixels])), 3)
            except: pass
        if fsi_raw is not None and isinstance(fsi_raw, np.ndarray) and fsi_raw.size > 0:
            try: fsi_score = round(float(np.nanmean(fsi_raw[pocket_pixels])), 3)
            except: pass
        
        score = round((mean_twi * 0.35 + mean_curv * 0.35 + hmi_score * 0.15 + fsi_score * 0.15) * 100, 1)
        
        cy, cx = int(np.mean(pocket_pixels[0])), int(np.mean(pocket_pixels[1]))
        lat, lon = pixel_to_lat_lon(cy, cx, dem_data, sat_data)
        if lat is None or lon is None:
            continue
        
        targets.append({
            "lat": lat, "lon": lon, "score": score,
            "source_type": "Geometric Deposition Pocket" if mean_curv > 0 else "Quartz Vein Weathering Zone",
            "twi_score": round(mean_twi, 3),
            "curvature_score": round(mean_curv, 3),
            "hmi_score": hmi_score, "fsi_score": fsi_score,
            "struct_score": 0.0,
            "trap_note": "High TWI + Convergent Curvature" if mean_curv > 0 else "High TWI zone"
        })
    
    _cb(f"✅ Done! {len(targets)} targets + {len(stream_polylines)} stream lines")
    
    return {
        "status": "success",
        "targets": targets,
        "stream_polylines": stream_polylines,
        "catchment_mask": catchment_mask,
        "twi_map": twi,
        "curvature_map": curvature,
        "pockets_mask": pockets,
        "flow_dir": flow_dir,
        "flow_acc": flow_acc,
        "data_source": f"AW3D30 D8 Routing (Radius: {search_radius}m)"
    }
