"""
SatIntel Export Utilities — KML/KMZ (Google Earth), GeoTIFF (QGIS/ArcGIS), PNG (reports)
"""
import io
import zipfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from xml.sax.saxutils import escape

try:
    from rasterio.io import MemoryFile
    from rasterio.transform import from_bounds as rio_from_bounds
    HAS_RASTERIO = True
except Exception:
    HAS_RASTERIO = False


# ========================================================
# IMAGE DEFINITIONS — name, key in sat_data, colormap, vmin, vmax
# ========================================================
IMAGE_SPECS = [
    ("01_True_Color_RGB",    "rgb",                  None,        None,    None),
    ("02_False_Color_SWIR",  "false_color",          None,        None,    None),
    ("03_Iron_Oxide_Ratio",  "iron_oxide_map",       "RdYlBu_r",  None,    None),
    ("04_Clay_Hydroxyl",     "clay_map",             "YlOrBr",    None,    None),
    ("05_NDVI_Vegetation",   "ndvi_map",             "RdYlGn",   -0.3,    0.8),
    ("06_Silica_Proxy",      "silica_map",           "bone",      None,    None),
    ("07_Crosta_Iron_PCA",   "crosta_iron_pca",      "RdYlBu_r",  None,    None),
    ("08_Crosta_Clay_PCA",   "crosta_clay_pca",      "YlOrBr",    None,    None),
    ("09_Lineament_Density", "lineament_density_map","hot",       None,    None),
    ("10_Intersection_Map",  "intersection_map",     "magma",      None,    None),
]


# ========================================================
# POLYGON → KML
# ========================================================

def polygon_to_kml(polygon_geojson, metadata=None):
    """Convert a GeoJSON polygon to a KML document string with metadata."""
    if not polygon_geojson:
        return None

    props = polygon_geojson.get("properties", {})
    name = props.get("name", "Concession")
    rings = polygon_geojson["geometry"]["coordinates"]
    outer_ring = rings[0]
    coord_str = " ".join(f"{lon},{lat},0" for lon, lat in outer_ring)

    desc_parts = []
    if metadata:
        for k, v in metadata.items():
            desc_parts.append(f"<b>{escape(str(k))}:</b> {escape(str(v))}")
    description = "<br/>".join(desc_parts) if desc_parts else name
    safe_name = escape(name)

    kml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
        '  <Document>\n'
        f'    <name>{safe_name}</name>\n'
        '    <Style id="concessionStyle">\n'
        '      <LineStyle><color>ff00d7ff</color><width>4</width></LineStyle>\n'
        '      <PolyStyle><color>33ffe500</color><fill>1</fill><outline>1</outline></PolyStyle>\n'
        '    </Style>\n'
        '    <Placemark>\n'
        f'      <name>{safe_name}</name>\n'
        f'      <description><![CDATA[{description}]]></description>\n'
        '      <styleUrl>#concessionStyle</styleUrl>\n'
        '      <Polygon><tessellate>1</tessellate>\n'
        '        <outerBoundaryIs><LinearRing><tessellate>1</tessellate>\n'
        f'          <coordinates>{coord_str}</coordinates>\n'
        '        </LinearRing></outerBoundaryIs>\n'
        '      </Polygon>\n'
        '    </Placemark>\n'
        '  </Document>\n'
        '</kml>'
    )
    return kml


# ========================================================
# ARRAY → PNG BYTES
# ========================================================

def _apply_colormap(array, cmap_name, vmin=None, vmax=None):
    """Apply a matplotlib colormap to a 2D float array, return RGB uint8."""
    arr = array.copy().astype(float)
    if vmin is None:
        vmin = np.nanpercentile(arr, 2)
    if vmax is None:
        vmax = np.nanpercentile(arr, 98)
    arr = np.clip(arr, vmin, vmax)
    norm = (arr - vmin) / (vmax - vmin + 1e-10)
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(norm)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    nan_mask = np.isnan(array) | np.isnan(norm)
    rgb[nan_mask] = [0, 0, 0]
    return rgb


def _array_to_png_bytes(array, cmap_name=None, vmin=None, vmax=None):
    """Convert a numpy array (2D or 3D) to PNG bytes."""
    if array.ndim == 3:
        rgb = array.astype(np.uint8)
        if rgb.max() <= 1:
            rgb = (rgb * 255).astype(np.uint8)
    elif cmap_name:
        rgb = _apply_colormap(array, cmap_name, vmin, vmax)
    else:
        mn, mx = np.nanmin(array), np.nanmax(array)
        gray = np.clip((array - mn) / (mx - mn + 1e-10) * 255, 0, 255).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=2)

    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ========================================================
# ARRAY → GEOTIFF BYTES
# ========================================================

def _array_to_geotiff_bytes(array, fetch_bbox):
    """Convert a numpy array to a GeoTIFF (EPSG:4326) as bytes."""
    if not HAS_RASTERIO:
        return None

    lon_min, lat_min, lon_max, lat_max = fetch_bbox

    if array.ndim == 2:
        h, w = array.shape
        count = 1
        data = array[np.newaxis, :, :].astype("float32")
    else:
        h, w = array.shape[:2]
        count = array.shape[2]
        data = array.transpose(2, 0, 1).astype("float32")

    transform = rio_from_bounds(lon_min, lat_min, lon_max, lat_max, w, h)

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff", height=h, width=w, count=count,
            dtype="float32", crs="EPSG:4326", transform=transform,
        ) as dst:
            for i in range(count):
                dst.write(data[i], i + 1)
        return memfile.read()


# ========================================================
# KML POLYGON FRAGMENT (for inclusion in master KML)
# ========================================================

def _kml_polygon_fragment(polygon_geojson, metadata=None):
    """Generate a KML Placemark fragment for the polygon."""
    props = polygon_geojson.get("properties", {})
    name = props.get("name", "Concession")
    rings = polygon_geojson["geometry"]["coordinates"]
    outer_ring = rings[0]
    coord_str = " ".join(f"{lon},{lat},0" for lon, lat in outer_ring)
    safe_name = escape(name)

    return (
        '    <Style id="concessionStyle">\n'
        '      <LineStyle><color>ff00d7ff</color><width>4</width></LineStyle>\n'
        '      <PolyStyle><color>33ffe500</color></PolyStyle>\n'
        '    </Style>\n'
        '    <Placemark>\n'
        f'      <name>{safe_name}</name>\n'
        '      <styleUrl>#concessionStyle</styleUrl>\n'
        '      <Polygon><tessellate>1</tessellate>\n'
        '        <outerBoundaryIs><LinearRing><tessellate>1</tessellate>\n'
        f'          <coordinates>{coord_str}</coordinates>\n'
        '        </LinearRing></outerBoundaryIs>\n'
        '      </Polygon>\n'
        '    </Placemark>'
    )


# ========================================================
# KMZ BUNDLE (Google Earth — polygon + all image overlays)
# ========================================================

def create_kmz_bundle(sat_data, polygon_geojson=None, metadata=None, fetch_bbox=None):
    """Create a KMZ file containing the polygon + all images as GroundOverlays."""
    if fetch_bbox is None:
        fetch_bbox = sat_data.get("fetch_bbox")
    if fetch_bbox is None:
        return None

    lon_min, lat_min, lon_max, lat_max = fetch_bbox
    scene_date = sat_data.get("scene_date", "")
    cloud_cover = sat_data.get("cloud_cover", "")
    satellite = sat_data.get("Satellite_Used", "")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        master_parts = []

        # Polygon placemark
        if polygon_geojson:
            master_parts.append(_kml_polygon_fragment(polygon_geojson, metadata))

        # Image GroundOverlays
        for img_name, key, cmap, vmin, vmax in IMAGE_SPECS:
            if key not in sat_data:
                continue
            png_filename = f"{img_name}.png"
            arr = sat_data[key]
            png_bytes = _array_to_png_bytes(arr, cmap_name=cmap, vmin=vmin, vmax=vmax)
            zf.writestr(png_filename, png_bytes)

            display_name = img_name.replace("_", " ")
            frag = (
                '    <GroundOverlay>\n'
                f'      <name>{display_name}</name>\n'
                f'      <Icon><href>{png_filename}</href></Icon>\n'
                '      <LatLonBox>\n'
                f'        <north>{lat_max}</north>\n'
                f'        <south>{lat_min}</south>\n'
                f'        <east>{lon_max}</east>\n'
                f'        <west>{lon_min}</west>\n'
                '      </LatLonBox>\n'
                '      <color>ffffffff</color>\n'
                '    </GroundOverlay>'
            )
            master_parts.append(frag)

        # Assemble master KML
        overlays_xml = "\n".join(master_parts)
        master_kml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
            '  <Document>\n'
            f'    <name>SatIntel Export - {scene_date}</name>\n'
            f'    <description><![CDATA[Scene: {scene_date}<br/>Cloud: {cloud_cover}%<br/>Source: {satellite}]]></description>\n'
            f'{overlays_xml}\n'
            '  </Document>\n'
            '</kml>'
        )
        zf.writestr("satintel_overlays.kml", master_kml)

    buf.seek(0)
    return buf.getvalue()


# ========================================================
# GEOTIFF BUNDLE (QGIS / ArcGIS — georeferenced rasters)
# ========================================================

def create_geotiff_bundle(sat_data, fetch_bbox=None):
    """Create a ZIP file containing all images as GeoTIFF (EPSG:4326)."""
    if fetch_bbox is None:
        fetch_bbox = sat_data.get("fetch_bbox")
    if fetch_bbox is None or not HAS_RASTERIO:
        return None

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for img_name, key, cmap, vmin, vmax in IMAGE_SPECS:
            if key not in sat_data:
                continue
            arr = sat_data[key]
            geotiff_bytes = _array_to_geotiff_bytes(arr, fetch_bbox)
            if geotiff_bytes:
                zf.writestr(f"{img_name}.tif", geotiff_bytes)

    buf.seek(0)
    return buf.getvalue()


# ========================================================
# PNG BUNDLE (reports / presentations)
# ========================================================

def create_png_bundle(sat_data):
    """Create a ZIP file containing all images as high-res PNGs."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for img_name, key, cmap, vmin, vmax in IMAGE_SPECS:
            if key not in sat_data:
                continue
            arr = sat_data[key]
            png_bytes = _array_to_png_bytes(arr, cmap_name=cmap, vmin=vmin, vmax=vmax)
            zf.writestr(f"{img_name}.png", png_bytes)

    buf.seek(0)
    return buf.getvalue()
