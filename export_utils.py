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

# ================================================================
# IMAGE SPECS
# ================================================================
IMAGE_SPECS = [
    ("01_True_Color_RGB",     "rgb",                  None,        None,    None),
    ("02_False_Color_SWIR",   "false_color",          None,        None,    None),
    ("03_Iron_Oxide_Ratio",   "iron_oxide_map",        "RdYlBu_r",  None,    None),
    ("04_Clay_Hydroxyl",      "clay_map",              "YlOrBr",    None,    None),
    ("05_NDVI_Vegetation",    "ndvi_map",              "RdYlGn",   -0.3,    0.8),
    ("06_Silica_Proxy",       "silica_map",            "bone",      None,    None),
    ("07_Crosta_Iron_PCA",    "crosta_iron_pca",       "RdYlBu_r",  None,    None),
    ("08_Crosta_Clay_PCA",    "crosta_clay_pca",       "YlOrBr",    None,    None),
    ("09_Lineament_Density",  "lineament_density_map", "hot",       None,    None),
    ("10_Intersection_Map",   "intersection_map",      "magma",      None,    None),
]

# ================================================================
# POLYGON → KML
# ================================================================
def polygon_to_kml(polygon_geojson, metadata=None):
    if not polygon_geojson:
        return None
    props = polygon_geojson.get("properties", {})
    name = props.get("name", "Concession")
    rings = polygon_geojson["geometry"]["coordinates"]
    outer_ring = rings[0]
    coord_str = "  ".join(f"{lon},{lat},0" for lon, lat in outer_ring)
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

# ================================================================
# ARRAY → PNG / GEOTIFF
# ================================================================
def _apply_colormap(array, cmap_name, vmin=None, vmax=None):
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

def _array_to_geotiff_bytes(array, fetch_bbox):
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
        with memfile.open(driver="GTiff", height=h, width=w, count=count,
                          dtype="float32", crs="EPSG:4326", transform=transform) as dst:
            for i in range(count):
                dst.write(data[i], i + 1)
        return memfile.read()

# ================================================================
# TARGETS KMZ (Professional Exploration Export)
# ================================================================
_STYLE_HIGH = (
    '     <Style id="highPriority">\n'
    '       <LineStyle><color>ff0000ff</color><width>2</width></LineStyle>\n'
    '       <PolyStyle><color>400000ff</color></PolyStyle>\n'
    '     </Style>'
)
_STYLE_MED = (
    '     <Style id="mediumPriority">\n'
    '       <LineStyle><color>ff00aaff</color><width>1.5</width></LineStyle>\n'
    '       <PolyStyle><color>4000aaff</color></PolyStyle>\n'
    '     </Style>'
)
_STYLE_LOW = (
    '     <Style id="lowPriority">\n'
    '       <LineStyle><color>ff00ffaa</color><width>1</width></LineStyle>\n'
    '       <PolyStyle><color>4000ffaa</color></PolyStyle>\n'
    '     </Style>'
)
_STYLE_BOUNDARY = (
    '     <Style id="licenseBoundary">\n'
    '       <LineStyle><color>ffffffff</color><width>2</width></LineStyle>\n'
    '       <PolyStyle><color>00ffffff</color><fill>0</fill><outline>1</outline></PolyStyle>\n'
    '     </Style>'
)

def _target_placemark(target):
    """Generate a KML Placemark for a single exploration target."""
    coord_str = "  ".join(f"{lon},{lat},0" for lon, lat in target["polygon"])
    style_map = {"HIGH": "highPriority", "MEDIUM": "mediumPriority", "LOW": "lowPriority"}
    style_id = style_map.get(target["priority"], "lowPriority")
    safe_id = escape(target["id"])
    
    desc = (
        f"<b>Target / Alvo:</b> {target['id']}<br/>"
        f"<b>Composite Score / Pontuacao Composta:</b> {target['score']}<br/>"
        f"<b>Priority / Prioridade:</b> {target['priority']}<br/>"
        f"<b>Structural Control / Controle Estrutural:</b> {escape(target['structural_control'])}<br/>"
        f"<b>Lithology / Litologia:</b> {escape(target['lithology'])}<br/>"
        f"<b>Radius / Raio:</b> ~{target['radius_m']} m<br/>"
        f"<b>Coordinates:</b> {target['lat']:.6f}, {target['lon']:.6f}<br/>"
        f"<b>IO Score:</b> {target['io_score']} | <b>Clay Score:</b> {target['clay_score']} | "
        f"<b>Structural:</b> {target['struct_score']} | <b>Geomorphology:</b> {target['geomorph_score']} | "
        f"<b>Lineament:</b> {target['line_score']}<br/>"
        f"<b>Description / Descricao:</b> {escape(target['description_en'])}<br/>"
        f"<b>Descricao PT:</b> {escape(target['description_pt'])}"
    )
    return (
        '    <Placemark>\n'
        f'      <name>{safe_id} (Score: {target["score"]})</name>\n'
        f'      <description><![CDATA[{desc}]]></description>\n'
        f'      <styleUrl>#{style_id}</styleUrl>\n'
        '      <Polygon><tessellate>1</tessellate>\n'
        '        <outerBoundaryIs><LinearRing><tessellate>1</tessellate>\n'
        f'          <coordinates>{coord_str}</coordinates>\n'
        '        </LinearRing></outerBoundaryIs>\n'
        '      </Polygon>\n'
        '    </Placemark>'
    )

def _boundary_placemark(polygon_geojson, metadata=None):
    """Generate a KML Placemark for the license boundary polygon."""
    props = polygon_geojson.get("properties", {})
    name = props.get("name", "License Boundary")
    rings = polygon_geojson["geometry"]["coordinates"]
    outer_ring = rings[0]
    coord_str = "  ".join(f"{lon},{lat},0" for lon, lat in outer_ring)
    safe_name = escape(name)
    
    area_str = ""
    if metadata and "Area / Dimensao" in metadata:
        area_str = metadata["Area / Dimensao"]
    elif metadata and "\xc1rea / Dimens\xc3o" in metadata:
        area_str = metadata["\xc1rea / Dimens\xc3o"]
    
    return (
        '    <Placemark>\n'
        f'      <name>{safe_name}</name>\n'
        f'      <description>{escape(area_str)} concession area</description>\n'
        '      <styleUrl>#licenseBoundary</styleUrl>\n'
        '      <Polygon><tessellate>1</tessellate>\n'
        '        <outerBoundaryIs><LinearRing><tessellate>1</tessellate>\n'
        f'          <coordinates>{coord_str}</coordinates>\n'
        '        </LinearRing></outerBoundaryIs>\n'
        '      </Polygon>\n'
        '    </Placemark>'
    )

def create_targets_kmz(targets, polygon_geojson=None, metadata=None, sat_data=None):
    """
    Create a professional exploration targets KMZ file matching the reference format:
    - Folders: License Boundary, High/Medium/Low Priority Targets
    - Color-coded polygons with bilingual descriptions
    - Composite scores, structural controls, lithology
    - Auto-detects commodity (Copper vs Gold) for correct WLC formula display
    """
    fetch_bbox = sat_data.get("fetch_bbox") if sat_data else None
    scene_date = sat_data.get("scene_date", "") if sat_data else ""
    satellite = sat_data.get("Satellite_Used", "") if sat_data else ""
    
    # Group targets by priority
    high = [t for t in targets if t["priority"] == "HIGH"]
    medium = [t for t in targets if t["priority"] == "MEDIUM"]
    low = [t for t in targets if t["priority"] == "LOW"]
    
    # Build license boundary folder
    boundary_folder = ""
    if polygon_geojson:
        boundary_folder = (
            '  <Folder>\n'
            '    <name>License Boundary / Limite da Licenca</name>\n'
            + _boundary_placemark(polygon_geojson, metadata) + '\n'
            '  </Folder>'
        )
    
    def build_priority_folder(name, target_list):
        if not target_list:
            return ""
        placemarks = "\n".join(_target_placemark(t) for t in target_list)
        return (
            f'  <Folder>\n'
            f'    <name>{name}</name>\n'
            f'{placemarks}\n'
            f'  </Folder>'
        )
    
    high_folder = build_priority_folder("High Priority Targets / Alvos de Alta Prioridade", high)
    med_folder = build_priority_folder("Medium Priority Targets / Alvos de Media Prioridade", medium)
    low_folder = build_priority_folder("Low Priority Targets / Alvos de Baixa Prioridade", low)
    
    # Assemble folders
    folders = "\n".join(f for f in [boundary_folder, high_folder, med_folder, low_folder] if f)
    
    # ✅ AUTO-DETECT COMMODITY AND WRITE CORRECT WLC FORMULA
    is_copper = any("[COPPER]" in t.get("description_en", "") for t in targets)
    if is_copper:
        desc_text = (
            f"Copper-target zones. Composite score: "
            f"IO(0.25) + CLAY(0.20) + Structural(0.20) + Geomorphology(0.15) + Lineament(0.10) + ASTER_CuProxy(0.10). "
            f"Scene: {scene_date}. Source: {satellite}."
        )
    else:
        # ✅ UPDATED FOR PHASE 1 OROGENIC GOLD WEIGHTS
        desc_text = (
            f"Gold-target zones. Composite score: "
            f"IO(0.20) + CLAY(0.20) + Structural(0.35) + Geomorphology(0.10) + Lineament(0.15). "
            f"Scene: {scene_date}. Source: {satellite}."
        )
    
    kml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
        '<Document>\n'
        f'  <name>SatIntel Exploration Targets</name>\n'
        f'  <description>{escape(desc_text)}</description>\n'
        f'{_STYLE_HIGH}\n'
        f'{_STYLE_MED}\n'
        f'{_STYLE_LOW}\n'
        f'{_STYLE_BOUNDARY}\n'
        f'{folders}\n'
        '</Document>\n'
        '</kml>'
    )
    
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml)
    buf.seek(0)
    return buf.getvalue()

# ================================================================
# IMAGE OVERLAY KMZ (Google Earth — polygon + all image overlays)
# ================================================================
def _kml_polygon_fragment(polygon_geojson, metadata=None):
    props = polygon_geojson.get("properties", {})
    name = props.get("name", "Concession")
    rings = polygon_geojson["geometry"]["coordinates"]
    outer_ring = rings[0]
    coord_str = "  ".join(f"{lon},{lat},0" for lon, lat in outer_ring)
    safe_name = escape(name)
    return (
        '     <Style id="concessionStyle">\n'
        '       <LineStyle><color>ff00d7ff</color><width>4</width></LineStyle>\n'
        '       <PolyStyle><color>33ffe500</color></PolyStyle>\n'
        '     </Style>\n'
        '     <Placemark>\n'
        f'       <name>{safe_name}</name>\n'
        '       <styleUrl>#concessionStyle</styleUrl>\n'
        '       <Polygon><tessellate>1</tessellate>\n'
        '         <outerBoundaryIs><LinearRing><tessellate>1</tessellate>\n'
        f'           <coordinates>{coord_str}</coordinates>\n'
        '         </LinearRing></outerBoundaryIs>\n'
        '       </Polygon>\n'
        '     </Placemark>'
    )

def create_kmz_bundle(sat_data, polygon_geojson=None, metadata=None, fetch_bbox=None):
    if sat_data is None:
        return None
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
        if polygon_geojson:
            master_parts.append(_kml_polygon_fragment(polygon_geojson, metadata))
        
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

# ================================================================
# GEOTIFF BUNDLE
# ================================================================
def create_geotiff_bundle(sat_data, fetch_bbox=None):
    if sat_data is None:
        return None
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

# ================================================================
# PNG BUNDLE
# ================================================================
def create_png_bundle(sat_data):
    if sat_data is None:
        return None
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
# ==============================================================================
# PHASE 6: KML EXPORT FOR ALLUVIAL SOURCE TRACER
# ==============================================================================

def create_kml(targets, filename="source_trace.kml", stream_polylines=None):
    """Generates a KML string for Google Earth from target dicts + optional stream network."""
    kml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    kml += '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
    kml += '<Document>\n'
    
    # Style: Blue stream lines (AABBGGRR format: ff=alpha, ff=blue, 00=green, 00=red)
    kml += '<Style id="blueStream">\n'
    kml += '  <LineStyle>\n'
    kml += '    <color>ffff0000</color>\n'
    kml += '    <width>3</width>\n'
    kml += '  </LineStyle>\n'
    kml += '</Style>\n'
    
    # Style: Yellow target pins
    kml += '<Style id="targetPin">\n'
    kml += '  <IconStyle>\n'
    kml += '    <color>ffffff00</color>\n'
    kml += '    <scale>1.2</scale>\n'
    kml += '    <Icon><href>http://maps.google.com/mapfiles/kml/paddle/ylw-circle.png</href></Icon>\n'
    kml += '  </IconStyle>\n'
    kml += '</Style>\n'
    
    # Stream network polylines (rendered first = background layer)
    if stream_polylines:
        for idx, line in enumerate(stream_polylines):
            if len(line) < 2:
                continue
            # Explicit float conversion + 6 decimal places for clean KML coordinates
            coords_str = " ".join([f"{float(lon):.6f},{float(lat):.6f},0" for lon, lat in line])
            kml += f'<Placemark>\n'
            kml += f'  <name>Stream Segment {idx+1}</name>\n'
            kml += f'  <styleUrl>#blueStream</styleUrl>\n'
            kml += f'  <LineString>\n'
            kml += f'    <tessellate>1</tessellate>\n'
            kml += f'    <coordinates>{coords_str}</coordinates>\n'
            kml += f'  </LineString>\n'
            kml += f'</Placemark>\n'
    
    # Target placemarks (rendered after = foreground layer)
    for t in targets:
        lat = float(t.get('lat', 0.0))
        lon = float(t.get('lon', 0.0))
        name = f"{t.get('source_type', 'Unknown')} (Score: {t.get('score', 0)})"
        desc = (f"TWI: {t.get('twi_score', 0)} | Curvature: {t.get('curvature_score', 0)} | "
                f"HMI: {t.get('hmi_score', 0)} | FSI: {t.get('fsi_score', 0)} | "
                f"Struct: {t.get('struct_score', 0)}\nTrap: {t.get('trap_note', '')}")
        
        kml += f'<Placemark>\n'
        kml += f'  <name>{name}</name>\n'
        kml += f'  <description>{desc}</description>\n'
        kml += f'  <styleUrl>#targetPin</styleUrl>\n'
        kml += f'  <Point>\n'
        kml += f'    <coordinates>{lon:.6f},{lat:.6f},0</coordinates>\n'
        kml += f'  </Point>\n'
        kml += f'</Placemark>\n'
    
    kml += '</Document>\n</kml>'
    return kml.encode('utf-8')


def create_gpx(targets, stream_polylines=None, filename="alluvial_source_trace.gpx"):
    """Generates GPX 1.1 file for Garmin/handheld GPS devices.
    
    Waypoints for targets + track segments for stream network.
    Compatible with Garmin GPSMAP, Etrex, Montana, and most field GPS units.
    """
    from datetime import datetime, timezone
    
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    gpx = '<?xml version="1.0" encoding="UTF-8"?>\n'
    gpx += '<gpx version="1.1" creator="GeoRemote Phase 6"\n'
    gpx += '     xmlns="http://www.topografix.com/GPX/1/1"\n'
    gpx += '     xmlns:gpxx="http://www.garmin.com/xmlschemas/GpxExtensions/v3"\n'
    gpx += '     xmlns:wptx1="http://www.garmin.com/xmlschemas/WaypointExtension/v1"\n'
    gpx += '     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
    gpx += '     xsi:schemaLocation="http://www.topografix.com/GPX/1/1\n'
    gpx += '     http://www.topografix.com/GPX/1/1/gpx.xsd">\n'
    gpx += f'  <metadata>\n'
    gpx += f'    <name>Alluvial Source Trace</name>\n'
    gpx += f'    <desc>Rastreador de Fonte Aluvial - Generated {ts}</desc>\n'
    gpx += f'    <time>{ts}</time>\n'
    gpx += f'  </metadata>\n'
    
    # Waypoints for each target
    for i, t in enumerate(targets, 1):
        lat = float(t.get('lat', 0.0))
        lon = float(t.get('lon', 0.0))
        ele = float(t.get('elevation', 0.0)) if t.get('elevation') else None
        score = t.get('score', 0)
        stype = t.get('source_type', 'Target')
        twi = t.get('twi_score', 0)
        curv = t.get('curvature_score', 0)
        
        # Short name for GPS display (max ~14 chars on most Garmin units)
        short_name = f"T{i:02d} S{score:.0f}"
        
        gpx += f'  <wpt lat="{lat:.6f}" lon="{lon:.6f}">\n'
        if ele:
            gpx += f'    <ele>{ele:.1f}</ele>\n'
        gpx += f'    <time>{ts}</time>\n'
        gpx += f'    <name>{short_name}</name>\n'
        gpx += f'    <desc>{stype} | TWI:{twi} Curv:{curv} Score:{score}</desc>\n'
        gpx += f'    <sym>Waypoint</sym>\n'
        gpx += f'    <type>Geocache</type>\n'
        # Garmin extension for display color
        gpx += f'    <extensions>\n'
        gpx += f'      <gpxx:WaypointExtension>\n'
        gpx += f'        <gpxx:DisplayColor>Yellow</gpxx:DisplayColor>\n'
        gpx += f'      </gpxx:WaypointExtension>\n'
        gpx += f'    </extensions>\n'
        gpx += f'  </wpt>\n'
    
    # Track for stream network (blue lines on GPS)
    if stream_polylines:
        gpx += '  <trk>\n'
        gpx += '    <name>Rede de Drenagem</name>\n'
        gpx += '    <desc>Stream network from D8 flow routing</desc>\n'
        gpx += '    <extensions>\n'
        gpx += '      <gpxx:TrackExtension>\n'
        gpx += '        <gpxx:DisplayColor>Blue</gpxx:DisplayColor>\n'
        gpx += '      </gpxx:TrackExtension>\n'
        gpx += '    </extensions>\n'
        
        for idx, line in enumerate(stream_polylines):
            if len(line) < 2:
                continue
            gpx += f'    <trkseg>\n'
            for lon, lat in line:
                gpx += f'      <trkpt lat="{float(lat):.6f}" lon="{float(lon):.6f}">\n'
                gpx += f'        <time>{ts}</time>\n'
                gpx += f'      </trkpt>\n'
            gpx += f'    </trkseg>\n'
        
        gpx += '  </trk>\n'
    
    gpx += '</gpx>\n'
    return gpx.encode("utf-8")


def create_csv(targets, stream_polylines=None):
    """Generates CSV file with waypoints for universal GPS import.
    
    Compatible with Garmin BaseCamp, QGIS, and most GPS software.
    """
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow(["Number", "Name", "Latitude", "Longitude", "Elevation", 
                      "Type", "Score", "TWI", "Curvature", "Notes"])
    
    # Target waypoints
    for i, t in enumerate(targets, 1):
        writer.writerow([
            i,
            f"Target_{i:02d}",
            f"{float(t.get('lat', 0.0)):.6f}",
            f"{float(t.get('lon', 0.0)):.6f}",
            f"{float(t.get('elevation', 0.0)):.1f}" if t.get('elevation') else "",
            t.get('source_type', 'Unknown'),
            t.get('score', 0),
            t.get('twi_score', 0),
            t.get('curvature_score', 0),
            t.get('trap_note', '')
        ])
    
    # Stream segment endpoints as additional waypoints
    if stream_polylines:
        writer.writerow([])
        writer.writerow(["Stream Network Points (use as track reference)"])
        writer.writerow(["Segment", "Point", "Latitude", "Longitude"])
        for idx, line in enumerate(stream_polylines, 1):
            for j, (lon, lat) in enumerate(line, 1):
                writer.writerow([f"Seg{idx:02d}", f"P{j:03d}", 
                                f"{float(lat):.6f}", f"{float(lon):.6f}"])
    
    return output.getvalue().encode("utf-8")
