"""
SatIntel Professional PDF Report Generator
Uses DejaVuSans (UTF-8) font for full Unicode support.
Generates multi-page reports with satellite imagery, spectral indices,
and geological analysis.
"""
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF

# Font paths — try multiple locations for Streamlit Cloud compatibility
import os
import matplotlib

def _find_font(name):
    """Find a DejaVu font file on the system."""
    candidates = [
        f"/usr/share/fonts/truetype/dejavu/{name}.ttf",
        os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", f"{name}.ttf"),
        os.path.join(os.path.dirname(matplotlib.get_data_path()), "fonts", "ttf", f"{name}.ttf"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    # Last resort: use matplotlib's font manager
    import matplotlib.font_manager as fm
    match = fm.findfont(f"{name}", fallback_to_default=True)
    return match if os.path.exists(match) else candidates[0]

FONT_REGULAR = _find_font("DejaVuSans")
FONT_BOLD = _find_font("DejaVuSans-Bold")
FONT_MONO = _find_font("DejaVuSansMono")
FONT_MONO_BOLD = _find_font("DejaVuSansMono-Bold")
FONT_ITALIC = _find_font("DejaVuSans-Oblique")
FONT_BOLD_ITALIC = _find_font("DejaVuSans-BoldOblique")

# Brand colors
COLOR_DARK_GREEN = (0, 77, 64)
COLOR_MEDIUM_GREEN = (0, 121, 107)
COLOR_LIGHT_GREEN = (200, 240, 230)
COLOR_ACCENT_GOLD = (212, 160, 23)
COLOR_DARK_GRAY = (45, 45, 45)
COLOR_MEDIUM_GRAY = (100, 100, 100)
COLOR_LIGHT_GRAY = (240, 240, 240)
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (200, 40, 40)
COLOR_ORANGE = (230, 140, 30)
COLOR_BLUE = (30, 80, 160)


def _fig_to_bytes(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf


def _make_index_image(img_array, cmap=None, vmin=None, vmax=None, title="",
                      polygon=None, fetch_bbox=None, targets=None, label=""):
    fig, ax = plt.subplots(figsize=(8, 6))
    display = img_array
    if polygon and fetch_bbox:
        try:
            lon_min, lat_min, lon_max, lat_max = fetch_bbox
            h, w = img_array.shape[:2]
            from matplotlib.path import Path
            grid = np.array([(x, y) for y in range(h) for x in range(w)])
            mask = np.zeros((h, w), dtype=bool)
            for ring in polygon["geometry"]["coordinates"]:
                verts = []
                for p in ring:
                    px = (p[0] - lon_min) / (lon_max - lon_min) * w
                    py = (lat_max - p[1]) / (lat_max - lat_min) * h
                    verts.append((px, py))
                path = Path(verts)
                mask |= path.contains_points(grid).reshape(h, w)
            if img_array.ndim == 3:
                masked = img_array.copy()
                masked[~mask] = 0
                display = masked
            else:
                masked = img_array.copy()
                masked[~mask] = np.nan
                display = masked
        except Exception:
            pass
    kw = {}
    if vmin is not None: kw["vmin"] = vmin
    if vmax is not None: kw["vmax"] = vmax
    if cmap:
        im = ax.imshow(display, cmap=cmap, **kw)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(label, fontsize=9)
    else:
        ax.imshow(display, **kw)
    if polygon and fetch_bbox:
        try:
            lon_min, lat_min, lon_max, lat_max = fetch_bbox
            h, w = img_array.shape[:2]
            for ring in polygon["geometry"]["coordinates"]:
                xs = [(p[0] - lon_min) / (lon_max - lon_min) * w for p in ring]
                ys = [(lat_max - p[1]) / (lat_max - lat_min) * h for p in ring]
                ax.plot(xs, ys, 'r-', linewidth=1.5)
        except Exception:
            pass
    if targets and fetch_bbox:
        try:
            lon_min, lat_min, lon_max, lat_max = fetch_bbox
            h, w = img_array.shape[:2]
            for t in targets:
                tx = (t["lon"] - lon_min) / (lon_max - lon_min) * w
                ty = (lat_max - t["lat"]) / (lat_max - lat_min) * h
                color = {"HIGH": "red", "MEDIUM": "orange", "LOW": "lime"}.get(t["priority"], "white")
                r = max(3, int(t.get("radius_m", 200) / 50))
                circle = plt.Circle((tx, ty), r, color=color, fill=False, linewidth=1.5)
                ax.add_patch(circle)
                ax.plot(tx, ty, 'o', color=color, markersize=3)
        except Exception:
            pass
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")
    return _fig_to_bytes(fig)


class SatIntelPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.author_name = ""
        self.report_title = "SatIntel Report"
        self.set_margins(15, 15, 15)  # Left, Top, Right margins = 15mm
        self.set_auto_page_break(True, 20)  # Auto page break with 20mm bottom margin
        self.add_font('DejaVu', '', FONT_REGULAR)
        self.add_font('DejaVu', 'B', FONT_BOLD)
        self.add_font('DejaVuMono', '', FONT_MONO)
        self.add_font('DeVuMono', 'B', FONT_MONO_BOLD)
        self.add_font('DejaVu', 'I', FONT_ITALIC)
        self.add_font('DejaVu', 'BI', FONT_BOLD_ITALIC)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font('DejaVu', 'B', 9)
        self.set_text_color(*COLOR_MEDIUM_GRAY)
        self.cell(0, 6, self.report_title, 0, 1, 'L')
        self.set_draw_color(180, 180, 180)
        self.line(15, 18, 195, 18)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font('DejaVu', '', 7)
        self.set_text_color(*COLOR_MEDIUM_GRAY)
        page = f'Página {self.page_no()}/{{nb}}'
        if self.author_name:
            page += f'  |  {self.author_name}'
        page += '  |  SatIntel AI'
        self.cell(0, 8, page, 0, 0, 'C')

    def _colored_band(self, y, height, color):
        self.set_fill_color(*color)
        self.rect(0, y, 210, height, 'F')

    def _section_header(self, text, color=COLOR_DARK_GREEN):
        self.set_fill_color(*color)
        self.set_text_color(*COLOR_WHITE)
        self.set_font('DejaVu', 'B', 11)
        self.ln(2)
        self.cell(0, 8, text, 0, 1, 'L', fill=True)
        self.set_text_color(*COLOR_DARK_GRAY)
        self.ln(2)

    def _info_box(self, label, value):
        self.set_font('DejaVu', 'B', 9)
        self.set_text_color(*COLOR_DARK_GREEN)
        self.cell(60, 5, label)
        self.set_font('DejaVu', '', 9)
        self.set_text_color(*COLOR_DARK_GRAY)
        self.cell(0, 5, str(value), 0, 1)



# Bing Maps static image fetcher (no API key needed for basic tiles via direct URL)
import urllib.request

def _fetch_bing_hybrid_image(lat, lon, zoom=14, size="640x640"):
    """Fetch a Bing Maps hybrid (satellite + labels) static image.
    Uses the public Bing Maps tile endpoint."""
    try:
        # Bing Maps Static Imagery via the tile system
        # We'll use the ArcGIS REST services World Imagery as fallback (also hybrid-capable)
        # ArcGIS World Imagery + Reference layers
        import io as _io

        # Use ArcGIS World Imagery (free, no key needed)
        # Calculate bbox from lat/lon + zoom level
        import math
        earth_circumference = 40075016.686
        resolution = 156543.03 * math.cos(math.radians(lat)) / (2 ** zoom)
        half_width_deg = (size.split('x')[0] if 'x' in str(size) else 640) * resolution / 111320 / 2
        half_height_deg = (size.split('x')[1] if 'x' in str(size) else 640) * resolution / 111320 / 2

        lon_min = lon - half_width_deg
        lon_max = lon + half_width_deg
        lat_min = lat - half_height_deg
        lat_max = lat + half_height_deg

        # ArcGIS World Imagery (satellite)
        w = int(str(size).split('x')[0]) if 'x' in str(size) else 640
        h = int(str(size).split('x')[1]) if 'x' in str(size) else 640

        img_url = (
            f"https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
            f"?f=image&bbox={lon_min},{lat_min},{lon_max},{lat_max}"
            f"&size={w},{h}&dpi=200&format=png32&imageSR=4326&bboxSR=4326"
        )
        ref_url = (
            f"https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/export"
            f"?f=image&bbox={lon_min},{lat_min},{lon_max},{lat_max}"
            f"&size={w},{h}&dpi=200&format=png32&imageSR=4326&bboxSR=4326&transparent=true"
        )

        # Fetch satellite image
        req1 = urllib.request.Request(img_url, headers={'User-Agent': 'Mozilla/5.0'})
        img_data = urllib.request.urlopen(req1, timeout=25).read()

        # Fetch reference overlay (boundaries + labels = "hybrid" look)
        try:
            req2 = urllib.request.Request(ref_url, headers={'User-Agent': 'Mozilla/5.0'})
            ref_data = urllib.request.urlopen(req2, timeout=15).read()
        except Exception:
            ref_data = None

        # Combine satellite + reference overlay
        from PIL import Image
        base_img = Image.open(_io.BytesIO(img_data)).convert("RGBA")
        if ref_data:
            ref_img = Image.open(_io.BytesIO(ref_data)).convert("RGBA")
            base_img = Image.alpha_composite(base_img, ref_img)

        # Draw the license polygon boundary on top
        # This is done by the caller who has polygon + bbox info
        return base_img, (lon_min, lat_min, lon_max, lat_max)
    except Exception as e:
        return None, None


def _draw_polygon_on_image(img, polygon, img_bbox, img_size=(640, 640)):
    """Draw license polygon boundary on the satellite image."""
    from PIL import ImageDraw
    import io as _io

    if img is None or polygon is None or img_bbox is None:
        return img

    lon_min, lat_min, lon_max, lat_max = img_bbox
    w, h = img_size

    # Convert polygon coordinates to pixel coordinates
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for ring in polygon.get("geometry", {}).get("coordinates", []):
        pixels = []
        for p in ring:
            px = int((p[0] - lon_min) / (lon_max - lon_min) * w)
            py = int((lat_max - p[1]) / (lat_max - lat_min) * h)
            pixels.append((px, py))
        if len(pixels) >= 2:
            # Draw bright yellow boundary line
            draw.line(pixels + [pixels[0]], fill=(255, 215, 0, 255), width=3)

    return Image.alpha_composite(img, overlay)


def _make_bing_cover_image(polygon, fetch_bbox, map_center, targets=None):
    """Create a hybrid satellite cover image with license boundary overlay."""
    try:
        lat = map_center[0] if map_center else -19.0
        lon = map_center[1] if map_center else 33.0

        # Determine zoom based on concession size
        if fetch_bbox:
            lon_range = fetch_bbox[2] - fetch_bbox[0]
            lat_range = fetch_bbox[3] - fetch_bbox[1]
            max_range = max(lon_range, lat_range)
            if max_range > 1.0:
                zoom = 10
            elif max_range > 0.5:
                zoom = 11
            elif max_range > 0.2:
                zoom = 12
            elif max_range > 0.1:
                zoom = 13
            else:
                zoom = 14
        else:
            zoom = 13

        img, img_bbox = _fetch_bing_hybrid_image(lat, lon, zoom=zoom, size="800x600")
        if img is None:
            return None

        # Draw the license polygon boundary
        if polygon:
            img = _draw_polygon_on_image(img, polygon, img_bbox, img_size=(800, 600))

        # Draw targets as red circles
        if targets and img_bbox:
            from PIL import ImageDraw
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            lon_min, lat_min, lon_max, lat_max = img_bbox
            w, h = 800, 600
            for t in targets:
                tlon, tlat = t.get("lon", 0), t.get("lat", 0)
                px = int((tlon - lon_min) / (lon_max - lon_min) * w)
                py = int((lat_max - tlat) / (lat_max - lat_min) * h)
                r = 8
                color = (255, 50, 50, 255) if t.get("priority") == "HIGH" else (255, 165, 0, 255) if t.get("priority") == "MEDIUM" else (50, 200, 50, 255)
                draw.ellipse([px-r, py-r, px+r, py+r], fill=color, outline=(255, 255, 255, 255), width=2)
            img = Image.alpha_composite(img, overlay)

        # Convert to bytes for PDF insertion
        buf = _io.BytesIO()
        img.convert("RGB").save(buf, format="PNG", dpi=(200, 200))
        buf.seek(0)
        return buf
    except Exception:
        return None


def generate_professional_report(
    report_text=None, sat_data=None, m_data=None, meta=None,
    targets=None, polygon=None, fetch_bbox=None, author_info=None,
    selected_year=None, target_commodity=None, map_center=None,
):
    pdf = SatIntelPDF()
    pdf.alias_nb_pages()

    author = author_info.get("author", "N/A")
    pdf.author_name = author

    conces_name = meta.get('Nome da Concessão', meta.get('Nome da Concessao', 'Concessão'))
    lic_code = meta.get('Código da Licença (Code)', meta.get('Codigo da Licenca (Code)', 'N/A'))
    pdf.report_title = f"Parecer Técnico — {conces_name}"

    # === PAGE 1: COVER with satellite image + license info ===
    pdf.add_page()
    pdf._colored_band(0, 35, COLOR_DARK_GREEN)
    pdf.set_xy(0, 8)
    pdf.set_font('DejaVu', 'B', 24)
    pdf.set_text_color(*COLOR_WHITE)
    pdf.cell(0, 14, 'SATINTEL', 0, 1, 'C')
    pdf.set_font('DejaVu', '', 9)
    pdf.set_xy(0, 24)
    pdf.cell(0, 5, 'Geological & Mining Insights Platform', 0, 1, 'C')
    pdf.set_font('DejaVu', '', 7)
    pdf.set_xy(0, 30)
    pdf.cell(0, 4, 'Relatório Técnico de Exploração Mineral', 0, 1, 'C')

    classification = author_info.get("classification", "Confidencial — Uso Interno")
    pdf.ln(3)
    pdf.set_font('DejaVu', 'B', 8)
    pdf.set_text_color(*COLOR_RED)
    pdf.cell(0, 5, classification, 0, 1, 'C')

    pdf.ln(3)
    # ── Cover image: Bing/ArcGIS hybrid satellite with license boundary ──
    hybrid_img = _make_bing_cover_image(polygon, fetch_bbox, map_center, targets)
    if hybrid_img:
        img_w = 175
        x_center = (210 - img_w) / 2
        pdf.image(hybrid_img, x=x_center, w=img_w)
        pdf.set_font('DejaVu', '', 7)
        pdf.set_text_color(*COLOR_MEDIUM_GRAY)
        pdf.ln(2)
        pdf.cell(0, 4, "Imagem Híbrida (Satélite + Limites) — ArcGIS World Imagery | Fronteira da Concessão a Amarelo", 0, 1, 'C')
    elif sat_data:
        sat_img = _make_index_image(
            sat_data.get("rgb"), title="Landsat Natural Color Composite",
            polygon=polygon, fetch_bbox=fetch_bbox, targets=targets)
        if sat_img:
            img_w = 160
            x_center = (210 - img_w) / 2
            pdf.image(sat_img, x=x_center, w=img_w)
            pdf.set_font('DejaVu', '', 7)
            pdf.set_text_color(*COLOR_MEDIUM_GRAY)
            pdf.ln(2)
            sat_label = f"Imagem Landsat — {sat_data.get('Satellite_Used', 'N/A')} | Cena: {sat_data.get('scene_date', 'N/A')} | Nuvens: {sat_data.get('cloud_cover', 'N/A')}%"
            pdf.cell(0, 4, sat_label, 0, 1, 'C')
    else:
        pdf.set_fill_color(*COLOR_LIGHT_GRAY)
        pdf.set_xy(25, 50)
        pdf.rect(25, 50, 160, 80, 'F')
        pdf.set_font('DejaVu', 'I', 10)
        pdf.set_text_color(*COLOR_MEDIUM_GRAY)
        pdf.set_xy(25, 85)
        pdf.cell(160, 5, "Imagem de satélite não disponível (dados preditivos)", 0, 1, 'C')

    pdf.ln(5)
    pdf.set_font('DejaVu', 'B', 14)
    pdf.set_text_color(*COLOR_DARK_GREEN)
    pdf.cell(0, 8, conces_name, 0, 1, 'C')
    pdf.set_font('DejaVu', '', 11)
    pdf.set_text_color(*COLOR_DARK_GRAY)
    pdf.cell(0, 6, f"Licença Nº {lic_code}", 0, 1, 'C')
    pdf.ln(2)

    pdf.set_draw_color(*COLOR_DARK_GREEN)
    pdf.set_line_width(0.3)
    meta_fields = [
        ("Titular", meta.get('Titular (Holder Company)', 'N/A')),
        ("Área / Dimensão", meta.get('Área / Dimensão', meta.get('Area / Dimensao', 'N/A'))),
        ("Substâncias", meta.get('Substâncias', meta.get('Substancias', 'N/A'))),
        ("Estado", meta.get('Estado (Status)', 'N/A')),
        ("Validade", meta.get('Data de Validade (Expiry)', 'N/A')),
        ("Coordenadas", f"{map_center[0]:.4f}, {map_center[1]:.4f}" if map_center else "N/A"),
    ]

    pdf.set_font('DejaVu', '', 8)
    for i, (label, value) in enumerate(meta_fields):
        if i % 2 == 0:
            pdf.set_fill_color(*COLOR_LIGHT_GRAY)
        else:
            pdf.set_fill_color(*COLOR_WHITE)
        pdf.set_text_color(*COLOR_DARK_GREEN)
        pdf.set_font('DejaVu', 'B', 8)
        pdf.cell(42, 5, label, border=0, fill=True)
        pdf.set_font('DejaVu', '', 8)
        pdf.set_text_color(*COLOR_DARK_GRAY)
        val_str = str(value)[:75]
        pdf.cell(138, 5, val_str, border=0, fill=True, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(3)
    pdf._section_header("RESPONSÁVEL TÉCNICO")
    author_fields = [
        ("Prepared by", author),
        ("Cargo", author_info.get("title", "N/A")),
        ("Empresa", author_info.get("company", "N/A")),
        ("Licença Prof.", author_info.get("license_no", "N/A")),
        ("Nº Relatório", author_info.get("ref_no", "N/A")),
        ("Data", author_info.get("date", "N/A")),
    ]
    pdf.set_font('DejaVu', '', 9)
    for label, value in author_fields:
        pdf._info_box(label, value)

    pdf.ln(5)
    pdf.set_draw_color(*COLOR_MEDIUM_GRAY)
    pdf.set_line_width(0.2)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(2)
    pdf.set_font('DejaVu', 'I', 7)
    pdf.set_text_color(*COLOR_MEDIUM_GRAY)
    pdf.multi_cell(180, 4,
        "Este documento foi gerado pelo SatIntel AI utilizando dados Landsat (USGS/NASA) e IBM watsonx.ai. "
        "O conteúdo técnico deve ser validado por trabalho de campo antes da tomada de decisão.")

    # === SPECTRAL INDEX PAGES ===
    if sat_data:
        indices = [
            {"key": "rgb", "title": "Natural Color Composite", "cmap": None, "label": "",
             "explanation": "Composto de cor natural Landsat (bandas 4-3-2). Mostra a aparência real da superfície terrestre, útil para identificação visual de feições geográficas, vegetação, corpos d'água e infraestrutura.",
             "score_key": None, "score_label": None},
            {"key": "false_color", "title": "Mineral Enhancement Composite", "cmap": None, "label": "",
             "explanation": "Composto de cor falsa (bandas 7-5-3) que realça minerais de alteração hidrotermal. Tons magenta/brancos indicam potenciais zonas de alteração argílica e sericítica.",
             "score_key": None, "score_label": None},
            {"key": "iron_oxide_map", "title": "Iron Oxide Ratio (B4/B2)", "cmap": "RdYlBu_r", "label": "Fe-Oxide",
             "explanation": "Razão de bandas Vermelho/Azul (B4/B2) realça óxidos de ferro (gossan). Valores elevados (vermelho) indicam zonas de oxidação e alteração ferruginosa — indicador chave para depósitos de ouro associados a sulfetos oxidados.",
             "score_key": "Way_1_Iron_Oxide_Gossan", "score_label": "Iron Oxide (Gossan)"},
            {"key": "clay_map", "title": "Clay/Hydroxyl Ratio (B6/B7)", "cmap": "YlOrBr", "label": "Clay",
             "explanation": "Razão SWIR1/SWIR2 (B6/B7) realça minerais argilosos e hidroxilos (caulim, alunito, sericita). Zonas de anomalia indicam alteração hidrotermal — importante para exploração de ouro epitermal e pórfiro.",
             "score_key": "Way_1_Clay_Phyllic", "score_label": "Clay/Hydroxyl Index"},
            {"key": "ndvi_map", "title": "NDVI — Normalized Difference Vegetation Index", "cmap": "RdYlGn",
             "vmin": -0.3, "vmax": 0.8, "label": "NDVI",
             "explanation": "Índice de Vegetação por Diferença Normalizada. Valores positivos (verde) indicam vegetação saudável; valores negativos (vermelho) indicam solo exposto ou rocha. Pode revelar estresse geobotânico associado a mineralização.",
             "score_key": "Way_4_Geobotanical_Stress", "score_label": "Geobotanical Stress"},
            {"key": "silica_map", "title": "Silica Proxy (B7/B6)", "cmap": "bone", "label": "Silica",
             "explanation": "Proxy de sílica usando SWIR2/SWIR1. Zonas brilhantes podem indicar silicificação — um importante guia para mineralização de ouro, especialmente em veios de quartzo.",
             "score_key": "Way_3_Silica_Flooding_Cap", "score_label": "Silica Flooding"},
            {"key": "crosta_iron_pca", "title": f"Crosta PCA — Iron Oxide (PC{sat_data.get('crosta_iron_pc', 0)+1})",
             "cmap": "RdYlBu_r", "label": "PC Score",
             "explanation": f"Análise de Componentes Principais orientada (Crosta) para óxido de ferro. PC{sat_data.get('crosta_iron_pc', 0)+1} selecionado pela maior contraste Red-vs-Blue. Cobertura de anomalia: {sat_data.get('crosta_iron_anomaly_pct', 0)}%. Isola o sinal de óxido de ferro eliminando ruído espectral.",
             "score_key": "crosta_iron_anomaly_pct", "score_label": "Iron Oxide Anomaly", "score_suffix": "%"},
            {"key": "crosta_clay_pca", "title": f"Crosta PCA — Clay/Hydroxyl (PC{sat_data.get('crosta_clay_pc', 0)+1})",
             "cmap": "YlOrBr", "label": "PC Score",
             "explanation": f"Análise de Componentes Principais orientada (Crosta) para argila/hidroxilo. PC{sat_data.get('crosta_clay_pc', 0)+1} selecionado pela maior contraste SWIR1-vs-SWIR2. Cobertura de anomalia: {sat_data.get('crosta_clay_anomaly_pct', 0)}%. Isola o sinal de alteração argílica eliminando contribuições de outras coberturas.",
             "score_key": "crosta_clay_anomaly_pct", "score_label": "Clay Alteration Anomaly", "score_suffix": "%"},
            {"key": "lineament_density_map", "title": "Lineament Density Map", "cmap": "hot", "label": "Density",
             "explanation": f"Mapa de densidade de lineamentos estruturais derivado de filtragem Sobel direcional (4 orientações: N-S, E-W, NE-SW, NW-SE) com suavização Gaussiana. Índice de densidade: {sat_data.get('lineament_density_val', 0)}. Zonas quentes indicam alto controle estrutural — favorável para permineralização.",
             "score_key": "lineament_density_val", "score_label": "Lineament Density Index"},
            {"key": "intersection_map", "title": "Lineament Intersection Map", "cmap": "magma", "label": "Intersection",
             "explanation": f"Mapa de interseções de lineamentos de diferentes orientações. Interseções de alta confiança: {sat_data.get('intersection_count', 0)}. Índice de densidade de interseção: {sat_data.get('intersection_density_val', 0)}. Nós de interseção são zonas de alta prospectividade para ouro (dilatância estrutural).",
             "score_key": "intersection_density_val", "score_label": "Intersection Density Index"},
        ]

        for idx_def in indices:
            key = idx_def["key"]
            if key not in sat_data or sat_data[key] is None:
                continue

            pdf.add_page()
            pdf.set_font('DejaVu', 'B', 14)
            pdf.set_text_color(*COLOR_DARK_GREEN)
            pdf.cell(0, 8, idx_def["title"], 0, 1)
            pdf.set_draw_color(*COLOR_DARK_GREEN)
            pdf.set_line_width(0.5)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(3)

            img_bytes = _make_index_image(
                sat_data[key], cmap=idx_def.get("cmap"),
                vmin=idx_def.get("vmin"), vmax=idx_def.get("vmax"),
                title=idx_def["title"], polygon=polygon,
                fetch_bbox=fetch_bbox, targets=targets,
                label=idx_def.get("label", ""))
            if img_bytes:
                img_w = 150
                x_center = (210 - img_w) / 2
                pdf.image(img_bytes, x=x_center, w=img_w)

            pdf.ln(4)

            score_key = idx_def.get("score_key")
            if score_key and m_data:
                score_val = m_data.get(score_key, sat_data.get(score_key, "N/A"))
                score_label = idx_def.get("score_label", "")
                score_suffix = idx_def.get("score_suffix", "")

                pdf._section_header("VALOR DE CÁLCULO", COLOR_DARK_GREEN)
                pdf.set_font('DejaVu', 'B', 24)
                pdf.set_text_color(*COLOR_ACCENT_GOLD)
                score_str = f"{score_val}{score_suffix}" if score_val != "N/A" else "N/A"
                pdf.cell(0, 14, str(score_str), 0, 1)
                pdf.set_font('DejaVu', 'B', 9)
                pdf.set_text_color(*COLOR_DARK_GREEN)
                pdf.cell(0, 5, score_label, 0, 1)
                pdf.ln(2)

            pdf._section_header("INTERPRETAÇÃO GEOLÓGICA", COLOR_DARK_GREEN)
            pdf.set_font('DejaVu', '', 9)
            pdf.set_text_color(*COLOR_DARK_GRAY)
            pdf.multi_cell(180, 5, idx_def["explanation"])

            if key == "crosta_iron_pca":
                loadings = sat_data.get("crosta_iron_loadings", {})
                if loadings:
                    pdf.ln(2)
                    pdf._section_header("EIGENVECTOR LOADINGS", COLOR_DARK_GREEN)
                    pdf.set_font('DejaVuMono', '', 8)
                    pdf.set_text_color(*COLOR_DARK_GRAY)
                    for band, loading in loadings.items():
                        pdf.cell(0, 5, f"  {band:8s}: {loading:+.4f}", 0, 1)

            if key == "crosta_clay_pca":
                loadings = sat_data.get("crosta_clay_loadings", {})
                if loadings:
                    pdf.ln(2)
                    pdf._section_header("EIGENVECTOR LOADINGS", COLOR_DARK_GREEN)
                    pdf.set_font('DejaVuMono', '', 8)
                    pdf.set_text_color(*COLOR_DARK_GRAY)
                    for band, loading in loadings.items():
                        pdf.cell(0, 5, f"  {band:8s}: {loading:+.4f}", 0, 1)

    # === WATSONX REPORT ===
    if report_text:
        pdf.add_page()
        pdf._section_header("PARECER TÉCNICO — IBM WATSONX.AI", COLOR_DARK_GREEN)
        clean_text = report_text.replace('**', '').replace('*', '').replace('#', '')
        clean_text = clean_text.replace('\r\n', '\n').replace('\r', '\n')
        lines = clean_text.split('\n')
        for line in lines:
            stripped = line.strip()
            if not stripped:
                pdf.ln(3)
                continue
            is_header = any(stripped.startswith(f"{i}.") for i in range(1, 10))
            if is_header and len(stripped) < 120:
                pdf.ln(3)
                pdf.set_font('DejaVu', 'B', 12)
                pdf.set_text_color(*COLOR_DARK_GREEN)
                pdf.multi_cell(180, 6, stripped)
                pdf.set_text_color(*COLOR_DARK_GRAY)
                pdf.ln(1)
            elif stripped.startswith("- ") or stripped.startswith("* "):
                pdf.set_font('DejaVu', '', 9)
                pdf.set_text_color(*COLOR_DARK_GRAY)
                pdf.multi_cell(180, 5, "  • " + stripped[2:])
            elif "|" in stripped and stripped.count("|") >= 2:
                pdf.set_font('DejaVuMono', '', 8)
                pdf.set_text_color(*COLOR_DARK_GRAY)
                pdf.multi_cell(180, 5, stripped)
            else:
                pdf.set_font('DejaVu', '', 9)
                pdf.set_text_color(*COLOR_DARK_GRAY)
                pdf.multi_cell(180, 5, stripped)

    # === TARGETS SUMMARY ===
    if targets:
        pdf.add_page()
        pdf._section_header("RESUMO DE ALVOS DE EXPLORAÇÃO", COLOR_DARK_GREEN)
        pdf.set_font('DejaVu', '', 9)
        pdf.set_text_color(*COLOR_DARK_GRAY)
        pdf.multi_cell(180, 5,
            "Composite score: IO(0.20) + CLAY(0.20) + Structural(0.15) + "
            "Geomorphology(0.30) + Lineament(0.15)")
        pdf.ln(3)

        high_c = sum(1 for t in targets if t["priority"] == "HIGH")
        med_c = sum(1 for t in targets if t["priority"] == "MEDIUM")
        low_c = sum(1 for t in targets if t["priority"] == "LOW")

        pdf.set_font('DejaVu', 'B', 9)
        pdf.set_fill_color(*COLOR_LIGHT_GREEN)
        pdf.cell(50, 7, f"Total: {len(targets)}", 0, 0, fill=True)
        pdf.set_fill_color(255, 230, 230)
        pdf.cell(50, 7, f"Alta: {high_c}", 0, 0, fill=True)
        pdf.set_fill_color(255, 240, 220)
        pdf.cell(50, 7, f"Média: {med_c}", 0, 0, fill=True)
        pdf.set_fill_color(230, 255, 230)
        pdf.cell(40, 7, f"Baixa: {low_c}", 0, 1, fill=True)
        pdf.ln(3)

        pdf.set_font('DejaVu', 'B', 7)
        pdf.set_fill_color(*COLOR_DARK_GREEN)
        pdf.set_text_color(*COLOR_WHITE)
        pdf.cell(12, 6, "ID", 1, 0, 'C', True)
        pdf.cell(15, 6, "Score", 1, 0, 'C', True)
        pdf.cell(18, 6, "Priority", 1, 0, 'C', True)
        pdf.cell(50, 6, "Structural Control", 1, 0, 'C', True)
        pdf.cell(60, 6, "Lithology", 1, 0, 'C', True)
        pdf.cell(35, 6, "Radius", 1, 1, 'C', True)

        pdf.set_text_color(*COLOR_DARK_GRAY)
        for t in targets:
            pdf.set_font('DejaVu', '', 7)
            if t["priority"] == "HIGH":
                pdf.set_fill_color(255, 235, 235)
            elif t["priority"] == "MEDIUM":
                pdf.set_fill_color(255, 245, 230)
            else:
                pdf.set_fill_color(235, 250, 235)
            pdf.cell(12, 5, str(t["id"]), 1, 0, 'C', True)
            pdf.cell(15, 5, f"{t['score']}%", 1, 0, 'C', True)
            pdf.cell(18, 5, str(t["priority"]), 1, 0, 'C', True)
            pdf.cell(50, 5, str(t["structural_control"])[:35], 1, 0, 'L', True)
            pdf.cell(60, 5, str(t["lithology"])[:40], 1, 0, 'L', True)
            pdf.cell(35, 5, f"~{t['radius_m']}m", 1, 1, 'C', True)

        pdf.ln(4)
        pdf.set_font('DejaVu', 'B', 9)
        pdf.set_text_color(*COLOR_DARK_GREEN)
        pdf.cell(0, 6, "Descrições Detalhadas", 0, 1)
        pdf.ln(2)

        for t in targets:
            priority_color = {"HIGH": COLOR_RED, "MEDIUM": COLOR_ORANGE, "LOW": (0, 128, 0)}.get(t["priority"], COLOR_MEDIUM_GRAY)
            pdf.set_font('DejaVu', 'B', 9)
            pdf.set_text_color(*priority_color)
            pdf.cell(0, 5, f"  {t['id']} — {t['lithology']} ({t['priority']})", 0, 1)
            pdf.set_font('DejaVu', '', 7)
            pdf.set_text_color(*COLOR_MEDIUM_GRAY)
            pdf.cell(0, 4, f"    Lat: {t['lat']:.4f}, Lon: {t['lon']:.4f} | Raio: ~{t['radius_m']}m", 0, 1)
            pdf.set_font('DejaVu', '', 7)
            pdf.set_text_color(*COLOR_DARK_GRAY)
            pdf.multi_cell(180, 4, f"    EN: {t['description_en']}")
            pdf.multi_cell(180, 4, f"    PT: {t['description_pt']}")
            pdf.set_font('DejaVuMono', '', 6)
            pdf.set_text_color(*COLOR_MEDIUM_GRAY)
            pdf.cell(0, 4, f"    IO={t['io_score']} Clay={t['clay_score']} Struct={t['struct_score']} Geo={t['geomorph_score']} Line={t['line_score']}", 0, 1)
            pdf.ln(1)

    # === SIGNATURE PAGE ===
    pdf.add_page()
    pdf.ln(60)
    pdf.set_font('DejaVu', '', 9)
    pdf.set_text_color(*COLOR_MEDIUM_GRAY)
    pdf.cell(0, 5, "Documento gerado em conformidade com as práticas de exploração mineral", 0, 1, 'C')
    pdf.cell(0, 5, "e análise de sensoriamento remoto para concessões em Moçambique.", 0, 1, 'C')
    pdf.ln(10)
    pdf.set_draw_color(*COLOR_DARK_GRAY)
    pdf.set_line_width(0.4)
    sig_y = pdf.get_y()
    pdf.line(65, sig_y, 145, sig_y)
    pdf.ln(3)
    pdf.set_font('DejaVu', 'B', 10)
    pdf.set_text_color(*COLOR_DARK_GRAY)
    pdf.cell(0, 5, author, 0, 1, 'C')
    pdf.set_font('DejaVu', '', 8)
    pdf.set_text_color(*COLOR_MEDIUM_GRAY)
    pdf.cell(0, 5, author_info.get("title", ""), 0, 1, 'C')
    pdf.cell(0, 5, author_info.get("company", ""), 0, 1, 'C')
    if author_info.get("license_no") and author_info["license_no"] != "N/A":
        pdf.cell(0, 5, f"Licença Prof.: {author_info['license_no']}", 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font('DejaVu', 'I', 7)
    pdf.set_text_color(*COLOR_MEDIUM_GRAY)
    pdf.multi_cell(180, 4,
        "DISCLAIMER: Este relatório foi gerado automaticamente pela plataforma SatIntel AI "
        "utilizando dados Landsat (USGS/NASA) processados com Crosta PCA, filtragem Sobel direcional, "
        "e análise de componentes principais. O parecer técnico foi gerado pelo modelo IBM watsonx.ai "
        "(meta-llama/llama-3-3-70b-instruct). Os resultados devem ser validados por trabalho de campo, "
        "amostragem geoquímica e sondagens antes de qualquer decisão de investimento.")

    return bytes(pdf.output())
