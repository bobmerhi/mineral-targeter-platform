# unified_selector.py
# Centralized logic for switching between 5 Deposit Models
# Maintains Free-Data Baseline + Paid Upgrade Path architecture

DEPOSIT_MODELS = {
    "Orogenic Gold (Vein-Hosted)": {
        "id": "orogenic_gold",
        "description": "Shear-zone hosted quartz-carbonate veins. Primary control: Structure.",
        "wlc_weights": {"IO": 0.20, "Clay": 0.20, "Struct": 0.35, "Geomorph": 0.10, "Lineament": 0.15},
        "required_sensors": ["Landsat-8/9", "Sentinel-2", "AW3D30 DEM"],
        "option_b_path": None,
        "validation_focus": "Z-Score Anomaly (8-18%) + Structural Intersection Density"
    },
    "Epithermal Gold (HS/LS)": {
        "id": "epithermal_gold",
        "description": "Volcanic-hosted. HS: Alunite/Vuggy Silica. LS: Sericite/Adularia.",
        "wlc_weights": {"Alunite": 0.30, "Kaolinite": 0.25, "Quartz": 0.20, "Struct": 0.15, "Sericite": 0.10},
        "required_sensors": ["ASTER SWIR/TIR (Mandatory)"],
        "option_b_path": {
            "name": "Airborne Hyperspectral (HyMap/PRISMA) + Field PIMA",
            "cost_usd": 1200,
            "accuracy_gain": ">85% (Direct mineral chemistry vs <50% Landsat)",
            "delivery_weeks": 3
        },
        "validation_focus": "Alunite/Kaolinite Discrimination + Vertical Zoning"
    },
    "Placer Gold (Alluvial)": {
        "id": "placer_gold",
        "description": "Sediment-hosted heavy minerals. Primary control: Geomorphology.",
        "wlc_weights": {"HMI": 0.40, "FlowAccum": 0.30, "SlopeProxy": 0.20, "TRI": 0.10},
        "required_sensors": ["Sentinel-2 (10m)", "AW3D30/SRTM DEM"],
        "option_b_path": {
            "name": "Airborne LiDAR (Bare Earth) + Drone Magnetometry",
            "cost_usd_per_km2": 12,
            "accuracy_gain": ">85% (Sub-canopy paleochannels vs <40% optical)",
            "delivery_weeks": 2
        },
        "validation_focus": "Heavy Mineral Index (B11/B8) + Paleochannel Reconstruction"
    },
    "Copper Porphyry": {
        "id": "copper_porphyry",
        "description": "Intrusion-related concentric zoning. Phyllic zone = highest potential.",
        "wlc_weights": {"Phyllic": 0.25, "Quartz_TIR": 0.20, "Argillic": 0.20, "Propylitic": 0.15, "IO_Gossan": 0.20},
        "required_sensors": ["ASTER SWIR/TIR (Logical Operators)"],
        "option_b_path": {
            "name": "Airborne Radiometrics (K-Th-U) + Ground MT/TEM",
            "cost_usd_per_km2": 25,
            "accuracy_gain": ">85% (Blind potassic core detection vs <40% optical)",
            "delivery_weeks": 4
        },
        "validation_focus": "Phyllic Zone Identification + Concentric Zoning Pattern"
    },
    "Reduced Intrusion-Related (RIR)": {
        "id": "rir_gold",
        "description": "Granite-hosted sheeted veins. Au-Bi-Te-As signature.",
        "wlc_weights": {"IO": 0.15, "Clay": 0.15, "Struct": 0.30, "Geomorph": 0.10, "Lineament": 0.10, "K-Feldspar": 0.20},
        "required_sensors": ["Landsat-8/9", "Sentinel-2 Lithology"],
        "option_b_path": {
            "name": "Ground IP/Resistivity + Airborne Magnetics",
            "cost_usd_per_km2": 15,
            "accuracy_gain": "Maps subsurface sulfide chargeability and granitic contacts",
            "delivery_weeks": 3
        },
        "validation_focus": "Granite Contact Zones + Bi/Te Pathfinder Geochemistry"
    }
}

def get_model_config(selected_mode_name):
    """Returns the full configuration dictionary for the selected mode."""
    return DEPOSIT_MODELS.get(selected_mode_name, DEPOSIT_MODELS["Orogenic Gold (Vein-Hosted)"])

def check_sensor_availability(model_config, sat_data):
    """
    Checks if required sensors are present in the current sat_data dict.
    Returns (is_available, missing_sensors_list)
    """
    required = model_config["required_sensors"]
    missing = []
    
    if "ASTER SWIR/TIR (Mandatory)" in required:
        if "alunite_map" not in sat_data and "phyllic_map" not in sat_data:
            missing.append("ASTER SWIR/TIR Data")
            
    if "AW3D30/SRTM DEM" in required or "AW3D30 DEM" in required:
        if "tri_map" not in sat_data and "flow_accum_map" not in sat_data:
            missing.append("High-Resolution DEM Data")
            
    if "Sentinel-2 (10m)" in required:
        if "hmi_map" not in sat_data:
            missing.append("Sentinel-2 High-Res Data")
            
    return len(missing) == 0, missing
