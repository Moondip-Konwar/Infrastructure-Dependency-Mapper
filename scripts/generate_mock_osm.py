#!/usr/bin/env python3
"""
Generate mock OpenStreetMap data in the Overpass API JSON format.
"""

import pathlib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "src" / "lib" / "data" / "osm_data.json"

METERS_PER_DEGREE_LATITUDE = 111_320.0
COORDINATE_DECIMAL_PLACES = 7
MAX_PLACEMENT_ATTEMPTS = 50

OSM_JSON_VERSION = 0.6
GENERATOR_NAME = "generate_mock_osm.py (mock data)"
MOCK_TIMESTAMP = "2026-01-01T00:00:00Z"
MOCK_COPYRIGHT = "mock data. not taken from OpenStreetMap."

MAXSPEED_KMH_BY_HIGHWAY = { "primary": "60", "secondary": "50", "tertiary": "40" }

BUILDING_TYPE_WEIGHTS = [
    ("house", 55),
    ("yes", 25),
    ("residential", 8),
    ("apartments", 5),
    ("commercial", 4),
    ("retail", 3)      # If i left something after this that should be here then tell me
]