# -*- coding: utf-8 -*-
"""
Shared configuration constants for ArchaeoTrace.
"""

PLUGIN_NAME = "ArchaeoTrace"
PLUGIN_MENU = f"&{PLUGIN_NAME}"
PLUGIN_TOOLBAR_OBJECT_NAME = f"{PLUGIN_NAME}Toolbar"

MODEL_IDX_CANNY = 0
MODEL_IDX_LSD = 1
MODEL_IDX_HED = 2
MODEL_IDX_SAM = 3

DEFAULT_EDGE_METHOD = "canny"
DEFAULT_SAM_MODEL_TYPE = "vit_t"

EDGE_METHOD_BY_MODEL = {
    MODEL_IDX_CANNY: "canny",
    MODEL_IDX_LSD: "lsd",
    MODEL_IDX_HED: "hed",
    MODEL_IDX_SAM: "canny",
}

MODE_NAME_BY_MODEL = {
    MODEL_IDX_CANNY: "Canny",
    MODEL_IDX_LSD: "LSD",
    MODEL_IDX_HED: "HED",
    MODEL_IDX_SAM: "SAM",
}

TRACE_BUTTON_IDLE_STYLE = "font-weight: bold; padding: 8px; background: #27ae60; color: white;"
TRACE_BUTTON_ACTIVE_STYLE = "font-weight: bold; padding: 8px; background: #e74c3c; color: white;"

DEFAULT_FREEDOM_SLIDER_VALUE = 30

DEFAULT_OUTPUT_LAYER_NAME = "Contours"
DEFAULT_SPOT_LAYER_NAME = "Spot Heights"
DEFAULT_CRS_AUTHID = "EPSG:4326"

MAX_RASTER_BANDS_FOR_RGB = 3
PREVIEW_EDGE_MAX_DIMENSION = 800
