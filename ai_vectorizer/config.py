# -*- coding: utf-8 -*-
"""
Shared configuration constants for ArchaeoTrace.
"""

PLUGIN_NAME = "ArchaeoTrace"
PLUGIN_MENU = f"&{PLUGIN_NAME}"
PLUGIN_TOOLBAR_OBJECT_NAME = f"{PLUGIN_NAME}Toolbar"
SETTINGS_LANG_KEY = f"{PLUGIN_NAME}/language"

MODEL_IDX_CANNY = 0
MODEL_IDX_LSD = 1
MODEL_IDX_HED = 2
MODEL_IDX_MOBILE_SAM = 3
MODEL_IDX_SAM = 4

DEFAULT_EDGE_METHOD = "canny"
SAM_BACKEND_MOBILE = "mobile_sam"
SAM_BACKEND_FULL = "segment_anything"
DEFAULT_MOBILE_SAM_MODEL_TYPE = "vit_t"
DEFAULT_FULL_SAM_MODEL_TYPE = "vit_b"
DEFAULT_SAM_MODEL_TYPE = DEFAULT_MOBILE_SAM_MODEL_TYPE
SAM_ASSIST_EDGE_METHOD = "canny"
SAM_MODEL_INDICES = (MODEL_IDX_MOBILE_SAM, MODEL_IDX_SAM)

SAM_ENGINE_SPEC_BY_MODEL = {
    MODEL_IDX_MOBILE_SAM: {
        "backend": SAM_BACKEND_MOBILE,
        "model_type": DEFAULT_MOBILE_SAM_MODEL_TYPE,
    },
    MODEL_IDX_SAM: {
        "backend": SAM_BACKEND_FULL,
        "model_type": DEFAULT_FULL_SAM_MODEL_TYPE,
    },
}

EDGE_METHOD_BY_MODEL = {
    MODEL_IDX_CANNY: "canny",
    MODEL_IDX_LSD: "lsd",
    MODEL_IDX_HED: "hed",
}

MODE_NAME_BY_MODEL = {
    MODEL_IDX_CANNY: "Canny",
    MODEL_IDX_LSD: "LSD",
    MODEL_IDX_HED: "HED",
    MODEL_IDX_MOBILE_SAM: "MobileSAM",
    MODEL_IDX_SAM: "SAM",
}

MODEL_MENU_LABELS = {
    MODEL_IDX_CANNY: {
        "ko": "🔧 OpenCV Canny (기본)",
        "en": "🔧 OpenCV Canny (Default)",
    },
    MODEL_IDX_LSD: {
        "ko": "📐 LSD 선분검출 (빠름)",
        "en": "📐 LSD Line Detector (Fast)",
    },
    MODEL_IDX_HED: {
        "ko": "🧠 HED 딥러닝 (매끄러움)",
        "en": "🧠 HED Deep Learning (Smooth)",
    },
    MODEL_IDX_MOBILE_SAM: {
        "ko": "🎯 MobileSAM (고품질)",
        "en": "🎯 MobileSAM (High Quality)",
    },
    MODEL_IDX_SAM: {
        "ko": "🧩 SAM (정밀)",
        "en": "🧩 SAM (Precise)",
    },
}

TRACE_BUTTON_IDLE_STYLE = "font-weight: bold; padding: 8px; background: #27ae60; color: white;"
TRACE_BUTTON_ACTIVE_STYLE = "font-weight: bold; padding: 8px; background: #e74c3c; color: white;"
STATUS_STYLE_READY = "color: green; font-weight: bold;"
STATUS_STYLE_NEUTRAL = ""
STATUS_STYLE_INFO = "color: green; font-size: 10px;"
STATUS_STYLE_WARNING = "color: orange; font-size: 10px;"
STATUS_STYLE_ERROR = "color: red; font-size: 10px;"

DEFAULT_FREEDOM_SLIDER_VALUE = 30

FIELD_ID = "id"
FIELD_ELEVATION = "elevation"
DEFAULT_OUTPUT_LAYER_NAME = "Contours"
DEFAULT_SPOT_LAYER_NAME = "Spot Heights"
DEFAULT_CRS_AUTHID = "EPSG:4326"
DEFAULT_VECTOR_FILE_ENCODING = "UTF-8"

MAX_RASTER_BANDS_FOR_RGB = 3
PREVIEW_EDGE_MAX_DIMENSION = 800
MOBILE_SAM_INSTALL_COMMAND = "pip install torch torchvision git+https://github.com/ChaoningZhang/MobileSAM.git"
SAM_INSTALL_COMMAND = "pip install torch torchvision git+https://github.com/facebookresearch/segment-anything.git"
SAM_REPORT_FILENAME = "archaeotrace_sam_report.json"
