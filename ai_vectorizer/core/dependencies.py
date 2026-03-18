# -*- coding: utf-8 -*-
"""
Helpers for optional third-party runtime dependencies.
"""

import sys


_CV2_MODULE = None
_CV2_ERROR = None
_CV2_CHECKED = False


def get_cv2():
    """Return the imported cv2 module, or None when unavailable."""
    global _CV2_MODULE, _CV2_ERROR, _CV2_CHECKED

    if not _CV2_CHECKED:
        try:
            import cv2 as imported_cv2
        except Exception as exc:
            _CV2_MODULE = None
            _CV2_ERROR = exc
        else:
            _CV2_MODULE = imported_cv2
            _CV2_ERROR = None
        _CV2_CHECKED = True

    return _CV2_MODULE


def get_cv2_error():
    """Return the last cv2 import error, if any."""
    get_cv2()
    return _CV2_ERROR


def get_cv2_error_text():
    """Return the cv2 import error as a user-facing string."""
    error = get_cv2_error()
    return "" if error is None else str(error)


def is_cv2_available():
    """Return True when OpenCV can be imported."""
    return get_cv2() is not None


def get_opencv_install_command():
    """Return a best-effort pip command for the current Python runtime."""
    executable = sys.executable or "python3"
    return f'"{executable}" -m pip install opencv-python-headless'


def build_missing_cv2_message(feature_name="This feature"):
    """Build a consistent missing-OpenCV error message."""
    detail = get_cv2_error_text()
    detail_suffix = f" ({detail})" if detail else ""
    return (
        f"{feature_name} requires OpenCV (cv2) in the QGIS Python environment. "
        f"Install it with: {get_opencv_install_command()}{detail_suffix}"
    )


def require_cv2(feature_name="This feature"):
    """Return cv2 or raise an ImportError with install guidance."""
    cv2 = get_cv2()
    if cv2 is None:
        raise ImportError(build_missing_cv2_message(feature_name))
    return cv2
