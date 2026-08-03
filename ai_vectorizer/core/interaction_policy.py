"""QGIS-independent interaction policy for the interactive tracer."""

MODE_FREEHAND = "freehand"
MODE_MOUSE_ASSIST = "mouse_assist"
MODE_AUTO_PATH = "auto_path"


def resolve_interaction_mode(*, freehand=False, auto_path=False, manual_override=False):
    """Resolve the active interaction mode from user-facing switches.

    Manual input always wins over automated routing.  ``auto_path`` is an
    explicit opt-in; the safe default is mouse-led local assistance.
    """

    if freehand or manual_override:
        return MODE_FREEHAND
    if auto_path:
        return MODE_AUTO_PATH
    return MODE_MOUSE_ASSIST


def uses_global_path_search(mode):
    """Return whether ``mode`` is allowed to replace a segment with A*/SAM."""

    return mode == MODE_AUTO_PATH


__all__ = [
    "MODE_AUTO_PATH",
    "MODE_FREEHAND",
    "MODE_MOUSE_ASSIST",
    "resolve_interaction_mode",
    "uses_global_path_search",
]
