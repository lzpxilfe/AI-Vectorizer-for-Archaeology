# -*- coding: utf-8 -*-
"""
Smart Trace Tool v2 - Magnetic Edge Snapping (No more glass walls!)

Key concept:
- User controls direction (mouse movement)
- AI just snaps to nearest edge within snap radius
- If no edge nearby, follows mouse exactly
- Result is smoothed with Bézier curves
"""
import numpy as np
import math
from qgis.gui import QgsMapToolEmitPoint, QgsRubberBand
from qgis.core import (
    QgsWkbTypes, QgsProject, QgsPointXY, QgsGeometry,
    QgsFeature, QgsCoordinateTransform,
    QgsVectorLayer, QgsField, QgsFieldConstraints, QgsApplication, QgsTask,
    QgsVectorLayerUtils, QgsRectangle, Qgis
)
from qgis.PyQt.QtCore import Qt, QTimer
try:
    from qgis.PyQt.QtCore import QVariant
except ImportError:  # PyQt6/QGIS 4
    QVariant = None
try:
    from qgis.PyQt.QtCore import QMetaType
except ImportError:
    QMetaType = None
from qgis.PyQt.QtGui import QColor

from ..core.dependencies import get_cv2, require_cv2
from ..core.edge_detector import EdgeDetector, InkEvidenceCancelled
from ..core.raster_utils import (
    compute_resampled_dimensions,
    read_raster_bands,
    read_raster_bands_with_native,
    stable_integer_band_to_uint8,
)
from ..core.interaction_policy import (
    MODE_AUTO_PATH,
    MODE_FREEHAND,
    MODE_MOUSE_ASSIST,
    resolve_interaction_mode,
    uses_global_path_search,
)
from ..core.livewire import (
    LiveWireCancelled,
    LiveWireConfig,
    LiveWireUnavailable,
    blend_path_with_cursor,
    build_livewire_tree,
    is_livewire_available,
)
from ..core.line_evidence import crop_line_evidence
from ..core.recovery_prompts import (
    RecoveryPromptError,
    build_recovery_prompt_tensors,
)
from ..core.sam_trace_kernel import (
    DEFAULT_CONFIG as DEFAULT_SAM_TRACE_CONFIG,
    SamTraceConfig,
    build_cost_map as build_sam_cost_map,
    nearest_active_pixel as find_nearest_active_pixel,
    postprocess_mask as postprocess_sam_mask,
    trace_mask as trace_sam_mask,
)
from ..core.smart_recovery import (
    arbitrate_routes,
    build_corridor_cost_map,
    recovery_gate,
)
from ..core.trace_kernel import (
    TraceConfig,
    chaikin_smooth_path,
    find_path,
    smooth_pixel_path,
)
from ..recovery import (
    RECOVERY_STATE_ENHANCED,
    RECOVERY_STATE_INK,
    RECOVERY_STATE_INK_FALLBACK,
    RECOVERY_STATE_RECOVERING,
    require_recovery_state,
)
from ..config import (
    DEFAULT_EDGE_METHOD,
    DEFAULT_OUTPUT_LAYER_NAME,
    DEFAULT_SPOT_LAYER_NAME,
    FIELD_ELEVATION,
    FIELD_ID,
    PLUGIN_NAME,
)


def _qt_value(legacy_name, scope_name):
    legacy = getattr(Qt, legacy_name, None)
    if legacy is not None:
        return legacy
    return getattr(getattr(Qt, scope_name), legacy_name)


def _geometry_type(legacy_name, modern_name):
    legacy = getattr(QgsWkbTypes, legacy_name, None)
    if legacy is not None:
        return legacy
    return getattr(Qgis.GeometryType, modern_name)


def _message_level(name):
    legacy = getattr(Qgis, name, None)
    if legacy is not None:
        return legacy
    return getattr(Qgis.MessageLevel, name)


def _task_can_cancel():
    legacy = getattr(QgsTask, "CanCancel", None)
    if legacy is not None:
        return legacy
    scoped = getattr(QgsTask, "Flag", None)
    if scoped is not None and hasattr(scoped, "CanCancel"):
        return scoped.CanCancel
    return Qgis.TaskFlag.CanCancel


def _field_type(name):
    # Prefer the QGIS 4-compatible API while retaining QGIS 3.22 fallback.
    for owner in (getattr(QMetaType, "Type", None), QVariant):
        if owner is not None and hasattr(owner, name):
            return getattr(owner, name)
    raise RuntimeError(f"Qt field type is unavailable: {name}")


def _constraint_strength(name):
    legacy = getattr(QgsFieldConstraints, name, None)
    if legacy is not None:
        return legacy
    return getattr(QgsFieldConstraints.ConstraintStrength, name)


def _rubber_band_icon(name):
    legacy = getattr(QgsRubberBand, name, None)
    if legacy is not None:
        return legacy
    return getattr(QgsRubberBand.IconType, name)


LINE_GEOMETRY = _geometry_type("LineGeometry", "Line")
POINT_GEOMETRY = _geometry_type("PointGeometry", "Point")
MESSAGE_INFO = _message_level("Info")
MESSAGE_WARNING = _message_level("Warning")
MESSAGE_CRITICAL = _message_level("Critical")
HARD_CONSTRAINT = _constraint_strength("ConstraintStrengthHard")


class _AStarPreviewTask(QgsTask):
    """Run the QGIS-free A* kernel without blocking canvas interaction."""

    def __init__(
        self,
        *,
        cost_map,
        start_pixel,
        target_pixel,
        target_xy,
        generation,
        config,
        callback,
    ):
        super().__init__("ArchaeoTrace live path preview", _task_can_cancel())
        self.cost_map = cost_map
        self.start_pixel = tuple(start_pixel)
        self.target_pixel = tuple(target_pixel)
        self.target_xy = tuple(target_xy)
        self.generation = int(generation)
        self.config = config
        self.callback = callback
        self.trace_result = None
        self.error = None

    def run(self):
        if self.isCanceled():
            return False
        try:
            self.trace_result = find_path(
                self.cost_map,
                self.start_pixel,
                self.target_pixel,
                allow_partial=True,
                config=self.config,
            )
            return not self.isCanceled()
        except Exception as exc:
            self.error = exc
            return False

    def finished(self, result):
        # QgsTask invokes finished() on the main thread, where QGIS geometry
        # and rubber-band updates are safe.
        self.callback(self, bool(result), self.trace_result, self.error)


class _InkEvidenceTask(QgsTask):
    """Build Ink v1/v2 evidence away from QGIS' main thread."""

    def __init__(
        self,
        *,
        detector,
        fallback_image,
        fallback_rgb_image,
        fallback_cache_identity,
        fallback_cache_extent,
        fallback_output_size,
        evidence_image,
        recovery_image,
        recovery_compatible,
        recovery_disabled_reason,
        tile_origin,
        enable_evidence,
        evidence_disabled_reason,
        build_cost_map,
        edge_weight,
        generation,
        cache_identity,
        cache_extent,
        output_size,
        callback,
    ):
        super().__init__("ArchaeoTrace Ink evidence", _task_can_cancel())
        self.detector = detector
        self.fallback_image = fallback_image
        self.fallback_rgb_image = fallback_rgb_image
        self.fallback_cache_identity = tuple(fallback_cache_identity)
        self.fallback_cache_extent = tuple(
            float(value) for value in fallback_cache_extent
        )
        self.fallback_output_size = tuple(
            int(value) for value in fallback_output_size
        )
        self.evidence_image = evidence_image
        self.recovery_image = recovery_image
        self.recovery_compatible = bool(recovery_compatible)
        self.recovery_disabled_reason = str(recovery_disabled_reason or "")
        self.tile_origin = tuple(tile_origin)
        self.enable_evidence = bool(enable_evidence)
        self.evidence_disabled_reason = str(evidence_disabled_reason or "")
        self.build_cost_map = bool(build_cost_map)
        self.edge_weight = float(edge_weight)
        self.generation = int(generation)
        self.cache_identity = tuple(cache_identity)
        self.cache_extent = tuple(float(value) for value in cache_extent)
        self.output_size = tuple(int(value) for value in output_size)
        self.callback = callback
        self.fallback_edges = None
        self.evidence = None
        self.evidence_error = None
        self.cost_map = None
        self.error = None

    def run(self):
        if self.isCanceled():
            return False
        try:
            # Always produce the stable v1 champion first. A v2 failure is a
            # recoverable result, not a failed tracing session.
            self.fallback_edges = self.detector.detect_edges(
                self.fallback_image
            )
            if self.isCanceled():
                return False
            detect_evidence = getattr(self.detector, "detect_ink_evidence", None)
            if not self.enable_evidence:
                self.evidence_error = RuntimeError(
                    self.evidence_disabled_reason
                    or "continuous Ink evidence is disabled for this cache"
                )
            elif not callable(detect_evidence):
                self.evidence_error = RuntimeError(
                    "continuous Ink evidence is unavailable"
                )
            else:
                try:
                    try:
                        self.evidence = detect_evidence(
                            self.evidence_image,
                            tile_origin=self.tile_origin,
                            cancel_check=self.isCanceled,
                        )
                    except TypeError as exc:
                        if not any(
                            keyword in str(exc)
                            for keyword in ("cancel_check", "tile_origin")
                        ):
                            raise
                        try:
                            self.evidence = detect_evidence(
                                self.evidence_image,
                                tile_origin=self.tile_origin,
                            )
                        except TypeError as legacy_exc:
                            if "tile_origin" not in str(legacy_exc):
                                raise
                            self.evidence = detect_evidence(self.evidence_image)
                except InkEvidenceCancelled:
                    return False
                except Exception as exc:
                    self.evidence_error = exc
            selected_edges = self.fallback_edges
            if self.evidence is not None:
                selected_edges = np.where(
                    np.asarray(self.evidence.centerline, dtype=bool),
                    255,
                    0,
                ).astype(np.uint8)
            if self.build_cost_map:
                self.cost_map = self.detector.get_edge_cost_map(
                    selected_edges,
                    self.edge_weight,
                )
            return not self.isCanceled()
        except Exception as exc:
            self.error = exc
            return False

    def finished(self, result):
        self.callback(
            self,
            bool(result),
            self.fallback_edges,
            self.evidence,
            self.cost_map,
            self.evidence_error,
            self.error,
        )


class _LiveWireTreeTask(QgsTask):
    """Build one anchor-rooted Live-Wire tree away from the UI thread."""

    def __init__(
        self,
        *,
        image,
        edges,
        evidence,
        anchor_pixel,
        incoming_direction,
        strength,
        generation,
        config,
        callback,
    ):
        super().__init__("ArchaeoTrace Live-Wire tree", _task_can_cancel())
        self.image = image
        self.edges = edges
        self.evidence = evidence
        self.anchor_pixel = tuple(anchor_pixel)
        self.incoming_direction = incoming_direction
        self.strength = float(strength)
        self.generation = int(generation)
        self.config = config
        self.callback = callback
        self.tree = None
        self.error = None

    def run(self):
        if self.isCanceled():
            return False
        try:
            self.tree = build_livewire_tree(
                self.image,
                self.edges,
                self.anchor_pixel,
                strength=self.strength,
                incoming_direction=self.incoming_direction,
                config=self.config,
                cancel_check=self.isCanceled,
                evidence=self.evidence,
            )
            return not self.isCanceled()
        except LiveWireCancelled:
            return False
        except Exception as exc:
            self.error = exc
            return False

    def finished(self, result):
        self.callback(self, bool(result), self.tree, self.error)


class _RecoveryPreviewTask(QgsTask):
    """Evaluate one EfficientSAM challenger without touching QGIS objects."""

    def __init__(
        self,
        *,
        engine,
        image,
        encoding,
        evidence,
        champion_path,
        start_pixel,
        target_pixel,
        prompt_points,
        prompt_labels,
        window_bounds,
        cache_generation,
        request_generation,
        preview_identity,
        smooth_window_size,
        trace_config,
        callback,
    ):
        super().__init__("ArchaeoTrace Smart Recovery", _task_can_cancel())
        self.engine = engine
        self.image = image
        self.encoding = encoding
        self.evidence = evidence
        self.champion_path = tuple(tuple(point) for point in champion_path)
        self.start_pixel = tuple(start_pixel)
        self.target_pixel = tuple(target_pixel)
        self.prompt_points = prompt_points
        self.prompt_labels = prompt_labels
        self.window_bounds = tuple(int(value) for value in window_bounds)
        self.cache_generation = int(cache_generation)
        self.request_generation = int(request_generation)
        self.preview_identity = preview_identity
        self.smooth_window_size = int(smooth_window_size)
        self.trace_config = trace_config
        self.callback = callback
        self.selection = None
        self.trace_result = None
        self.challenger_path = None
        self.error = None

    def run(self):
        if self.isCanceled():
            return False
        try:
            prepare_image = getattr(self.engine, "set_image", None)
            if callable(prepare_image):
                self.encoding = prepare_image(self.image)
            elif self.encoding is None:
                self.encoding = getattr(self.engine, "encode")(self.image)
            if self.isCanceled():
                return False

            predict = getattr(self.engine, "predict")
            if callable(prepare_image):
                prediction = predict(self.prompt_points, self.prompt_labels)
            else:
                prediction = predict(
                    self.encoding,
                    self.prompt_points,
                    self.prompt_labels,
                )
            corridor = getattr(prediction, "mask", prediction)
            corridor = np.asarray(corridor, dtype=np.float32)
            evidence_shape = tuple(int(value) for value in self.evidence.shape)
            if corridor.shape != evidence_shape:
                raise ValueError(
                    "Recovery corridor must match the full Ink evidence grid"
                )
            x0, y0, x1, y1 = self.window_bounds
            height, width = evidence_shape
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError("Recovery window leaves the Ink evidence grid")
            local_start = (
                float(self.start_pixel[0]) - x0,
                float(self.start_pixel[1]) - y0,
            )
            local_target = (
                float(self.target_pixel[0]) - x0,
                float(self.target_pixel[1]) - y0,
            )
            window_width = x1 - x0
            window_height = y1 - y0
            for label, (x, y) in (
                ("start", local_start),
                ("target", local_target),
            ):
                if not (0.0 <= x < window_width and 0.0 <= y < window_height):
                    raise ValueError(
                        f"Recovery {label} lies outside the Live-Wire window"
                    )

            bounded_evidence = crop_line_evidence(
                self.evidence,
                self.window_bounds,
            )
            bounded_corridor = np.ascontiguousarray(
                corridor[y0:y1, x0:x1],
                dtype=np.float32,
            )
            cost_map = build_corridor_cost_map(
                bounded_evidence,
                bounded_corridor,
            )
            if self.isCanceled():
                return False

            self.trace_result = find_path(
                cost_map,
                local_start,
                local_target,
                allow_partial=False,
                config=self.trace_config,
            )
            raw_challenger = tuple(
                (float(x) + x0, float(y) + y0)
                for x, y in self.trace_result.points_xy
            )
            challenger_path = list(
                smooth_pixel_path(
                    raw_challenger,
                    window_size=self.smooth_window_size,
                )
            )
            if challenger_path:
                challenger_path[0] = tuple(
                    float(value) for value in self.start_pixel
                )
                challenger_path[-1] = tuple(
                    float(value) for value in self.target_pixel
                )
            self.challenger_path = tuple(challenger_path)
            self.selection = arbitrate_routes(
                self.champion_path,
                self.challenger_path,
                self.evidence,
                expected_start=self.start_pixel,
                expected_end=self.target_pixel,
            )
            return not self.isCanceled()
        except Exception as exc:
            self.error = exc
            return False

    def finished(self, result):
        self.callback(
            self,
            bool(result),
            self.trace_result,
            self.challenger_path,
            self.selection,
            self.error,
        )


class SmartTraceTool(QgsMapToolEmitPoint):
    SNAP_RADIUS_BASE = 15
    SNAP_RADIUS_EDGE_WEIGHT_FACTOR = 0.7
    SAMPLE_INTERVAL_PIXELS = 3
    PREVIEW_INTERVAL_PIXELS = 1

    # The slider is literal: at 100% a nearby fallback snap can reach the
    # detected line completely; at 0% it cannot move the cursor at all.
    EDGE_BLEND_FACTOR = 1.0
    MAX_EDGE_BLEND = 1.0
    EDGE_PIXEL_THRESHOLD = 128

    ANGLE_CONSTRAINED_SNAP_RADIUS = 6
    LOCAL_EDGE_SEARCH_RADIUS_PIXELS = 7
    MAX_EDGE_ATTRACTION_PIXELS = 4
    GENTLE_SNAP_RADIUS = 5
    MAX_TURN_ANGLE_DEGREES = 40

    ENDPOINT_SNAP_TOLERANCE_PIXELS = 10
    CLOSE_TOLERANCE_BASE_PIXELS = 20
    CLOSE_TOLERANCE_SPOT_PIXELS = 30

    CACHE_MAX_DIMENSION = 1000
    CACHE_MIN_DIMENSION = 10
    CACHE_MAX_BANDS_FOR_RGB = 3
    CACHE_DEBOUNCE_MS = 150
    # Keep the product cache on the detector's source-pixel grid.  A visible
    # viewport is expanded to complete normalization tiles plus their halo so
    # that a source pixel receives the same local context after a small pan.
    INK_EVIDENCE_TILE_SOURCE_PIXELS = int(
        EdgeDetector.INK_EVIDENCE_TILE_SIZE
    )
    INK_EVIDENCE_FILTER_RADIUS_SOURCE_PIXELS = int(
        max(EdgeDetector.INK_EVIDENCE_SCALES) // 2
    )
    INK_EVIDENCE_HALO_SOURCE_PIXELS = int(
        EdgeDetector.INK_EVIDENCE_TILE_HALO
        + INK_EVIDENCE_FILTER_RADIUS_SOURCE_PIXELS
    )
    PROPOSAL_DEBOUNCE_MS = 110
    PROPOSAL_ACCEPT_TOLERANCE_PIXELS = 12

    # 320px retained the same real-map route as 384px while cutting the
    # anchor-tree build by roughly 28% in QGIS 3.40.5. Directional bias still
    # leaves about 218px of forward look-ahead for ordinary cursor segments.
    LIVEWIRE_WINDOW_PIXELS = 320
    LIVEWIRE_TARGET_SNAP_PIXELS = 6
    LIVEWIRE_SMOOTH_WINDOW_SIZE = 5

    PATH_MOVE_COST_STRAIGHT = 1.0
    PATH_MOVE_COST_DIAGONAL = 1.41421356237
    PATH_MAX_ITER_BASE = 100000
    PATH_MAX_ITER_DISTANCE_FACTOR = 500
    PATH_SMOOTH_WINDOW_SIZE = 5
    PATH_TIMEOUT_MESSAGE_SECONDS = 3

    SAM_MASK_MIN_PIXELS = DEFAULT_SAM_TRACE_CONFIG.mask_min_pixels
    SAM_MASK_MAX_AREA_RATIO = DEFAULT_SAM_TRACE_CONFIG.mask_max_area_ratio
    SAM_PROMPT_HISTORY_POINTS = 2
    SAM_NEGATIVE_DISTANCE_PIXELS = 10
    SAM_NEAREST_ACTIVE_RADIUS = DEFAULT_SAM_TRACE_CONFIG.nearest_active_radius
    SAM_OUTSIDE_COST = DEFAULT_SAM_TRACE_CONFIG.outside_cost
    SAM_INSIDE_COST = DEFAULT_SAM_TRACE_CONFIG.inside_cost
    SAM_EDGE_COST = DEFAULT_SAM_TRACE_CONFIG.edge_cost
    SAM_SKELETON_COST = DEFAULT_SAM_TRACE_CONFIG.skeleton_cost
    SAM_CENTERLINE_BONUS = DEFAULT_SAM_TRACE_CONFIG.centerline_bonus
    SAM_MASK_CLOSE_KERNEL = DEFAULT_SAM_TRACE_CONFIG.mask_close_kernel

    ELEVATION_DEFAULT = 0.0
    ELEVATION_MIN = -1000.0
    ELEVATION_MAX = 10000.0
    ELEVATION_DECIMALS = 1

    CHAIKIN_ITERATIONS = 3
    CHAIKIN_Q_WEIGHT = 0.75
    CHAIKIN_R_WEIGHT = 0.25

    PREVIEW_BAND_COLOR = (0, 180, 0, 180)
    PREVIEW_BAND_WIDTH = 8
    PREVIEW_BAND_LINE_STYLE = _qt_value("DashLine", "PenStyle")
    # Keep every uncommitted suggestion in the same visual language. The
    # distinction is interaction state, not another competing line color.
    PROPOSAL_BAND_COLOR = (0, 180, 0, 180)
    PROPOSAL_BAND_WIDTH = 8
    PROPOSAL_BAND_LINE_STYLE = _qt_value("DashLine", "PenStyle")
    CONFIRM_BAND_COLOR = (255, 50, 50, 255)
    CONFIRM_BAND_WIDTH = 3
    START_MARKER_COLOR = (255, 255, 0, 255)
    START_MARKER_WIDTH = 12
    START_MARKER_ICON = _rubber_band_icon("ICON_CIRCLE")
    CLOSE_INDICATOR_COLOR = (0, 255, 255, 200)
    CLOSE_INDICATOR_WIDTH = 16
    CLOSE_INDICATOR_ICON = _rubber_band_icon("ICON_CIRCLE")
    CHECKPOINT_MARKER_COLOR = (50, 150, 255, 255)
    CHECKPOINT_MARKER_WIDTH = 10
    CHECKPOINT_MARKER_ICON = _rubber_band_icon("ICON_BOX")
    SNAP_MARKER_COLOR = (255, 0, 255, 200)
    SNAP_MARKER_WIDTH = 15
    SNAP_MARKER_ICON = _rubber_band_icon("ICON_X")
    SPOT_LAYER_OWNERSHIP_PROPERTY = "ArchaeoTrace/ownedSpotHeightLayer"
    A_STAR_NEIGHBORS = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]

    def _tr(self, ko_text, en_text):
        return en_text if getattr(self, "language", "ko") == "en" else ko_text

    @staticmethod
    def unsupported_output_reason(layer):
        """Return why a layer cannot be edited without dimensional data loss."""

        if layer is None:
            return "missing"
        if layer.geometryType() != LINE_GEOMETRY:
            return "not_line"
        wkb_type = layer.wkbType()
        if QgsWkbTypes.hasZ(wkb_type) or QgsWkbTypes.hasM(wkb_type):
            return "z_or_m"
        return None

    def _needs_edge_cache(self):
        """Return whether this interaction actually needs raster edge data.

        A zero-strength mouse-led trace must be a literal cursor trace.  In
        that state, skipping the raster read is important both semantically
        (no hidden AI nudge remains) and for startup/zoom responsiveness.
        SAM still needs an image cache for its explicit prompt mode. A
        non-SAM Auto Path at 0% remains a literal cursor segment and skips
        all raster work too.
        """
        return (
            not self.freehand
            and (self.edge_weight > 0.0 or self.use_sam)
        )

    @staticmethod
    def _configure_band(band, color, width, icon=None, line_style=None):
        band.setColor(QColor(*color))
        band.setWidth(width)
        if icon is not None:
            band.setIcon(icon)
        if line_style is not None:
            band.setLineStyle(line_style)

    def _set_extent_cache_listener(self, enabled):
        if enabled == getattr(self, "_extent_cache_listener_connected", False):
            return

        signal = self.canvas.extentsChanged
        try:
            if enabled:
                signal.connect(self._schedule_edge_cache_update)
            else:
                signal.disconnect(self._schedule_edge_cache_update)
                self._edge_cache_timer.stop()
            self._extent_cache_listener_connected = enabled
        except (RuntimeError, TypeError) as exc:
            if not enabled:
                self._extent_cache_listener_connected = False
            print(f"Extent cache listener update failed: {exc}")

    def _set_coordinate_crs_listeners(self, enabled):
        if enabled == getattr(self, "_coordinate_crs_listeners_connected", False):
            return

        connections = []
        for owner, signal_name in (
            (self.canvas, "destinationCrsChanged"),
            (self.raster_layer, "crsChanged"),
        ):
            try:
                signal = getattr(owner, signal_name, None)
            except RuntimeError:
                signal = None
            if signal is not None:
                connections.append((signal, self._on_cache_crs_changed))
        try:
            for signal, callback in connections:
                if signal is None:
                    continue
                if enabled:
                    signal.connect(callback)
                else:
                    signal.disconnect(callback)
            self._coordinate_crs_listeners_connected = enabled
        except (RuntimeError, TypeError) as exc:
            if not enabled:
                self._coordinate_crs_listeners_connected = False
            print(f"CRS listener update failed: {exc}")

    def _set_source_lifecycle_listeners(self, enabled):
        if enabled == getattr(self, "_source_lifecycle_listeners_connected", False):
            return

        connections = []
        for layer in (self.raster_layer, self.vector_layer):
            for signal_name in ("dataSourceChanged", "willBeDeleted"):
                try:
                    signal = getattr(layer, signal_name, None)
                except RuntimeError:
                    signal = None
                if signal is not None:
                    connections.append(signal)
        # A vector CRS reassignment changes the interpretation of every
        # existing coordinate. Stop instead of silently extending a feature
        # whose source semantics changed under an active trace.
        try:
            vector_crs_changed = getattr(self.vector_layer, "crsChanged", None)
        except RuntimeError:
            vector_crs_changed = None
        if vector_crs_changed is not None:
            connections.append(vector_crs_changed)

        try:
            raster_data_changed = getattr(self.raster_layer, "dataChanged", None)
        except RuntimeError:
            raster_data_changed = None

        try:
            for signal in connections:
                if enabled:
                    signal.connect(self._on_source_layer_invalidated)
                else:
                    signal.disconnect(self._on_source_layer_invalidated)
            if raster_data_changed is not None:
                if enabled:
                    raster_data_changed.connect(self._on_raster_data_changed)
                else:
                    raster_data_changed.disconnect(self._on_raster_data_changed)
            self._source_lifecycle_listeners_connected = enabled
        except (RuntimeError, TypeError) as exc:
            if not enabled:
                self._source_lifecycle_listeners_connected = False
            print(f"Layer lifecycle listener update failed: {exc}")

    def _on_raster_data_changed(self, *_args):
        """Invalidate source pixels changed in place and request a fresh cache."""

        if self._is_active and self._needs_edge_cache():
            self._schedule_edge_cache_update()
        else:
            self._clear_edge_cache()

    def _on_source_layer_invalidated(self, *_args):
        """Stop before a replaced/deleted source can receive trace edits."""

        # Invalidate worker generations before asking QGIS to deactivate us.
        # Some providers emit this signal while their C++ layer is already
        # being torn down, and unsetMapTool() is therefore not guaranteed to
        # reach deactivate().  A late Ink task must never republish a cache
        # captured from that obsolete source.
        self._clear_edge_cache()
        try:
            if self.canvas.mapTool() is self:
                self.canvas.unsetMapTool(self)
        except RuntimeError as exc:
            print(f"Could not stop tracing after a layer change: {exc}")

    def _refresh_crs_transforms(self):
        canvas_crs = self.canvas.mapSettings().destinationCrs()
        raster_crs = self.raster_layer.crs()
        self.to_raster_transform = QgsCoordinateTransform(
            canvas_crs,
            raster_crs,
            QgsProject.instance(),
        )
        self.to_map_transform = QgsCoordinateTransform(
            raster_crs,
            canvas_crs,
            QgsProject.instance(),
        )
        self._transform_canvas_crs = canvas_crs
        self._transform_raster_crs = raster_crs

    def _ensure_crs_transforms_current(self):
        canvas_crs = self.canvas.mapSettings().destinationCrs()
        raster_crs = self.raster_layer.crs()
        if (
            canvas_crs != getattr(self, "_transform_canvas_crs", None)
            or raster_crs != getattr(self, "_transform_raster_crs", None)
        ):
            self._refresh_crs_transforms()

    def _on_cache_crs_changed(self, *_args):
        """Discard coordinates/cache created under an obsolete CRS."""

        self.reset_tracing()
        try:
            self._refresh_crs_transforms()
        except RuntimeError as exc:
            self._push_message(
                self._tr(
                    f"좌표계 변경을 적용하지 못했습니다: {exc}",
                    f"Could not apply the CRS change: {exc}",
                ),
                MESSAGE_CRITICAL,
            )
            return
        self._clear_edge_cache()
        if self._needs_edge_cache():
            self._edge_cache_timer.start()

    def _schedule_edge_cache_update(self):
        """Debounce expensive raster reads while the map is being zoomed."""
        self._clear_edge_cache()
        # A preview can contain pixel-to-map points from the previous extent.
        # Never commit that stale preview after a zoom or pan.
        self.preview_path = []
        self.preview_is_global = False
        self.preview_target = None
        self._proposal_timer.stop()
        self._cancel_proposal_task()
        self._proposal_generation += 1
        self._proposal_request_point = None
        self.preview_band.reset(LINE_GEOMETRY)
        self.last_sample_pos = None
        self.last_preview_pos = None
        self._edge_cache_timer.start()

    def __init__(self, canvas, raster_layer, vector_layer, model_type=0,
                 sam_engine=None, edge_weight=0.5, freehand=False, edge_method=DEFAULT_EDGE_METHOD,
                 iface=None, language="ko", auto_path=False,
                 recovery_engine=None, smart_recovery=False,
                 recovery_state_callback=None):
        self.canvas = canvas
        super().__init__(self.canvas)
        self.iface = iface
        self.language = language

        self.raster_layer = raster_layer
        self.vector_layer = vector_layer
        if not self.vector_layer:
            self.vector_layer = self.create_output_layer()
        unsupported_reason = self.unsupported_output_reason(self.vector_layer)
        if unsupported_reason:
            raise ValueError(
                "ArchaeoTrace requires a 2D line output layer; Z/M layers "
                "cannot be edited without losing dimensional values."
            )
        # All user feature mutations must go through QGIS' edit buffer.  A
        # direct provider write commits immediately for file-backed layers,
        # bypassing both Undo and QGIS' normal save/discard confirmation.
        if self.vector_layer.readOnly():
            raise ValueError("ArchaeoTrace cannot edit a read-only output layer.")
        if not self._ensure_edit_session(self.vector_layer):
            raise RuntimeError(
                "Could not start a QGIS edit session for the output layer."
            )
        self.sam_engine = sam_engine
        self.model_type = model_type
        self.use_sam = (
            self.sam_engine is not None
            and getattr(self.sam_engine, "is_ready", False)
        )
        self.freehand = freehand
        self.edge_method = edge_method
        # The UI exposes a 0-100 slider. Clamp programmatic callers too so
        # the interaction contract remains exactly [0.0, 1.0].
        self.edge_weight = max(0.0, min(1.0, float(edge_weight)))
        self.auto_path = bool(auto_path) and not self.freehand
        self.smart_recovery_requested = bool(smart_recovery)
        self.recovery_engine = recovery_engine
        self.recovery_state_callback = recovery_state_callback
        self.smart_recovery_enabled = (
            self.smart_recovery_requested
            and not self.freehand
            and self.edge_weight > 0.0
            and self.edge_method == EdgeDetector.METHOD_INK
            and self.recovery_engine is not None
        )
        self._current_recovery_state = RECOVERY_STATE_INK

        # Keep the search geometry local. The strength slider controls the
        # actual attraction radius and blend below; it must not turn into a
        # global route selector.
        self.snap_radius = max(
            1,
            int(
                self.SNAP_RADIUS_BASE
                * (1.0 - self.edge_weight * self.SNAP_RADIUS_EDGE_WEIGHT_FACTOR)
            ),
        )

        # Path tracking
        self.path_points = []
        self.preview_path = []  # For hovering preview
        self.preview_is_global = False
        self.preview_target = None
        self.is_tracing = False
        self.start_point = None
        self.last_map_point = None
        self.last_input_point = None
        self.last_preview_pos = None

        # RubberBands for visualization
        self.preview_band = QgsRubberBand(self.canvas, LINE_GEOMETRY)
        self._preview_style_is_global = None
        self._configure_band(
            self.preview_band,
            self.PREVIEW_BAND_COLOR,
            self.PREVIEW_BAND_WIDTH,
            line_style=self.PREVIEW_BAND_LINE_STYLE,
        )

        self.confirm_band = QgsRubberBand(self.canvas, LINE_GEOMETRY)
        self._configure_band(
            self.confirm_band,
            self.CONFIRM_BAND_COLOR,
            self.CONFIRM_BAND_WIDTH,
        )

        self.start_marker = QgsRubberBand(self.canvas, POINT_GEOMETRY)
        self._configure_band(
            self.start_marker,
            self.START_MARKER_COLOR,
            self.START_MARKER_WIDTH,
            icon=self.START_MARKER_ICON,
        )

        self.close_indicator = QgsRubberBand(self.canvas, POINT_GEOMETRY)
        self._configure_band(
            self.close_indicator,
            self.CLOSE_INDICATOR_COLOR,
            self.CLOSE_INDICATOR_WIDTH,
            icon=self.CLOSE_INDICATOR_ICON,
        )

        # Checkpoint markers (blue diamonds)
        self.checkpoint_markers = QgsRubberBand(self.canvas, POINT_GEOMETRY)
        self._configure_band(
            self.checkpoint_markers,
            self.CHECKPOINT_MARKER_COLOR,
            self.CHECKPOINT_MARKER_WIDTH,
            icon=self.CHECKPOINT_MARKER_ICON,
        )

        # Checkpoints: list of point indices where user clicked
        self.checkpoints = []

        # Snap marker (for resuming drawing)
        self.snap_marker = QgsRubberBand(self.canvas, POINT_GEOMETRY)
        self._configure_band(
            self.snap_marker,
            self.SNAP_MARKER_COLOR,
            self.SNAP_MARKER_WIDTH,
            icon=self.SNAP_MARKER_ICON,
        )

        # Spot Height Layer (Point)
        self.spot_height_layer = None

        self.cv2 = get_cv2()
        # Ink Centerline and Legacy Canny do not require a separate OpenCV
        # install. SAM, LSD, and HED still need cv2.
        needs_cv2 = self.use_sam or (
            self.edge_weight > 0.0
            and self.edge_method in (
                EdgeDetector.METHOD_LSD,
                EdgeDetector.METHOD_HED,
            )
        )
        if needs_cv2:
            self.cv2 = require_cv2("OpenCV tracing")

        # Edge detector
        self.edge_detector = None
        if self._needs_edge_cache():
            self.edge_detector = EdgeDetector(method=self.edge_method)

        # Edge cache
        self.cached_edges = None
        self.cached_cost = None
        self.cache_extent = None
        self.cache_tile_origin = None
        self.cache_identity = None
        self.cache_transform = None  # Pixel <-> Map transform
        self.cached_rgb_image = None
        self.cached_ink_evidence = None
        self._recovery_cache_compatible = False
        self._recovery_cache_disabled_reason = (
            "Smart Recovery is waiting for a native Byte Ink v2 cache."
        )
        self.sam_image_ready = False
        self.sam_warning_emitted = False
        self.cache_dirty = True
        self._cache_generation = 0
        self._ink_evidence_task = None
        self._ink_evidence_generation = 0
        self._pending_cache_identity = None
        self._is_active = False

        self._edge_cache_timer = QTimer(self)
        self._edge_cache_timer.setSingleShot(True)
        self._edge_cache_timer.setInterval(self.CACHE_DEBOUNCE_MS)
        self._edge_cache_timer.timeout.connect(self.update_edge_cache)
        self._extent_cache_listener_connected = False
        self._coordinate_crs_listeners_connected = False
        self._source_lifecycle_listeners_connected = False

        # Auto Path/SAM proposals are debounced so the expensive route is
        # calculated after the cursor pauses, not for every mouse event.
        self._proposal_timer = QTimer(self)
        self._proposal_timer.setSingleShot(True)
        self._proposal_timer.setInterval(self.PROPOSAL_DEBOUNCE_MS)
        self._proposal_timer.timeout.connect(self._update_auto_path_preview)
        self._proposal_request_point = None
        self._proposal_generation = 0
        self._proposal_task = None

        # Human-led Live-Wire builds once per accepted anchor. Mouse moves
        # only trace predecessor indices from this immutable tree.
        self._livewire_tree = None
        self._livewire_task = None
        self._livewire_generation = 0
        self._livewire_anchor_pixel = None
        self._livewire_request_point = None
        self._livewire_warning_emitted = False
        self._livewire_disabled = False
        self._livewire_failed_anchor = None

        # Smart Recovery evaluates only an already-rendered Ink candidate.
        # Requests are coalesced so the ONNX session is never invoked by two
        # concurrent cursor tasks, and stale results cannot replace a newer
        # champion preview.
        self._recovery_task = None
        self._recovery_request = None
        self._recovery_generation = 0
        self._recovery_preview_identity = None
        self._recovery_encoding = None
        self._recovery_encoding_cache_generation = None

        # CRS transforms are refreshed if the canvas or raster CRS changes.
        # Reusing transforms created for an earlier project CRS silently moves
        # traces by hundreds of kilometres in otherwise valid projects.
        self._transform_canvas_crs = None
        self._transform_raster_crs = None
        self._refresh_crs_transforms()

        # Resume/Merge State
        self.resume_feature_id = None
        self.resume_at_start = False  # True if appending to Start of existing line

        # Stability (Anti-Pulse)
        self.last_hover_pos = None
        self.last_sample_pos = None

        if self.smart_recovery_requested and not self.smart_recovery_enabled:
            self._emit_recovery_state(
                RECOVERY_STATE_INK_FALLBACK,
                "Recovery model unavailable; Ink remains active.",
            )
        else:
            self._emit_recovery_state(RECOVERY_STATE_INK)

    def create_output_layer(self):
        crs = self.canvas.mapSettings().destinationCrs().authid()
        layer = QgsVectorLayer(f"LineString?crs={crs}", DEFAULT_OUTPUT_LAYER_NAME, "memory")
        pr = layer.dataProvider()
        pr.addAttributes([QgsField(FIELD_ID, _field_type("Int"))])
        layer.updateFields()
        QgsProject.instance().addMapLayer(layer)
        return layer

    def get_or_create_spot_layer(self):
        """Get or create the Spot Heights (Point) layer."""
        if self.spot_height_layer is not None:
            try:
                if not self.spot_height_layer.isValid():
                    self.spot_height_layer = None
            except RuntimeError:
                # QgsProject owns layers and deletes the C++ object when a
                # user removes it, even while Python still has its wrapper.
                self.spot_height_layer = None

        target_crs = (
            self.vector_layer.crs()
            if self.vector_layer and self.vector_layer.crs().isValid()
            else self.canvas.mapSettings().destinationCrs()
        )

        if self.spot_height_layer is None:
            # Reuse only a layer explicitly created by this plugin. A project
            # can legitimately contain a user-owned layer with the same
            # display name, which must never be mutated implicitly.
            for layer in QgsProject.instance().mapLayers().values():
                ownership = layer.customProperty(
                    self.SPOT_LAYER_OWNERSHIP_PROPERTY,
                    False,
                )
                if (
                    str(ownership).lower() in ("1", "true", "yes")
                    and layer.isValid()
                    and layer.geometryType() == POINT_GEOMETRY
                    and layer.crs() == target_crs
                ):
                    self.spot_height_layer = layer
                    break

        if self.spot_height_layer is None:
            crs = target_crs.authid()
            self.spot_height_layer = QgsVectorLayer(f"Point?crs={crs}", DEFAULT_SPOT_LAYER_NAME, "memory")
            pr = self.spot_height_layer.dataProvider()
            pr.addAttributes([QgsField(FIELD_ELEVATION, _field_type("Double"))])
            self.spot_height_layer.updateFields()
            self.spot_height_layer.setCustomProperty(
                self.SPOT_LAYER_OWNERSHIP_PROPERTY,
                True,
            )
            QgsProject.instance().addMapLayer(self.spot_height_layer)

        return self.spot_height_layer

    def _push_message(self, text, level=MESSAGE_WARNING, duration=4):
        if getattr(self, "iface", None):
            self.iface.messageBar().pushMessage(PLUGIN_NAME, text, level, duration)
        else:
            print(text)

    def _emit_recovery_state(self, state, detail=""):
        """Publish one validated, main-thread recovery state to the dock."""

        state = require_recovery_state(state)
        self._current_recovery_state = state
        callback = getattr(self, "recovery_state_callback", None)
        if not callable(callback):
            return
        try:
            callback(state, str(detail or ""))
        except (RuntimeError, TypeError) as exc:
            print(f"Smart Recovery state callback failed: {exc}")

    def set_recovery_engine(self, engine):
        """Attach a background-prepared engine without changing the Ink route."""

        if getattr(self, "_disposed", False):
            return False
        self._cancel_recovery_task(clear_request=True)
        self._recovery_generation += 1
        self._recovery_encoding = None
        self._recovery_encoding_cache_generation = None
        self.recovery_engine = engine
        self.smart_recovery_enabled = (
            self.smart_recovery_requested
            and not self.freehand
            and self.edge_weight > 0.0
            and self.edge_method == EdgeDetector.METHOD_INK
            and engine is not None
            and bool(getattr(engine, "is_ready", True))
        )
        if self.smart_recovery_enabled:
            self._emit_recovery_state(
                RECOVERY_STATE_INK,
                "Recovery model is ready; the current Ink route is unchanged.",
            )
        elif self.smart_recovery_requested:
            self._emit_recovery_state(
                RECOVERY_STATE_INK_FALLBACK,
                "Recovery model is unavailable; Ink remains active.",
            )
        return self.smart_recovery_enabled

    def _cancel_recovery_task(self, *, clear_request=True):
        task = getattr(self, "_recovery_task", None)
        if clear_request:
            self._recovery_request = None
        if task is not None:
            try:
                task.cancel()
            except RuntimeError:
                pass
        return task is not None

    def _cancel_ink_evidence_task(self):
        task = getattr(self, "_ink_evidence_task", None)
        self._ink_evidence_task = None
        self._pending_cache_identity = None
        self._ink_evidence_generation += 1
        if task is not None:
            try:
                task.cancel()
            except RuntimeError:
                pass
        return task is not None

    def _invalidate_recovery(self, detail=""):
        """Make every outstanding challenger stale while preserving Ink."""

        self._cancel_recovery_task(clear_request=True)
        self._recovery_generation += 1
        self._recovery_preview_identity = None
        if self.smart_recovery_requested:
            self._emit_recovery_state(RECOVERY_STATE_INK, detail)

    def _clear_edge_cache(self):
        ink_evidence_was_running = self._cancel_ink_evidence_task()
        recovery_was_running = self._cancel_recovery_task(clear_request=True)
        self._recovery_generation += 1
        self._recovery_preview_identity = None
        self._recovery_encoding = None
        self._recovery_encoding_cache_generation = None
        self._cache_generation += 1
        self._cancel_livewire_task()
        self._livewire_generation += 1
        self._livewire_tree = None
        self._livewire_anchor_pixel = None
        self._livewire_request_point = None
        self._livewire_failed_anchor = None
        self.cached_edges = None
        self.cached_cost = None
        self.cached_ink_evidence = None
        self.cache_extent = None
        self.cache_tile_origin = None
        self.cache_identity = None
        self.cache_transform = None
        self.cached_rgb_image = None
        self._recovery_cache_compatible = False
        self._recovery_cache_disabled_reason = (
            "Smart Recovery is waiting for a native Byte Ink v2 cache."
        )
        self.sam_image_ready = False
        self.sam_warning_emitted = False
        self.cache_dirty = True
        if recovery_was_running and self.smart_recovery_requested:
            self._emit_recovery_state(
                RECOVERY_STATE_INK_FALLBACK,
                "Recovery cancelled after the map view changed; Ink was kept.",
            )
        elif ink_evidence_was_running:
            self._emit_recovery_state(
                RECOVERY_STATE_INK_FALLBACK,
                "Ink evidence refresh was cancelled; exact cursor remains active.",
            )

    @staticmethod
    def _ensure_edit_session(layer):
        if layer.isEditable():
            return True
        try:
            return bool(layer.startEditing())
        except Exception:
            return False

    @classmethod
    def _run_edit_command(cls, layer, label, operation):
        """Run one undoable edit-buffer mutation and roll it back on failure."""
        if not cls._ensure_edit_session(layer):
            return False

        command_started = False
        try:
            if hasattr(layer, "beginEditCommand"):
                layer.beginEditCommand(label)
                command_started = True
            ok = bool(operation())
            if command_started:
                if ok:
                    layer.endEditCommand()
                else:
                    layer.destroyEditCommand()
            return ok
        except Exception:
            if command_started:
                try:
                    layer.destroyEditCommand()
                # Do not mask the edit-command failure.
                except Exception:  # nosec B110
                    pass
            return False

    def _existing_field_index(self, layer, field_name):
        fields = layer.fields()
        field_idx = fields.indexOf(field_name)
        if field_idx >= 0:
            existing = fields.at(field_idx)
            if field_name in (FIELD_ID, FIELD_ELEVATION) and not existing.isNumeric():
                self._push_message(
                    self._tr(
                        f"필드 '{field_name}'은 숫자 형식이어야 합니다.",
                        f"Field '{field_name}' must be numeric.",
                    ),
                    MESSAGE_CRITICAL,
                )
                return -1
            return field_idx

        return None

    def _ensure_field_in_edit_buffer(self, layer, field_name, field_type):
        """Return a field index, adding it inside an already-open command."""
        field_idx = self._existing_field_index(layer, field_name)
        if field_idx is not None:
            return field_idx

        field = QgsField(field_name, field_type)
        if not layer.addAttribute(field):
            self._push_message(
                self._tr(
                    f"필드 '{field_name}' 추가에 실패했습니다.",
                    f"Failed to add field '{field_name}'.",
                ),
                MESSAGE_CRITICAL,
            )
            return -1

        layer.updateFields()
        field_idx = layer.fields().indexOf(field_name)
        if field_idx < 0:
            self._push_message(
                self._tr(
                    f"필드 '{field_name}'을 확인하지 못했습니다.",
                    f"Could not resolve the added field '{field_name}'.",
                ),
                MESSAGE_CRITICAL,
            )
        return field_idx

    def _ensure_field(self, layer, field_name, field_type):
        """Ensure a field as one standalone QGIS edit command."""
        field_idx = self._existing_field_index(layer, field_name)
        if field_idx is not None:
            return field_idx

        result = {"index": -1}

        def add_field():
            result["index"] = self._ensure_field_in_edit_buffer(
                layer,
                field_name,
                field_type,
            )
            return result["index"] >= 0

        if not self._run_edit_command(
            layer,
            f"ArchaeoTrace add {field_name} field",
            add_field,
        ):
            try:
                layer.updateFields()
            # Preserve the primary edit failure.
            except Exception:  # nosec B110
                pass
            return -1
        return result["index"]

    def _next_feature_id_value(self, layer):
        id_idx = layer.fields().indexOf(FIELD_ID)
        if id_idx < 0:
            return None

        max_id = 0
        for feature in layer.getFeatures():
            try:
                value = int(feature[id_idx])
            except (TypeError, ValueError):
                continue
            max_id = max(max_id, value)
        return max_id + 1

    def _build_feature(self, layer, geometry, elevation=None):
        attrs = {}
        elev_idx = layer.fields().indexOf(FIELD_ELEVATION)
        if elev_idx >= 0 and elevation is not None:
            attrs[elev_idx] = float(elevation)

        feature_geometry = QgsGeometry(geometry)
        if (
            QgsWkbTypes.isMultiType(layer.wkbType())
            and not feature_geometry.isMultipart()
        ):
            feature_geometry.convertToMultiType()

        # createFeature evaluates provider/layer default value clauses before
        # applying our explicit values. Building a full [None, ...] row here
        # would erase unrelated defaults and violate NOT NULL constraints.
        feature = QgsVectorLayerUtils.createFeature(
            layer,
            feature_geometry,
            attrs,
        )

        # Preserve a user/provider default on an existing field named "id".
        # Only supply the plugin's sequential fallback when no default was
        # evaluated and the field can actually store a numeric identifier.
        id_idx = layer.fields().indexOf(FIELD_ID)
        if (
            id_idx >= 0
            and layer.fields().at(id_idx).isNumeric()
            and feature[id_idx] is None
        ):
            next_id = self._next_feature_id_value(layer)
            if next_id is not None:
                feature[id_idx] = next_id
        return feature

    @staticmethod
    def _hard_constraint_failures(layer, feature):
        """Return hard field-constraint failures for a completed feature."""

        failures = []
        for field_idx, field in enumerate(layer.fields()):
            try:
                valid, errors = QgsVectorLayerUtils.validateAttribute(
                    layer,
                    feature,
                    field_idx,
                    HARD_CONSTRAINT,
                )
            except (AttributeError, RuntimeError, TypeError) as exc:
                failures.append((field.name(), [str(exc)]))
                continue
            if not valid:
                failures.append((field.name(), list(errors or ())))
        return failures

    def _validate_feature_constraints(self, layer, feature):
        failures = self._hard_constraint_failures(layer, feature)
        if not failures:
            return True

        detail = "; ".join(
            f"{field_name}: {', '.join(errors) or 'invalid value'}"
            for field_name, errors in failures
        )
        self._push_message(
            self._tr(
                f"필수 필드 제약을 만족하지 못해 저장하지 않았습니다: {detail}",
                f"The feature was not saved because required field constraints failed: {detail}",
            ),
            MESSAGE_CRITICAL,
        )
        return False

    def _add_feature(self, layer, feature):
        if not self._validate_feature_constraints(layer, feature):
            return False
        ok = self._run_edit_command(
            layer,
            "ArchaeoTrace add feature",
            lambda: layer.addFeature(feature),
        )
        if ok:
            layer.updateExtents()
        return ok

    def _add_geometry_feature(self, layer, geometry, elevation=None, label=None):
        """Add an optional elevation field and feature as one Undo command."""

        def add_to_edit_buffer():
            if elevation is not None:
                field_idx = self._ensure_field_in_edit_buffer(
                    layer,
                    FIELD_ELEVATION,
                    _field_type("Double"),
                )
                if field_idx < 0:
                    return False
            feature = self._build_feature(layer, geometry, elevation)
            if not self._validate_feature_constraints(layer, feature):
                return False
            return bool(layer.addFeature(feature))

        ok = self._run_edit_command(
            layer,
            label or "ArchaeoTrace add feature",
            add_to_edit_buffer,
        )
        if ok:
            layer.updateExtents()
        else:
            try:
                layer.updateFields()
            # Preserve the primary edit failure.
            except Exception:  # nosec B110
                pass
        return ok

    def _update_feature(self, layer, feature_id, geometry, attributes=None):
        """Update geometry and attributes together, preserving edit semantics."""

        attributes = dict(attributes or {})

        ok = self._run_edit_command(
            layer,
            "ArchaeoTrace extend feature",
            lambda: self._update_feature_in_edit_buffer(
                layer,
                feature_id,
                geometry,
                attributes,
            ),
        )
        if ok:
            layer.updateExtents()
        return ok

    def _update_feature_in_edit_buffer(self, layer, feature_id, geometry, attributes=None):
        """Apply one feature update inside an already-open edit command."""
        ok = bool(layer.changeGeometry(feature_id, geometry))
        for field_idx, value in dict(attributes or {}).items():
            ok = ok and bool(
                layer.changeAttributeValue(feature_id, field_idx, value)
            )
        if not ok:
            return False
        feature = layer.getFeature(feature_id)
        return feature.isValid() and self._validate_feature_constraints(
            layer,
            feature,
        )

    def _update_geometry(self, layer, feature_id, geometry):
        return self._update_feature(layer, feature_id, geometry)

    def _canvas_extent_in_raster_crs(self):
        self._ensure_crs_transforms_current()
        extent = self.canvas.extent()
        if self.canvas.mapSettings().destinationCrs() == self.raster_layer.crs():
            return extent

        try:
            return self.to_raster_transform.transformBoundingBox(extent)
        except Exception as exc:
            self._push_message(
                self._tr(
                    f"좌표계 변환 실패로 엣지 캐시를 만들지 못했습니다: {exc}",
                    f"Failed to transform extent for edge cache: {exc}",
                ),
                MESSAGE_WARNING,
            )
            return None

    def _map_point_to_raster(self, map_point):
        self._ensure_crs_transforms_current()
        if self.canvas.mapSettings().destinationCrs() == self.raster_layer.crs():
            return QgsPointXY(map_point.x(), map_point.y())

        transformed = self.to_raster_transform.transform(map_point)
        return QgsPointXY(transformed.x(), transformed.y())

    def _raster_point_to_map(self, point):
        self._ensure_crs_transforms_current()
        if self.canvas.mapSettings().destinationCrs() == self.raster_layer.crs():
            return QgsPointXY(point.x(), point.y())

        transformed = self.to_map_transform.transform(point)
        return QgsPointXY(transformed.x(), transformed.y())

    @staticmethod
    def _transform_geometry(geometry, source_crs, target_crs):
        """Return a geometry copy transformed between two valid layer CRSs."""

        if geometry is None or geometry.isEmpty():
            raise ValueError("Cannot transform an empty geometry.")
        if not source_crs.isValid() or not target_crs.isValid():
            raise ValueError("Source and destination CRS must be valid.")

        transformed = QgsGeometry(geometry)
        if source_crs == target_crs:
            return transformed

        coordinate_transform = QgsCoordinateTransform(
            source_crs,
            target_crs,
            QgsProject.instance(),
        )
        result = transformed.transform(coordinate_transform)
        if result is not None:
            try:
                if int(result) != 0:
                    raise ValueError(f"Geometry transform failed with result {result}.")
            except TypeError:
                # Some older bindings expose a non-integer success enum.
                pass
        return transformed

    def _map_geometry_to_layer(self, geometry, layer):
        return self._transform_geometry(
            geometry,
            self.canvas.mapSettings().destinationCrs(),
            layer.crs(),
        )

    def _layer_geometry_to_map(self, geometry, layer):
        return self._transform_geometry(
            geometry,
            layer.crs(),
            self.canvas.mapSettings().destinationCrs(),
        )

    @staticmethod
    def _is_pixel_in_bounds(px, py, width, height):
        return 0 <= int(px) < width and 0 <= int(py) < height

    @staticmethod
    def _clamp_pixel(px, py, width, height):
        return (
            max(0, min(width - 1, int(round(px)))),
            max(0, min(height - 1, int(round(py)))),
        )

    @staticmethod
    def _build_cached_rgb_image(bands):
        if len(bands) >= 3:
            rgb = np.stack(bands[:3], axis=-1)
        else:
            rgb = np.stack([bands[0], bands[0], bands[0]], axis=-1)
        return np.ascontiguousarray(rgb)

    @classmethod
    def _build_stable_integer_rgb_image(cls, bands):
        """Create a pan-stable Recovery RGB cache from native integer DNs."""

        converted = [stable_integer_band_to_uint8(band) for band in bands]
        return cls._build_cached_rgb_image(converted)

    def _sam_trace_config(self):
        """Bind compatibility constants to the shared QGIS-free SAM kernel."""

        return SamTraceConfig(
            mask_min_pixels=self.SAM_MASK_MIN_PIXELS,
            mask_max_area_ratio=self.SAM_MASK_MAX_AREA_RATIO,
            mask_close_kernel=tuple(self.SAM_MASK_CLOSE_KERNEL),
            nearest_active_radius=self.SAM_NEAREST_ACTIVE_RADIUS,
            edge_pixel_threshold=self.EDGE_PIXEL_THRESHOLD,
            outside_cost=self.SAM_OUTSIDE_COST,
            inside_cost=self.SAM_INSIDE_COST,
            edge_cost=self.SAM_EDGE_COST,
            skeleton_cost=self.SAM_SKELETON_COST,
            centerline_bonus=self.SAM_CENTERLINE_BONUS,
            straight_move_cost=self.PATH_MOVE_COST_STRAIGHT,
            diagonal_move_cost=self.PATH_MOVE_COST_DIAGONAL,
            max_iterations_base=self.PATH_MAX_ITER_BASE,
            max_iterations_distance_factor=self.PATH_MAX_ITER_DISTANCE_FACTOR,
            max_dimension=self.CACHE_MAX_DIMENSION,
            smooth_window_size=self.PATH_SMOOTH_WINDOW_SIZE,
            chaikin_iterations=self.CHAIKIN_ITERATIONS,
            neighbors=tuple(self.A_STAR_NEIGHBORS),
        )

    def _ensure_sam_image(self):
        if not self.use_sam or self.sam_engine is None or self.cached_rgb_image is None:
            return False
        if self.sam_image_ready:
            return True

        try:
            self.sam_engine.set_image(self.cached_rgb_image)
            self.sam_image_ready = True
            return True
        except Exception as exc:
            if not self.sam_warning_emitted:
                self._push_message(
                    self._tr(
                        f"SAM 이미지 준비 실패: {exc}",
                        f"Failed to prepare SAM image: {exc}",
                    ),
                    MESSAGE_WARNING,
                )
                self.sam_warning_emitted = True
            return False

    @staticmethod
    def _append_prompt_if_distinct(points, labels, px, py, label, min_distance=3):
        for existing_x, existing_y in points:
            if (existing_x - px) ** 2 + (existing_y - py) ** 2 < min_distance ** 2:
                return
        points.append((int(px), int(py)))
        labels.append(int(label))

    def _build_sam_prompts(self, target_point):
        if self.cache_transform is None or not self.path_points:
            return None, None

        height, width = self.cached_rgb_image.shape[:2]
        recent_points = self.path_points[-(self.SAM_PROMPT_HISTORY_POINTS + 1):]
        prompt_points = []
        prompt_labels = []

        for map_point in recent_points:
            px, py = self.map_to_pixel(map_point)
            if self._is_pixel_in_bounds(px, py, width, height):
                self._append_prompt_if_distinct(prompt_points, prompt_labels, px, py, 1)

        target_px, target_py = self.map_to_pixel(target_point)
        if not self._is_pixel_in_bounds(target_px, target_py, width, height):
            return None, None
        self._append_prompt_if_distinct(prompt_points, prompt_labels, target_px, target_py, 1)

        if len(prompt_points) < 2:
            return None, None

        base_start_x, base_start_y = prompt_points[-2]
        base_end_x, base_end_y = prompt_points[-1]
        direction_x = float(base_end_x - base_start_x)
        direction_y = float(base_end_y - base_start_y)
        if abs(direction_x) + abs(direction_y) < 1.0 and len(prompt_points) >= 3:
            direction_x = float(prompt_points[-1][0] - prompt_points[0][0])
            direction_y = float(prompt_points[-1][1] - prompt_points[0][1])

        norm = math.hypot(direction_x, direction_y)
        if norm > 0:
            perp_x = -direction_y / norm
            perp_y = direction_x / norm
            negative_bases = (prompt_points[-2], prompt_points[-1])
            for base_x, base_y in negative_bases:
                for sign in (-1, 1):
                    neg_x = int(round(base_x + perp_x * self.SAM_NEGATIVE_DISTANCE_PIXELS * sign))
                    neg_y = int(round(base_y + perp_y * self.SAM_NEGATIVE_DISTANCE_PIXELS * sign))
                    if self._is_pixel_in_bounds(neg_x, neg_y, width, height):
                        self._append_prompt_if_distinct(
                            prompt_points,
                            prompt_labels,
                            neg_x,
                            neg_y,
                            0,
                        )

        return (
            np.array(prompt_points, dtype=np.float32),
            np.array(prompt_labels, dtype=np.int32),
        )

    def _build_recovery_prompts(self, target_point):
        """Build the benchmark-parity anchor/target recovery prompt.

        Prior confirmed vertices express incoming direction for Live-Wire;
        they are never additional positive EfficientSAM prompts. This keeps
        product inference aligned with the recovery worker's explicit start
        and end positives while retaining conservative perpendicular
        negatives around only those two endpoints.
        """

        if (
            self.cache_transform is None
            or not self.path_points
            or self.cached_rgb_image is None
        ):
            return None, None
        height, width = self.cached_rgb_image.shape[:2]
        start_px, start_py = self.map_to_pixel(self.path_points[-1])
        target_px, target_py = self.map_to_pixel(target_point)
        if not self._is_pixel_in_bounds(start_px, start_py, width, height):
            return None, None
        if not self._is_pixel_in_bounds(target_px, target_py, width, height):
            return None, None

        try:
            tensors = build_recovery_prompt_tensors(
                (start_px, start_py),
                (target_px, target_py),
                width=width,
                height=height,
            )
        except RecoveryPromptError:
            return None, None
        return tensors.as_numpy(np)

    def _predict_sam_mask(self, target_point):
        if not self._ensure_sam_image():
            return None

        prompt_points, prompt_labels = self._build_sam_prompts(target_point)
        if prompt_points is None or prompt_labels is None:
            return None

        try:
            mask = self.sam_engine.predict_point(prompt_points, prompt_labels)
        except Exception:
            return None

        if mask is None:
            return None

        mask = np.asarray(mask)
        if mask.ndim != 2:
            return None

        return postprocess_sam_mask(
            mask,
            cv2_module=self.cv2,
            np_module=np,
            config=self._sam_trace_config(),
        )

    def _nearest_active_pixel(self, binary_mask, px, py, max_radius=None):
        return find_nearest_active_pixel(
            binary_mask,
            px,
            py,
            max_radius=max_radius,
            config=self._sam_trace_config(),
        )

    def _build_sam_cost_map(self, mask):
        return build_sam_cost_map(
            mask,
            self.cached_edges,
            cv2_module=self.cv2,
            np_module=np,
            thin_binary_mask=EdgeDetector.thin_binary_mask,
            config=self._sam_trace_config(),
        )

    def _a_star_trace_config(self):
        return TraceConfig(
            straight_move_cost=self.PATH_MOVE_COST_STRAIGHT,
            diagonal_move_cost=self.PATH_MOVE_COST_DIAGONAL,
            max_iterations_base=self.PATH_MAX_ITER_BASE,
            max_iterations_distance_factor=self.PATH_MAX_ITER_DISTANCE_FACTOR,
            max_width=self.CACHE_MAX_DIMENSION,
            max_height=self.CACHE_MAX_DIMENSION,
            max_cells=self.CACHE_MAX_DIMENSION * self.CACHE_MAX_DIMENSION,
            validate_all_costs=False,
            validate_accessed_costs=False,
            neighbors=tuple(self.A_STAR_NEIGHBORS),
        )

    def _recovery_trace_config(self, window_bounds):
        """Bind strict Recovery A* to the exact Live-Wire tree window."""

        x0, y0, x1, y1 = (int(value) for value in window_bounds)
        width = x1 - x0
        height = y1 - y0
        if (
            width < 1
            or height < 1
            or width > self.LIVEWIRE_WINDOW_PIXELS
            or height > self.LIVEWIRE_WINDOW_PIXELS
        ):
            raise ValueError("Recovery window must fit the 320px Live-Wire bound")
        return TraceConfig(
            straight_move_cost=self.PATH_MOVE_COST_STRAIGHT,
            diagonal_move_cost=self.PATH_MOVE_COST_DIAGONAL,
            max_iterations_base=self.PATH_MAX_ITER_BASE,
            max_iterations_distance_factor=self.PATH_MAX_ITER_DISTANCE_FACTOR,
            max_width=width,
            max_height=height,
            max_cells=width * height,
            validate_all_costs=False,
            validate_accessed_costs=False,
            neighbors=tuple(self.A_STAR_NEIGHBORS),
        )

    def _run_a_star_path(self, cost_map, start_px, start_py, end_px, end_py, allow_partial=True):
        result = find_path(
            cost_map,
            (start_px, start_py),
            (end_px, end_py),
            allow_partial=allow_partial,
            config=self._a_star_trace_config(),
        )
        return list(result.path), result.used_partial

    def _pixel_path_to_map(self, pixel_path):
        if not pixel_path:
            return []

        smoothed_path = smooth_pixel_path(
            pixel_path,
            window_size=self.PATH_SMOOTH_WINDOW_SIZE,
        )
        return [self.pixel_to_map(point[0], point[1]) for point in smoothed_path]

    def _livewire_config(self):
        return LiveWireConfig(
            max_window_size=self.LIVEWIRE_WINDOW_PIXELS,
            target_snap_radius=self.LIVEWIRE_TARGET_SNAP_PIXELS,
        )

    def _cancel_livewire_task(self):
        task = getattr(self, "_livewire_task", None)
        self._livewire_task = None
        if task is not None:
            try:
                task.cancel()
            except RuntimeError:
                pass

    def _current_livewire_anchor_pixel(self):
        if not self.path_points or self.cache_transform is None:
            return None
        try:
            anchor = self.map_to_pixel(self.path_points[-1])
        except Exception:
            return None
        if self.cached_edges is None:
            return None
        height, width = self.cached_edges.shape
        if not self._is_pixel_in_bounds(anchor[0], anchor[1], width, height):
            return None
        return tuple(anchor)

    def _livewire_incoming_direction(self, anchor_pixel):
        if len(self.path_points) < 2:
            return None
        try:
            previous_pixel = self.map_to_pixel(self.path_points[-2])
        except Exception:
            return None
        direction = (
            anchor_pixel[0] - previous_pixel[0],
            anchor_pixel[1] - previous_pixel[1],
        )
        if direction == (0, 0):
            return None
        return direction

    def _request_livewire_tree(self, force=False):
        """Build one tree for the latest accepted point, if needed."""
        if (
            self._livewire_disabled
            or self.freehand
            or self.use_sam
            or self.edge_weight <= 0.0
            or self.cached_edges is None
            or self.cached_rgb_image is None
            or not self.path_points
        ):
            return False

        if not is_livewire_available():
            self._livewire_disabled = True
            if not self._livewire_warning_emitted:
                self._push_message(
                    self._tr(
                        "SciPy가 없어 방향 인식 Live-Wire 대신 가까운 선 스냅을 사용합니다.",
                        "SciPy is unavailable; using nearby-edge snapping instead of Live-Wire.",
                    ),
                    MESSAGE_WARNING,
                    5,
                )
                self._livewire_warning_emitted = True
            return False

        anchor_pixel = self._current_livewire_anchor_pixel()
        if anchor_pixel is None:
            return False
        if not force and self._livewire_failed_anchor == anchor_pixel:
            return False
        if (
            not force
            and self._livewire_tree is not None
            and self._livewire_tree.root == anchor_pixel
        ):
            return True
        if (
            not force
            and self._livewire_task is not None
            and self._livewire_task.anchor_pixel == anchor_pixel
        ):
            return True

        self._cancel_livewire_task()
        self._livewire_generation += 1
        generation = self._livewire_generation
        self._livewire_tree = None
        self._livewire_anchor_pixel = anchor_pixel
        task = _LiveWireTreeTask(
            image=self.cached_rgb_image,
            edges=self.cached_edges,
            evidence=self.cached_ink_evidence,
            anchor_pixel=anchor_pixel,
            incoming_direction=self._livewire_incoming_direction(anchor_pixel),
            strength=self.edge_weight,
            generation=generation,
            config=self._livewire_config(),
            callback=self._on_livewire_tree_finished,
        )
        self._livewire_task = task
        QgsApplication.taskManager().addTask(task)
        return True

    def _on_livewire_tree_finished(self, task, succeeded, tree, error):
        """Publish a current tree and redraw the latest cursor proposal."""
        if self._livewire_task is not task:
            return
        self._livewire_task = None

        current_anchor = self._current_livewire_anchor_pixel()
        is_current = (
            succeeded
            and tree is not None
            and task.generation == self._livewire_generation
            and current_anchor == task.anchor_pixel
            and self.is_tracing
        )
        if is_current:
            self._livewire_tree = tree
            self._livewire_failed_anchor = None
            request_point = self._livewire_request_point
            if request_point is not None and not self.use_sam:
                self._present_livewire_cursor_preview(
                    request_point,
                    global_mode=self.auto_path,
                    request_tree=False,
                )
            return

        if isinstance(error, LiveWireUnavailable):
            self._livewire_disabled = True
            if not self._livewire_warning_emitted:
                self._push_message(
                    self._tr(
                        "SciPy Live-Wire를 시작할 수 없어 가까운 선 스냅을 사용합니다.",
                        "SciPy Live-Wire could not start; using nearby-edge snapping.",
                    ),
                    MESSAGE_WARNING,
                    5,
                )
                self._livewire_warning_emitted = True
            return

        if error is not None:
            self._livewire_failed_anchor = task.anchor_pixel
            print(f"Live-Wire tree build failed: {error}")

        # A drag or click may have advanced the accepted anchor while the old
        # tree was building. Coalesce that state into one fresh build.
        if (
            error is None
            and self.is_tracing
            and current_anchor is not None
            and current_anchor != task.anchor_pixel
        ):
            self._request_livewire_tree(force=False)

    def _livewire_preview_path(self, target_point, request_tree=True):
        """Return a cursor-led path; tree lookup is sub-millisecond."""
        if self.edge_weight <= 0.0 or self.cached_edges is None:
            return [QgsPointXY(target_point)]

        anchor_pixel = self._current_livewire_anchor_pixel()
        if request_tree:
            self._request_livewire_tree(force=False)
        tree = self._livewire_tree
        if tree is None or anchor_pixel is None or tree.root != anchor_pixel:
            return [self.angle_constrained_snap(target_point)]

        try:
            target_pixel = self.map_to_pixel_float(target_point)
            pixel_path = tree.trace(target_pixel)
            if len(pixel_path) < 2:
                return [self.angle_constrained_snap(target_point)]

            # Suppress 8-neighbour staircase artifacts while preserving the
            # exact anchor and strength-blended endpoint.
            if len(pixel_path) > self.LIVEWIRE_SMOOTH_WINDOW_SIZE:
                smoothed = list(
                    smooth_pixel_path(
                        pixel_path,
                        window_size=self.LIVEWIRE_SMOOTH_WINDOW_SIZE,
                    )
                )
                smoothed[0] = pixel_path[0]
                smoothed[-1] = pixel_path[-1]
                pixel_path = smoothed

            map_path = [self.pixel_to_map(x, y) for x, y in pixel_path]
            # LiveWireTree always includes its root. The rubber band prepends
            # the exact accepted map point, so discard the pixel-quantized
            # root unconditionally.
            if len(map_path) >= 2:
                map_path = map_path[1:]
            return map_path or [QgsPointXY(target_point)]
        except Exception as exc:
            print(f"Live-Wire preview failed: {exc}")
            return [self.angle_constrained_snap(target_point)]

    def _present_livewire_cursor_preview(
        self,
        target_point,
        *,
        global_mode=False,
        request_tree=True,
    ):
        self._livewire_request_point = QgsPointXY(target_point)
        self.preview_path = self._livewire_preview_path(
            target_point,
            request_tree=request_tree,
        )
        self.preview_is_global = bool(global_mode)
        self.preview_target = QgsPointXY(target_point) if global_mode else None
        self._render_preview()
        if self.smart_recovery_enabled:
            self._schedule_smart_recovery(target_point, force=False)

    def _recovery_pixel_path(self):
        """Return the visible Ink champion in the current evidence grid."""

        evidence = self.cached_ink_evidence
        if evidence is None or not self.path_points or not self.preview_path:
            return None
        height, width = evidence.center_score.shape
        map_path = [self.path_points[-1], *self.preview_path]
        pixel_path = []
        for map_point in map_path:
            px, py = self.map_to_pixel_float(map_point)
            px = float(px)
            py = float(py)
            if not (0.0 <= px < width and 0.0 <= py < height):
                return None
            point = (px, py)
            if not pixel_path or point != pixel_path[-1]:
                pixel_path.append(point)
        return tuple(pixel_path) if len(pixel_path) >= 2 else None

    def _build_recovery_request(
        self,
        target_point,
        *,
        force,
        champion_path,
        request_generation,
        preview_identity,
    ):
        if champion_path is None or self.cached_rgb_image is None:
            return None
        height, width = self.cached_rgb_image.shape[:2]
        start_pixel = champion_path[0]
        target_pixel = self.map_to_pixel_float(target_point)
        target_pixel = tuple(float(value) for value in target_pixel)
        if not (
            0.0 <= target_pixel[0] < width
            and 0.0 <= target_pixel[1] < height
        ):
            return None

        prompt_points, prompt_labels = self._build_recovery_prompts(target_point)
        if prompt_points is None or prompt_labels is None:
            return None
        gate = recovery_gate(
            champion_path,
            self.cached_ink_evidence,
            expected_start=start_pixel,
            expected_end=target_pixel,
            force=bool(force),
        )
        if not gate.trigger:
            return {
                "trigger": False,
                "reason": gate.reason,
            }
        tree = self._livewire_tree
        if tree is None:
            return None
        origin_x, origin_y = (int(value) for value in tree.origin)
        tree_height, tree_width = (int(value) for value in tree.shape)
        window_bounds = (
            origin_x,
            origin_y,
            origin_x + tree_width,
            origin_y + tree_height,
        )
        if not tree.contains(start_pixel) or not tree.contains(target_pixel):
            return None
        return {
            "trigger": True,
            "reason": gate.reason,
            "target_map": QgsPointXY(target_point),
            "champion_path": champion_path,
            "start_pixel": start_pixel,
            "target_pixel": target_pixel,
            "prompt_points": prompt_points,
            "prompt_labels": prompt_labels,
            "window_bounds": window_bounds,
            "cache_generation": self._cache_generation,
            "request_generation": int(request_generation),
            "preview_identity": preview_identity,
        }

    def _start_recovery_request(self, request):
        if (
            not self.smart_recovery_enabled
            or self.recovery_engine is None
            or request is None
            or not request.get("trigger")
            or not self.is_tracing
        ):
            return False
        if "request_generation" not in request:
            self._recovery_generation += 1
            request["request_generation"] = self._recovery_generation
        if "preview_identity" not in request:
            request["preview_identity"] = (
                int(request["cache_generation"]),
                tuple(tuple(point) for point in request["champion_path"]),
                tuple(request["target_pixel"]),
            )
            self._recovery_preview_identity = request["preview_identity"]
        if self._recovery_task is not None:
            self._recovery_request = request
            return True

        request_generation = request["request_generation"]
        self._recovery_request = request
        encoding = (
            self._recovery_encoding
            if self._recovery_encoding_cache_generation == self._cache_generation
            else None
        )
        task = _RecoveryPreviewTask(
            engine=self.recovery_engine,
            image=self.cached_rgb_image,
            encoding=encoding,
            evidence=self.cached_ink_evidence,
            champion_path=request["champion_path"],
            start_pixel=request["start_pixel"],
            target_pixel=request["target_pixel"],
            prompt_points=request["prompt_points"],
            prompt_labels=request["prompt_labels"],
            window_bounds=request["window_bounds"],
            cache_generation=request["cache_generation"],
            request_generation=request_generation,
            preview_identity=request["preview_identity"],
            smooth_window_size=self.PATH_SMOOTH_WINDOW_SIZE,
            trace_config=self._recovery_trace_config(
                request["window_bounds"]
            ),
            callback=self._on_recovery_preview_finished,
        )
        self._recovery_task = task
        self._emit_recovery_state(
            RECOVERY_STATE_RECOVERING,
            "Ink evidence is weak; evaluating an optional local corridor.",
        )
        QgsApplication.taskManager().addTask(task)
        return True

    def _schedule_smart_recovery(self, target_point, *, force=False):
        """Gate a challenger only after the Ink champion is available."""

        if not self.smart_recovery_enabled:
            return False
        if not self._recovery_cache_compatible:
            self._emit_recovery_state(
                RECOVERY_STATE_INK_FALLBACK,
                self._recovery_cache_disabled_reason
                or "Smart Recovery is unavailable for this raster; Ink was kept.",
            )
            return False
        # Every visible Ink champion advances identity before any gate can
        # return. A previous ONNX task is allowed to finish, but it becomes
        # stale even when this newer route is confident or lacks evidence.
        self._recovery_generation += 1
        request_generation = self._recovery_generation
        champion_path = self._recovery_pixel_path()
        try:
            target_identity = tuple(
                float(value) for value in self.map_to_pixel_float(target_point)
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            target_identity = (
                float(target_point.x()),
                float(target_point.y()),
            )
        preview_identity = (
            int(self._cache_generation),
            tuple(
                tuple(float(value) for value in point)
                for point in (champion_path or ())
            ),
            target_identity,
        )
        self._recovery_preview_identity = preview_identity
        self._recovery_request = None

        anchor_pixel = self._current_livewire_anchor_pixel()
        if (
            self._livewire_disabled
            or self._livewire_tree is None
            or anchor_pixel is None
            or self._livewire_tree.root != anchor_pixel
        ):
            self._emit_recovery_state(
                RECOVERY_STATE_INK,
                "Ink preview updated; recovery is waiting for line evidence.",
            )
            return False
        try:
            request = self._build_recovery_request(
                target_point,
                force=force,
                champion_path=champion_path,
                request_generation=request_generation,
                preview_identity=preview_identity,
            )
        except Exception as exc:
            self._emit_recovery_state(
                RECOVERY_STATE_INK_FALLBACK,
                f"Recovery gate failed; Ink was kept: {exc}",
            )
            return False
        if request is None:
            detail = (
                "No current Ink segment is available to retry."
                if force
                else "Recovery is unavailable for this segment; Ink was kept."
            )
            self._emit_recovery_state(
                RECOVERY_STATE_INK_FALLBACK,
                detail,
            )
            return False
        if not request.get("trigger"):
            self._emit_recovery_state(
                RECOVERY_STATE_INK,
                "Ink evidence is confident; recovery was not run.",
            )
            return False
        return self._start_recovery_request(request)

    def _on_recovery_preview_finished(
        self,
        task,
        succeeded,
        trace_result,
        challenger_path,
        selection,
        error,
    ):
        """Publish a challenger only when the conservative arbiter accepts."""

        if self._recovery_task is not task:
            return
        self._recovery_task = None
        if getattr(self, "_disposed", False):
            self._recovery_request = None
            return
        latest_request = self._recovery_request
        result_is_current = (
            succeeded
            and error is None
            and task.request_generation == self._recovery_generation
            and task.cache_generation == self._cache_generation
            and task.preview_identity == self._recovery_preview_identity
            and self.is_tracing
            and self.smart_recovery_enabled
        )
        if result_is_current:
            self._recovery_encoding = task.encoding
            self._recovery_encoding_cache_generation = task.cache_generation
            if (
                selection is not None
                and selection.accepted
                and trace_result is not None
                and trace_result.reached_target
                and challenger_path is not None
                and len(challenger_path) >= 2
            ):
                # The task already smoothed and arbitrated this exact global
                # route. Convert without a second geometry-changing pass and
                # omit its root because _render_preview prepends the anchor.
                self.preview_path = [
                    self.pixel_to_map(x, y)
                    for x, y in challenger_path[1:]
                ]
                self.preview_is_global = bool(self.auto_path)
                self.preview_target = None
                # Keep the existing map-space acceptance target in Auto Path;
                # ordinary Ink still accepts the enhanced green line once.
                if self.auto_path and latest_request is not None:
                    self.preview_target = QgsPointXY(latest_request["target_map"])
                self._render_preview()
                self._recovery_request = None
                self._emit_recovery_state(
                    RECOVERY_STATE_ENHANCED,
                    "A safer corridor improved the weak Ink segment.",
                )
                return

            reason = getattr(selection, "reason", "no_complete_route")
            self._recovery_request = None
            self._emit_recovery_state(
                RECOVERY_STATE_INK_FALLBACK,
                f"Recovery was rejected ({reason}); Ink was kept.",
            )
            return

        # A cursor move coalesces into exactly one fresh request after the old
        # ONNX call returns. A cancelled/error result never alters the Ink
        # preview already visible on the canvas.
        if (
            latest_request is not None
            and latest_request.get("request_generation") != task.request_generation
            and latest_request.get("cache_generation") == self._cache_generation
            and self.is_tracing
            and self.smart_recovery_enabled
        ):
            self._start_recovery_request(latest_request)
            return

        # A later confident/empty Ink preview intentionally has no queued
        # challenger. Its visible route and state must remain untouched by a
        # stale completion from the previous cursor target.
        if (
            task.request_generation != self._recovery_generation
            or task.preview_identity != self._recovery_preview_identity
        ):
            return

        detail = "Recovery was cancelled; Ink was kept."
        if error is not None:
            detail = f"Recovery failed; Ink was kept: {error}"
        self._emit_recovery_state(RECOVERY_STATE_INK_FALLBACK, detail)

    def retry_current_segment(self):
        """Explicitly re-run recovery for the current uncommitted Ink route."""

        if not self.smart_recovery_requested:
            self._emit_recovery_state(
                RECOVERY_STATE_INK_FALLBACK,
                "Enable Smart Recovery before retrying the current segment.",
            )
            return False
        if not self.smart_recovery_enabled:
            self._emit_recovery_state(
                RECOVERY_STATE_INK_FALLBACK,
                "Recovery model is unavailable; Ink remains active.",
            )
            return False
        if self._current_recovery_state == RECOVERY_STATE_ENHANCED:
            self._emit_recovery_state(
                RECOVERY_STATE_ENHANCED,
                "This segment is already enhanced; move the cursor to create a new Ink champion.",
            )
            return False
        target = self._livewire_request_point or self.last_hover_pos
        if not self.is_tracing or target is None:
            self._emit_recovery_state(
                RECOVERY_STATE_INK_FALLBACK,
                "Draw or hover an Ink segment before retrying recovery.",
            )
            return False
        return self._schedule_smart_recovery(target, force=True)

    def _find_sam_path(self, target_point):
        if not self.use_sam or self.cached_rgb_image is None or not self.path_points:
            return []

        try:
            start_px, start_py = self.map_to_pixel(self.path_points[-1])
            target_px, target_py = self.map_to_pixel(target_point)
            height, width = self.cached_rgb_image.shape[:2]
            if not self._is_pixel_in_bounds(start_px, start_py, width, height):
                return []
            if not self._is_pixel_in_bounds(target_px, target_py, width, height):
                return []

            sam_mask = self._predict_sam_mask(target_point)
            if sam_mask is None:
                return []

            traced = trace_sam_mask(
                sam_mask,
                self.cached_edges,
                (start_px, start_py),
                (target_px, target_py),
                cv2_module=self.cv2,
                np_module=np,
                thin_binary_mask=EdgeDetector.thin_binary_mask,
                config=self._sam_trace_config(),
            )
            if traced is None or not traced.path:
                return []
            return self._pixel_path_to_map(traced.path)
        except Exception:
            return []

    def _render_preview(self):
        """Render the exact candidate segment without committing it."""
        self.preview_band.reset(LINE_GEOMETRY)
        if self._preview_style_is_global != self.preview_is_global:
            if self.preview_is_global:
                self.preview_band.setColor(QColor(*self.PROPOSAL_BAND_COLOR))
                self.preview_band.setWidth(self.PROPOSAL_BAND_WIDTH)
                self.preview_band.setLineStyle(self.PROPOSAL_BAND_LINE_STYLE)
            else:
                self.preview_band.setColor(QColor(*self.PREVIEW_BAND_COLOR))
                self.preview_band.setWidth(self.PREVIEW_BAND_WIDTH)
                self.preview_band.setLineStyle(self.PREVIEW_BAND_LINE_STYLE)
            self._preview_style_is_global = self.preview_is_global

        if not self.preview_path:
            return

        render_points = []
        if self.path_points:
            render_points.append(self.path_points[-1])
        render_points.extend(self.preview_path)
        if len(render_points) < 2:
            return

        # One C++ geometry update is considerably cheaper than crossing the
        # Python/QGIS boundary once for every route vertex. Long A*/SAM
        # previews used to become visible only after thousands of addPoint
        # calls had completed.
        self.preview_band.setToGeometry(
            QgsGeometry.fromPolylineXY(render_points),
            None,
        )

    def _clear_preview(self, stop_timer=True):
        """Discard an uncommitted candidate segment."""
        if stop_timer:
            self._proposal_timer.stop()
            self._cancel_proposal_task()
        self._proposal_generation += 1
        self._proposal_request_point = None
        self.preview_path = []
        self.preview_is_global = False
        self.preview_target = None
        self._render_preview()

    def _schedule_auto_path_preview(self, point):
        """Debounce an actual A*/SAM proposal until the cursor pauses."""
        if not self.auto_path or not self.is_tracing or not self.path_points:
            return

        self._proposal_generation += 1
        self._proposal_request_point = QgsPointXY(point)
        if self.edge_weight <= 0.0:
            # The slider contract also applies to explicit SAM mode. At zero,
            # do not run the model and make the literal green segment
            # immediately acceptable with one click.
            self._proposal_timer.stop()
            self._cancel_proposal_task()
            self.preview_path = [QgsPointXY(point)]
            self.preview_is_global = True
            self.preview_target = QgsPointXY(point)
            self._render_preview()
            return
        self.preview_path = [
            self.angle_constrained_snap(point)
            if self.cached_edges is not None
            else point
        ]
        self.preview_is_global = False
        self.preview_target = None
        self._render_preview()
        self._proposal_timer.start()

    def _cancel_proposal_task(self):
        task = self._proposal_task
        self._proposal_task = None
        if task is not None:
            try:
                task.cancel()
            except RuntimeError:
                pass

    def _present_auto_path_preview(self, proposed_path, target_point):
        """Display one completed proposal on the QGIS main thread."""
        proposed_path = list(proposed_path or [])
        if len(proposed_path) > 2:
            proposed_path = self.smooth_bezier(proposed_path, closed=False)

        if proposed_path and self.path_points:
            anchor = self.path_points[-1]
            if proposed_path[0] != anchor:
                proposed_path.insert(0, anchor)
            blended = blend_path_with_cursor(
                ((point.x(), point.y()) for point in proposed_path),
                (anchor.x(), anchor.y()),
                (target_point.x(), target_point.y()),
                self.edge_weight,
            )
            proposed_path = [QgsPointXY(x, y) for x, y in blended[1:]]
        if not proposed_path:
            proposed_path = [QgsPointXY(target_point)]

        self.preview_path = list(proposed_path)
        self.preview_is_global = True
        self.preview_target = QgsPointXY(target_point)
        self._render_preview()

    def _set_auto_path_preview(self, target_point):
        """Calculate and display one complete route proposal synchronously."""
        self._present_auto_path_preview(
            self.find_optimal_path(target_point),
            target_point,
        )

    def _start_background_a_star_preview(self, target_point, generation):
        """Start one coalesced A* preview task using only immutable data."""
        if self._proposal_task is not None or self.cached_cost is None:
            return False

        try:
            start_px, start_py = self.map_to_pixel(self.path_points[-1])
            target_px, target_py = self.map_to_pixel(target_point)
        except Exception:
            return False

        task = _AStarPreviewTask(
            cost_map=self.cached_cost,
            start_pixel=(start_px, start_py),
            target_pixel=(target_px, target_py),
            target_xy=(target_point.x(), target_point.y()),
            generation=generation,
            config=self._a_star_trace_config(),
            callback=self._on_background_a_star_finished,
        )
        self._proposal_task = task
        QgsApplication.taskManager().addTask(task)
        return True

    def _on_background_a_star_finished(self, task, succeeded, trace_result, error):
        """Publish a current result and immediately coalesce a stale one."""
        if self._proposal_task is not task:
            return
        self._proposal_task = None

        result_is_current = (
            succeeded
            and trace_result is not None
            and task.generation == self._proposal_generation
            and self.is_tracing
            and self.auto_path
            and bool(self.path_points)
        )
        if result_is_current:
            if trace_result.used_partial:
                self._push_message(
                    self._tr(
                        "경로 탐색 시간이 초과되어 단순화된 경로를 사용했습니다. (확대해서 시도해보세요)",
                        "Pathfinding timeout - simplified path used (Try zooming in)",
                    ),
                    MESSAGE_WARNING,
                    self.PATH_TIMEOUT_MESSAGE_SECONDS,
                )
            target_point = QgsPointXY(*task.target_xy)
            self._present_auto_path_preview(
                self._pixel_path_to_map(trace_result.path),
                target_point,
            )
            return

        if error is not None:
            print(f"Background path preview failed: {error}")

        # The cursor may have moved while this task was running. Start exactly
        # one new task for the newest request instead of queueing every mouse
        # event and allowing old results to overwrite the current preview.
        if (
            self._proposal_request_point is not None
            and self.is_tracing
            and self.auto_path
            and self.path_points
            and task.generation != self._proposal_generation
        ):
            self._update_auto_path_preview()

    def _update_auto_path_preview(self):
        """Update the candidate after the debounced cursor pause."""
        request_point = self._proposal_request_point
        generation = self._proposal_generation
        if (
            request_point is None
            or not self.is_tracing
            or not self.auto_path
            or not self.path_points
        ):
            return

        # SAM prediction remains a model-specific synchronous path for now.
        # Ordinary edge/A* search is QGIS-free and safe to run as a task,
        # keeping the live cursor line and map canvas responsive.
        if not self.use_sam and self.cached_cost is not None:
            self._start_background_a_star_preview(request_point, generation)
            return

        self._set_auto_path_preview(request_point)

    def _proposal_target_matches(self, point):
        if not self.preview_is_global or self.preview_target is None:
            return False
        distance = math.hypot(
            point.x() - self.preview_target.x(),
            point.y() - self.preview_target.y(),
        )
        return distance <= (
            self.canvas.mapUnitsPerPixel() * self.PROPOSAL_ACCEPT_TOLERANCE_PIXELS
        )

    def _accept_or_prepare_auto_path(self, point):
        """Accept a visible proposal, or show one for the first click."""
        if self._proposal_target_matches(point):
            self._proposal_timer.stop()
            self._cancel_proposal_task()
            self._proposal_generation += 1
            self._proposal_request_point = None
            accepted = list(self.preview_path)
            if accepted and accepted[0] == self.path_points[-1]:
                accepted = accepted[1:]
            self.path_points.extend(accepted or [QgsPointXY(point)])
            self.preview_path = []
            self.preview_is_global = False
            self.preview_target = None
            self._render_preview()
            return True

        self._proposal_timer.stop()
        self._cancel_proposal_task()
        self._proposal_generation += 1
        self._proposal_request_point = None
        self._set_auto_path_preview(point)
        self._push_message(
            self._tr(
                "경로 제안을 표시했습니다. 같은 위치를 다시 클릭하면 채택됩니다.",
                "Proposal shown. Click the same target again to accept it.",
            ),
            MESSAGE_INFO,
            3,
        )
        return False

    def canvasPressEvent(self, event):
        if event.button() == _qt_value("RightButton", "MouseButton"):
            # Right click = Finish Line (Enter)
            if not self.is_tracing:
                return

            # If there's a preview path (green line), DO NOT include it
            # User request: "삐져나온 초록선이 거슬린다" -> Only save clicked points
            # if self.preview_path:
            #    self.path_points.extend(self.preview_path)

            if len(self.path_points) >= 2:
                elevation = self.ask_elevation()
                if elevation is None:
                    return
                if self.save_to_layer(closed=False, elevation=elevation):
                    self.reset_tracing()
                return

            self.reset_tracing()
            return

        if event.button() != _qt_value("LeftButton", "MouseButton"):
            return

        point = self.toMapCoordinates(event.pos())

        if not self.is_tracing:
            self._invalidate_recovery("Ink tracing started.")
            # Start tracing

            # Check if snapping to existing endpoint (Resume)
            snapped_start, feat_id, is_start = self.snap_to_existing_endpoint(point)

            self.resume_feature_id = feat_id
            self.resume_at_start = is_start

            if snapped_start:
                place_point = snapped_start
            else:
                place_point = point

            self.is_tracing = True
            self.start_point = place_point
            self.last_map_point = place_point
            self.last_input_point = place_point
            self.last_hover_pos = None
            self.last_preview_pos = event.pos()
            self.path_points = [place_point]
            self.checkpoints = [0]  # Start point is first checkpoint
            self.last_sample_pos = event.pos()

            # Show start marker
            self.start_marker.reset(POINT_GEOMETRY)
            self.start_marker.addPoint(place_point)
            self.snap_marker.reset(POINT_GEOMETRY)  # Hide snap marker

            # Reset checkpoint markers
            self.checkpoint_markers.reset(POINT_GEOMETRY)

            # Update edge cache
            if self._needs_edge_cache():
                self.update_edge_cache()
                self._request_livewire_tree(force=False)

            self.confirm_band.reset(LINE_GEOMETRY)
            self.confirm_band.addPoint(place_point)
            self.preview_band.reset(LINE_GEOMETRY)
        else:
            # A click accepts the currently visible Ink/enhanced candidate.
            # Any still-running challenger belongs to the previous anchor.
            self._invalidate_recovery("Segment accepted; Ink is the new champion.")
            # Preserve the existing double-click spot-height gesture before
            # Auto Path's two-click proposal acceptance can intercept it.
            if self.is_near_start(point) and len(self.path_points) == 1:
                elevation = self.ask_elevation()
                if elevation is None:
                    return
                if self.create_spot_height(self.start_point, elevation):
                    self.reset_tracing()
                return

            auto_path_accepted = False
            if self.auto_path and self.use_sam:
                # A visible complete proposal is accepted by clicking its
                # target. If the route is not ready yet, this click only
                # displays it and the next click becomes the acceptance.
                if not self._accept_or_prepare_auto_path(point):
                    return
                auto_path_accepted = True

            # Check if closing polygon (near start)
            if self.is_near_start(point):
                # SPECIAL CASE: Double Click on Start Point = Spot Height
                if len(self.path_points) == 1:
                    elevation = self.ask_elevation()
                    if elevation is None:
                        return
                    if self.create_spot_height(self.start_point, elevation):
                        self.reset_tracing()
                    return

                # Preserve WYSIWYG for the closing segment too. SAM keeps its
                # explicit proposal flow; Live-Wire commits the green route
                # already shown under the cursor.
                if self.auto_path and self.use_sam and not auto_path_accepted:
                    closing_path = self.find_optimal_path(self.start_point)
                    if len(closing_path) > 2:
                        closing_path = self.smooth_bezier(closing_path, closed=False)
                elif self.auto_path and self.use_sam:
                    closing_path = []
                elif self.preview_path:
                    closing_path = list(self.preview_path)
                    closing_path[-1] = self.start_point
                else:
                    closing_path = [self.start_point]

                self.path_points.extend(closing_path)

                # Check for duplicate end point and remove to prevent artifact
                if len(self.path_points) > 1 and self.path_points[-1] == self.path_points[0]:
                    self.path_points.pop()

                # Ask for elevation value
                elevation = self.ask_elevation()
                if elevation is None:
                    return

                if self.save_to_layer(closed=True, elevation=elevation):
                    self.reset_tracing()
                return

            # ADD CHECKPOINT: only the visible, explicitly accepted proposal
            # or the human-led preview becomes part of the confirmed path.
            if auto_path_accepted:
                # The accepted proposal was already appended above.
                self.preview_path = []
            elif self.preview_path:
                # Commit SMOOTHED AI path (WYSIWYG)
                # preview_path is already smoothed in canvasMoveEvent
                self.path_points.extend(self.preview_path)
                self.preview_path = []
            else:
                # Manual click point
                # If manual mode is active, we might have just clicked.
                # If path empty, user clicked start. If path not empty, user is adding points.
                if len(self.path_points) > 0:
                    # If points exist, add straight line to click
                    self.path_points.append(point)

            if self.path_points:
                self.last_input_point = point
                self.last_map_point = self.path_points[-1]

            # Add checkpoint
            self.checkpoints.append(len(self.path_points) - 1)
            self.checkpoint_markers.addPoint(self.path_points[-1])

            # Confirm current preview path
            self.redraw_confirmed_path()
            self.last_sample_pos = event.pos()
            self._livewire_request_point = None
            self._request_livewire_tree(force=True)

    def canvasMoveEvent(self, event):
        current_point = self.toMapCoordinates(event.pos())
        # Keep the raw cursor as the source of truth.  The previous 70/30 EMA
        # introduced visible cursor lag and leaked pre-trace hover state into
        # the first segment.  Local snapping below is enough to tame detector
        # jitter without delaying the user's input.
        self.last_hover_pos = current_point

        # 1. NOT TRACING: Check for Snap-to-Resume
        if not self.is_tracing:
            snapped, _, _ = self.snap_to_existing_endpoint(current_point)  # Use raw point for snapping (snappier)
            self.snap_marker.reset(POINT_GEOMETRY)
            if snapped:
                self.snap_marker.addPoint(snapped)
            return

        # 2. TRACING ACTIVE

        # Check close indicator
        if self.is_near_start(current_point):
            self.close_indicator.reset(POINT_GEOMETRY)
            self.close_indicator.addPoint(self.start_point)
        else:
            self.close_indicator.reset(POINT_GEOMETRY)

        if self.last_map_point is None:
            self.last_map_point = current_point
            self.last_input_point = current_point
            return

        # MODE CHECK: Dragging vs Hovering
        is_manual_mode = (
            event.modifiers()
            & (
                _qt_value("ShiftModifier", "KeyboardModifier")
                | _qt_value("ControlModifier", "KeyboardModifier")
            )
        )
        interaction_mode = resolve_interaction_mode(
            freehand=self.freehand,
            auto_path=self.auto_path,
            manual_override=bool(is_manual_mode),
        )

        is_dragging = bool(
            event.buttons() & _qt_value("LeftButton", "MouseButton")
        )

        # Preview motion and committed drag sampling are separate concerns.
        # The old Auto Path branch throttled the visible green cursor line to
        # one update per 12 screen pixels, which made it visibly trail behind
        # the mouse before any AI work even began.
        if uses_global_path_search(interaction_mode) and self.use_sam:
            if self.last_preview_pos is not None:
                preview_dx = event.pos().x() - self.last_preview_pos.x()
                preview_dy = event.pos().y() - self.last_preview_pos.y()
                if (
                    preview_dx * preview_dx + preview_dy * preview_dy
                    < self.PREVIEW_INTERVAL_PIXELS * self.PREVIEW_INTERVAL_PIXELS
                ):
                    return
            self.last_preview_pos = event.pos()
            self.last_input_point = current_point
            self._schedule_auto_path_preview(current_point)
            return

        if is_dragging:
            if self._recovery_task is not None or self._recovery_request is not None:
                self._invalidate_recovery("Manual drawing kept the Ink path.")
            # Committed freehand/local-assist points remain lightly sampled so
            # long traces do not accumulate one vertex per OS mouse event.
            if self.last_sample_pos is None:
                self.last_sample_pos = event.pos()
            screen_dx = event.pos().x() - self.last_sample_pos.x()
            screen_dy = event.pos().y() - self.last_sample_pos.y()
            if (
                screen_dx * screen_dx + screen_dy * screen_dy
                < self.SAMPLE_INTERVAL_PIXELS * self.SAMPLE_INTERVAL_PIXELS
            ):
                return
            self.last_sample_pos = event.pos()

            # DRAGGING: Manual Draw (Mouse Following + Gentle Snap)
            self.preview_path = []
            self.preview_is_global = False
            self.preview_target = None
            self._proposal_timer.stop()
            self._proposal_generation += 1
            self._livewire_request_point = None

            # If Manual Mode (Shift/Ctrl): No snapping, just exact mouse pos
            if interaction_mode in (MODE_MOUSE_ASSIST, MODE_AUTO_PATH) and self.cached_edges is not None:
                final_point = self.angle_constrained_snap(current_point)
            else:
                final_point = current_point

            self.path_points.append(final_point)
            self.last_input_point = current_point
            self.last_map_point = current_point
            self.preview_band.reset(LINE_GEOMETRY)
            self.confirm_band.addPoint(final_point)
        else:
            # HOVERING (Not Dragging)

            if self.last_preview_pos is not None:
                preview_dx = event.pos().x() - self.last_preview_pos.x()
                preview_dy = event.pos().y() - self.last_preview_pos.y()
                if (
                    preview_dx * preview_dx + preview_dy * preview_dy
                    < self.PREVIEW_INTERVAL_PIXELS * self.PREVIEW_INTERVAL_PIXELS
                ):
                    return
            self.last_preview_pos = event.pos()

            # 1. Not Tracing yet? Check for Resume Snap
            if not self.path_points:
                snap_pt, snap_fid, is_start = self.snap_to_existing_endpoint(current_point)
                if snap_pt:
                    self.snap_marker.reset(POINT_GEOMETRY)
                    self.snap_marker.addPoint(snap_pt)
                    if self.iface:
                        self.iface.mapCanvas().setCursor(
                            _qt_value("PointingHandCursor", "CursorShape")
                        )
                else:
                    self.snap_marker.reset(POINT_GEOMETRY)
                    if self.iface:
                        self.iface.mapCanvas().setCursor(
                            _qt_value("CrossCursor", "CursorShape")
                        )
                return

            # 2. Tracing: Prediction Logic
            if interaction_mode == MODE_FREEHAND:
                # MANUAL MODE PREVIEW: Literal straight line
                self._livewire_request_point = None
                smoothed_preview = [current_point]
            else:
                # The expensive tree was built once at the accepted anchor;
                # this cursor path is only a predecessor traceback. While a
                # new tree is still building, a direct/local-snap segment is
                # shown immediately and replaced in place when ready.
                self._present_livewire_cursor_preview(
                    current_point,
                    global_mode=(interaction_mode == MODE_AUTO_PATH),
                )
                self.last_input_point = current_point
                return

            self.preview_path = smoothed_preview
            self.preview_is_global = False
            self.preview_target = None
            # Advance the throttle anchor after each accepted preview.  The
            # old hover branch kept measuring from the last committed point,
            # which made it effectively run on every mouse event.
            self.last_input_point = current_point

            # Draw preview (Green line)
            self._render_preview()

    def angle_constrained_snap(self, map_point):
        """
        Smart gently snap that checks ANGLE continuity.
        Only snaps to edge if it continues the current line naturally.
        Prevents jumping to perpendicular noise (broken glass effect).
        """
        if self.cached_edges is None:
            return map_point

        try:
            px, py = self.map_to_pixel(map_point)
            h, w = self.cached_edges.shape

            if px < 0 or py < 0 or px >= w or py >= h:
                return map_point

            if self.edge_weight <= 0.0:
                return map_point

            # Search a deliberately small neighborhood with NumPy instead of
            # a Python double loop. A nearby edge attracts the cursor; a
            # farther edge is ignored so the suggestion cannot jump across
            # unrelated map detail.
            snap_radius = min(
                max(self.ANGLE_CONSTRAINED_SNAP_RADIUS, self.snap_radius),
                self.LOCAL_EDGE_SEARCH_RADIUS_PIXELS,
            )
            x_min = max(0, px - snap_radius)
            x_max = min(w, px + snap_radius + 1)
            y_min = max(0, py - snap_radius)
            y_max = min(h, py + snap_radius + 1)
            edge_pixels = np.argwhere(
                self.cached_edges[y_min:y_max, x_min:x_max] > self.EDGE_PIXEL_THRESHOLD
            )
            if edge_pixels.size == 0:
                return map_point

            candidate_y = edge_pixels[:, 0] + y_min
            candidate_x = edge_pixels[:, 1] + x_min
            delta_x = candidate_x - px
            delta_y = candidate_y - py
            distances = np.hypot(delta_x, delta_y)
            attraction_radius = max(
                1,
                int(round(self.MAX_EDGE_ATTRACTION_PIXELS * self.edge_weight)),
            )
            nearby = distances <= attraction_radius
            if not np.any(nearby):
                return map_point
            candidate_x = candidate_x[nearby]
            candidate_y = candidate_y[nearby]
            distances = distances[nearby]

            # Approximate direction continuity in cache pixels. The old
            # implementation transformed every candidate back to map space;
            # the local pixel approximation is both faster and stable enough
            # for a small attraction radius.
            if len(self.path_points) >= 2:
                previous_px, previous_py = self.map_to_pixel(self.path_points[-2])
                last_px, last_py = self.map_to_pixel(self.path_points[-1])
                last_angle = math.atan2(
                    last_py - previous_py,
                    last_px - previous_px,
                )
                new_angles = np.arctan2(
                    candidate_y - last_py,
                    candidate_x - last_px,
                )
                angle_diff = np.abs(
                    np.arctan2(
                        np.sin(new_angles - last_angle),
                        np.cos(new_angles - last_angle),
                    )
                )
                allowed = angle_diff <= math.radians(self.MAX_TURN_ANGLE_DEGREES)
                if not np.any(allowed):
                    return map_point
                candidate_x = candidate_x[allowed]
                candidate_y = candidate_y[allowed]
                distances = distances[allowed]

            best_index = int(np.argmin(distances))
            edge_point = self.pixel_to_map(
                int(candidate_x[best_index]),
                int(candidate_y[best_index]),
            )
            # Keep the cursor authoritative; the slider controls only the
            # strength of the local nudge and never permits a route jump.
            proximity = max(
                0.0,
                1.0 - float(distances[best_index]) / (attraction_radius + 1.0),
            )
            blend = min(
                self.MAX_EDGE_BLEND,
                max(
                    0.0,
                    self.EDGE_BLEND_FACTOR * self.edge_weight * proximity,
                ),
            )
            result_x = map_point.x() * (1 - blend) + edge_point.x() * blend
            result_y = map_point.y() * (1 - blend) + edge_point.y() * blend
            return QgsPointXY(result_x, result_y)

        except Exception:
            return map_point

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for undo and save."""

        is_checkpoint_undo = (
            event.key() == _qt_value("Key_Z", "Key")
            and event.modifiers() & _qt_value("ControlModifier", "KeyboardModifier")
        ) or event.key() == _qt_value("Key_Backspace", "Key")
        if is_checkpoint_undo:
            if self.is_tracing:
                self.undo_to_checkpoint()
                event.accept()
            else:
                # Once a trace has been saved, leave Ctrl+Z/Backspace to QGIS
                # so the layer's ordinary edit stack remains reachable.
                event.ignore()
            return

        if not self.is_tracing:
            return

        # Esc: Remove last 10 points (quick undo)

        # Esc: Cancel entire line (Reset Tracing)
        if event.key() == _qt_value("Key_Escape", "Key"):
            self.reset_tracing()
            return

        # Delete: Cancel entire line
        if event.key() == _qt_value("Key_Delete", "Key"):
            self.reset_tracing()
            return

        # Enter: Save current line (Capture PREVIEW if exists)
        if event.key() in (
            _qt_value("Key_Return", "Key"),
            _qt_value("Key_Enter", "Key"),
        ):
            if self.is_tracing:
                # If there's a green preview line, DO NOT include it
                # User request: "삐져나온 초록선이 거슬린다" -> Only save clicked points
                # if self.preview_path:
                #    self.path_points.extend(self.preview_path)

                if len(self.path_points) >= 2:
                    # Ask for elevation
                    elevation = self.ask_elevation()
                    if elevation is None:
                        return
                    if self.save_to_layer(closed=False, elevation=elevation):
                        self.reset_tracing()
                else:
                    self.reset_tracing()
            return

    def find_optimal_path(self, target_point):
        """
        A* Path Finding from last point to target point.
        Uses cached_cost map to prefer edges.
        """
        if not self.path_points:
            return [target_point]

        sam_path = self._find_sam_path(target_point)
        if sam_path:
            return sam_path

        if self.cached_cost is None:
            return [target_point]

        try:
            start_point = self.path_points[-1]
            start_px, start_py = self.map_to_pixel(start_point)
            end_px, end_py = self.map_to_pixel(target_point)
            pixel_path, used_partial = self._run_a_star_path(
                self.cached_cost,
                start_px,
                start_py,
                end_px,
                end_py,
                allow_partial=True,
            )

            if used_partial:
                if self.iface:
                    self.iface.messageBar().pushMessage(
                        PLUGIN_NAME,
                        self._tr(
                            "경로 탐색 시간이 초과되어 단순화된 경로를 사용했습니다. (확대해서 시도해보세요)",
                            "Pathfinding timeout - simplified path used (Try zooming in)",
                        ),
                        MESSAGE_WARNING,
                        self.PATH_TIMEOUT_MESSAGE_SECONDS,
                    )
                else:
                    print(self._tr("Pathfinding timeout", "Pathfinding timeout"))

            if pixel_path:
                return self._pixel_path_to_map(pixel_path)

            return [target_point]

        except Exception:
            return [target_point]

    def undo_to_checkpoint(self):
        """Undo back to the last checkpoint, but KEEP the checkpoint to continue from."""
        if len(self.checkpoints) <= 1:
            # Only start checkpoint - can't undo further, just notify
            return

        # Get last checkpoint index (the one we want to KEEP)
        last_cp_idx = self.checkpoints[-1]

        # Check if we're already AT the checkpoint (no new points after it)
        if len(self.path_points) <= last_cp_idx + 1:
            # Already at checkpoint, go back to PREVIOUS checkpoint
            if len(self.checkpoints) > 1:
                self.checkpoints.pop()  # Remove current checkpoint
                if self.checkpoints:
                    last_cp_idx = self.checkpoints[-1]
                else:
                    self.reset_tracing()
                    return

        # Trim path to checkpoint (keep points UP TO AND INCLUDING checkpoint)
        self.path_points = self.path_points[:last_cp_idx + 1]

        # Update last_map_point so user can continue from checkpoint
        if self.path_points:
            self.last_map_point = self.path_points[-1]

        # Rebuild checkpoint markers
        self.checkpoint_markers.reset(POINT_GEOMETRY)
        for cp_idx in self.checkpoints[1:]:  # Skip start point
            if cp_idx < len(self.path_points):
                self.checkpoint_markers.addPoint(self.path_points[cp_idx])

        # Redraw
        self.redraw_confirmed_path()

    def undo_points(self, count):
        """Remove last N points."""
        if len(self.path_points) <= 1:
            return

        # Remove points but keep at least the start
        remove_count = min(count, len(self.path_points) - 1)
        self.path_points = self.path_points[:-remove_count]

        # Update last_map_point
        if self.path_points:
            self.last_map_point = self.path_points[-1]

        # Remove checkpoints that are now beyond the path
        while self.checkpoints and self.checkpoints[-1] >= len(self.path_points):
            self.checkpoints.pop()

        # Rebuild checkpoint markers
        self.checkpoint_markers.reset(POINT_GEOMETRY)
        for cp_idx in self.checkpoints[1:]:
            if cp_idx < len(self.path_points):
                self.checkpoint_markers.addPoint(self.path_points[cp_idx])

        # Redraw
        self.redraw_confirmed_path()

    def gentle_snap(self, map_point):
        """
        VERY gentle edge snapping:
        - Only nudge slightly toward edge if very close
        - Never jump or cause jittery motion
        - Prioritize smooth drawing over edge accuracy
        """
        if self.cached_edges is None or self.cache_transform is None:
            return map_point

        try:
            px, py = self.map_to_pixel(map_point)
            h, w = self.cached_edges.shape

            if px < 0 or py < 0 or px >= w or py >= h:
                return map_point

            # Check only immediate vicinity (5 pixels)
            snap_radius = max(1, min(self.GENTLE_SNAP_RADIUS, self.snap_radius))

            # Check if directly on edge first
            ipx, ipy = int(px), int(py)
            if 0 <= ipx < w and 0 <= ipy < h:
                if self.cached_edges[ipy, ipx] > self.EDGE_PIXEL_THRESHOLD:
                    # Already on edge, no change needed
                    return map_point

            # Look for nearby edge
            best_dist = snap_radius + 1
            best_px, best_py = px, py
            found = False

            for dy in range(-snap_radius, snap_radius + 1):
                for dx in range(-snap_radius, snap_radius + 1):
                    nx, ny = int(px + dx), int(py + dy)
                    if 0 <= nx < w and 0 <= ny < h:
                        if self.cached_edges[ny, nx] > self.EDGE_PIXEL_THRESHOLD:
                            dist = abs(dx) + abs(dy)  # Manhattan distance for stability
                            if dist < best_dist:
                                best_dist = dist
                                best_px, best_py = nx, ny
                                found = True

            if found:
                edge_point = self.pixel_to_map(best_px, best_py)
                # VERY gentle nudge - only 30% toward edge
                blend = self.EDGE_BLEND_FACTOR
                result_x = map_point.x() * (1 - blend) + edge_point.x() * blend
                result_y = map_point.y() * (1 - blend) + edge_point.y() * blend
                return QgsPointXY(result_x, result_y)

            # No edge nearby - just follow mouse exactly
            return map_point

        except Exception:
            return map_point

    def snap_to_existing_endpoint(self, point):
        """
        Find simplest endpoint of existing lines to snap to.
        Returns: (Point, FeatureID, IsStartOfLine)
        """
        if not self.vector_layer or self.vector_layer.featureCount() == 0:
            return None, None, False

        tolerance = self.canvas.mapUnitsPerPixel() * self.ENDPOINT_SNAP_TOLERANCE_PIXELS
        min_dist = tolerance
        best_point = None
        best_fid = None
        best_is_start = False

        for feat in self.vector_layer.getFeatures():
            geom = feat.geometry()
            if not geom or geom.isEmpty():
                continue

            try:
                geom = self._layer_geometry_to_map(geom, self.vector_layer)
            except Exception as exc:
                print(f"Endpoint snap CRS transform failed: {exc}")
                return None, None, False

            # Skip non-line geometries (e.g. Polygons) to prevent crash
            if geom.type() != LINE_GEOMETRY:
                continue

            if geom.isMultipart():
                lines = geom.asMultiPolyline()
            else:
                lines = [geom.asPolyline()]

            # Only support single line merging for simplicity
            line = lines[0]
            if not line:
                continue

            # Start point
            p1 = line[0]
            d1 = np.sqrt((p1.x()-point.x())**2 + (p1.y()-point.y())**2)
            if d1 < min_dist:
                min_dist = d1
                best_point = p1
                best_fid = feat.id()
                best_is_start = True  # Snapped to Start

            # End point
            p2 = line[-1]
            d2 = np.sqrt((p2.x()-point.x())**2 + (p2.y()-point.y())**2)
            if d2 < min_dist:
                min_dist = d2
                best_point = p2
                best_fid = feat.id()
                best_is_start = False  # Snapped to End

        return best_point, best_fid, best_is_start

    def create_spot_height(self, point, elevation):
        """Create a point feature on the Spot Height layer."""
        layer = self.get_or_create_spot_layer()
        if not layer:
            return False

        if layer.readOnly():
            self._push_message(
                self._tr("Spot Height 레이어가 읽기 전용입니다.", "Spot Height layer is read-only."),
                MESSAGE_CRITICAL,
            )
            return False

        try:
            geometry = self._map_geometry_to_layer(QgsGeometry.fromPointXY(point), layer)
        except Exception as exc:
            self._push_message(
                self._tr(
                    f"Spot Height 좌표계 변환에 실패했습니다: {exc}",
                    f"Failed to transform Spot Height coordinates: {exc}",
                ),
                MESSAGE_CRITICAL,
            )
            return False
        if not self._add_geometry_feature(
            layer,
            geometry,
            elevation,
            "ArchaeoTrace add spot height",
        ):
            self._push_message(
                self._tr("Spot Height 저장에 실패했습니다.", "Failed to save spot height."),
                MESSAGE_CRITICAL,
            )
            return False

        layer.triggerRepaint()
        return True

    def _cache_request_identity(self, read_extent, output_size, tile_origin):
        """Capture the raster/extent/CRS identity on QGIS' main thread."""

        def safe_value(owner, name, fallback=""):
            try:
                value = getattr(owner, name)()
                return str(value)
            except (AttributeError, RuntimeError, TypeError):
                return fallback

        provider = self.raster_layer.dataProvider()
        raster_extent = self.raster_layer.extent()
        return (
            safe_value(self.raster_layer, "id", str(id(self.raster_layer))),
            safe_value(provider, "dataSourceUri"),
            safe_value(self.raster_layer.crs(), "authid"),
            safe_value(self.canvas.mapSettings().destinationCrs(), "authid"),
            int(self.raster_layer.width()),
            int(self.raster_layer.height()),
            float(raster_extent.xMinimum()),
            float(raster_extent.yMinimum()),
            float(raster_extent.xMaximum()),
            float(raster_extent.yMaximum()),
            float(read_extent.xMinimum()),
            float(read_extent.yMinimum()),
            float(read_extent.xMaximum()),
            float(read_extent.yMaximum()),
            int(output_size[0]),
            int(output_size[1]),
            int(tile_origin[0]),
            int(tile_origin[1]),
        )

    def _current_ink_cache_identity(self):
        """Recompute the live view/source identity for an async result."""

        if not self._is_active or getattr(self, "_disposed", False):
            return None
        try:
            if not self.raster_layer or not self.raster_layer.isValid():
                return None
            canvas_extent = self._canvas_extent_in_raster_crs()
            if canvas_extent is None:
                return None
            raster_extent = self.raster_layer.extent()
            read_extent, tile_origin = self._ink_evidence_extent_and_origin(
                canvas_extent,
                raster_extent,
            )
            if read_extent.isEmpty():
                return None
            output_size, _enable_evidence, _reason = (
                self._ink_evidence_sampling_plan(
                    read_extent,
                    raster_extent,
                )
            )
            return self._cache_request_identity(
                read_extent,
                output_size,
                tile_origin,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            # The source may disappear between the task manager callback and
            # this main-thread snapshot. Treat that as stale, never as a
            # reason to publish old pixels.
            return None

    def _publish_edge_cache(
        self,
        *,
        rgb_image,
        edges,
        evidence,
        cost_map,
        read_extent,
        output_size,
        tile_origin,
        cache_identity,
        recovery_compatible=False,
        recovery_disabled_reason="",
    ):
        """Publish one current immutable task result on the main thread."""

        out_w, out_h = output_size
        self.cached_rgb_image = rgb_image
        self.cached_edges = np.ascontiguousarray(edges, dtype=np.uint8)
        self.cached_ink_evidence = evidence
        self._recovery_cache_compatible = bool(
            evidence is not None and recovery_compatible
        )
        self._recovery_cache_disabled_reason = str(
            recovery_disabled_reason or ""
        )
        self.cached_cost = cost_map
        self.cache_extent = read_extent
        self.cache_tile_origin = tuple(tile_origin)
        self.cache_identity = tuple(cache_identity)
        self.cache_transform = {
            "x_min": read_extent.xMinimum(),
            "y_max": read_extent.yMaximum(),
            "px_w": read_extent.width() / out_w,
            "px_h": read_extent.height() / out_h,
            "width": out_w,
            "height": out_h,
            "source_tile_origin": tuple(tile_origin),
        }
        self.sam_image_ready = False
        self.sam_warning_emitted = False
        self.cache_dirty = False
        if self.is_tracing and self.path_points:
            self._request_livewire_tree(force=True)

    def _on_ink_evidence_finished(
        self,
        task,
        succeeded,
        fallback_edges,
        evidence,
        cost_map,
        evidence_error,
        error,
    ):
        if self._ink_evidence_task is not task:
            return
        self._ink_evidence_task = None
        current_identity = self._current_ink_cache_identity()
        is_current = (
            succeeded
            and task.generation == self._ink_evidence_generation
            and task.cache_identity == self._pending_cache_identity
            and task.cache_identity == current_identity
            and not getattr(self, "_disposed", False)
        )
        if not is_current:
            if error is not None and not getattr(self, "_disposed", False):
                self._emit_recovery_state(
                    RECOVERY_STATE_INK_FALLBACK,
                    f"Ink evidence failed; exact cursor remains available: {error}",
                )
            return

        selected_edges = fallback_edges
        if evidence is not None:
            selected_edges = np.where(
                np.asarray(evidence.centerline, dtype=bool),
                255,
                0,
            ).astype(np.uint8)
        if selected_edges is None:
            self._clear_edge_cache()
            self._emit_recovery_state(
                RECOVERY_STATE_INK_FALLBACK,
                "Ink evidence produced no champion; exact cursor remains active.",
            )
            return
        if evidence is not None:
            published_image = task.recovery_image
            read_extent = QgsRectangle(*task.cache_extent)
            output_size = task.output_size
            tile_origin = task.tile_origin
            published_identity = task.cache_identity
        else:
            # The v2 request is validated with its expanded source-grid
            # identity above, but a failed/disabled v2 computation must
            # publish the frozen 0.1.5 visible-extent cache byte-for-byte.
            published_image = task.fallback_rgb_image
            read_extent = QgsRectangle(*task.fallback_cache_extent)
            output_size = task.fallback_output_size
            tile_origin = (0, 0)
            published_identity = task.fallback_cache_identity
        self._publish_edge_cache(
            rgb_image=published_image,
            edges=selected_edges,
            evidence=evidence,
            cost_map=cost_map,
            read_extent=read_extent,
            output_size=output_size,
            tile_origin=tile_origin,
            cache_identity=published_identity,
            recovery_compatible=task.recovery_compatible,
            recovery_disabled_reason=task.recovery_disabled_reason,
        )
        if evidence_error is not None:
            self._emit_recovery_state(
                RECOVERY_STATE_INK_FALLBACK,
                f"Continuous Ink evidence failed; Ink v1 was kept: {evidence_error}",
            )
        elif (
            self.smart_recovery_requested
            and not self._recovery_cache_compatible
        ):
            self._emit_recovery_state(
                RECOVERY_STATE_INK_FALLBACK,
                self._recovery_cache_disabled_reason
                or "Smart Recovery is unavailable for this raster; Ink v2 was kept.",
            )
        elif not self.smart_recovery_requested or self.smart_recovery_enabled:
            self._emit_recovery_state(
                RECOVERY_STATE_INK,
                "Ink evidence ready.",
            )

    def _ink_evidence_extent_and_origin(self, extent, raster_ext):
        """Read complete anchored source tiles and their detector halo.

        Ink v2 normalizes each 128px source-grid tile using a 16px contextual
        response halo after a maximum 15px morphology radius. Reading only
        ``visible + one halo`` makes the raw response at the normalization
        boundary depend on crop-edge padding. Expand every intersected tile
        to its complete core first, then add the combined 31px read context
        and clip at the raster boundary. Viewports within the same tile set
        consequently share the exact same image, origin and percentile
        context.
        """

        visible = extent.intersect(raster_ext)
        if visible.isEmpty():
            return visible, (0, 0)
        source_width = max(1, int(self.raster_layer.width()))
        source_height = max(1, int(self.raster_layer.height()))
        source_pixel_width = raster_ext.width() / source_width
        source_pixel_height = raster_ext.height() / source_height
        tile_size = self.INK_EVIDENCE_TILE_SOURCE_PIXELS
        halo = self.INK_EVIDENCE_HALO_SOURCE_PIXELS

        visible_x0 = max(
            0,
            int(math.floor(
                (visible.xMinimum() - raster_ext.xMinimum())
                / source_pixel_width
            )),
        )
        visible_x1 = min(
            source_width,
            int(math.ceil(
                (visible.xMaximum() - raster_ext.xMinimum())
                / source_pixel_width
            )),
        )
        visible_y0 = max(
            0,
            int(math.floor(
                (raster_ext.yMaximum() - visible.yMaximum())
                / source_pixel_height
            )),
        )
        visible_y1 = min(
            source_height,
            int(math.ceil(
                (raster_ext.yMaximum() - visible.yMinimum())
                / source_pixel_height
            )),
        )

        tile_x0 = (visible_x0 // tile_size) * tile_size
        tile_x1 = min(
            source_width,
            ((visible_x1 + tile_size - 1) // tile_size) * tile_size,
        )
        tile_y0 = (visible_y0 // tile_size) * tile_size
        tile_y1 = min(
            source_height,
            ((visible_y1 + tile_size - 1) // tile_size) * tile_size,
        )
        source_x0 = max(0, tile_x0 - halo)
        source_x1 = min(source_width, tile_x1 + halo)
        source_y0 = max(0, tile_y0 - halo)
        source_y1 = min(source_height, tile_y1 + halo)
        read_extent = QgsRectangle(
            raster_ext.xMinimum() + source_x0 * source_pixel_width,
            raster_ext.yMaximum() - source_y1 * source_pixel_height,
            raster_ext.xMinimum() + source_x1 * source_pixel_width,
            raster_ext.yMaximum() - source_y0 * source_pixel_height,
        )
        return read_extent.intersect(raster_ext), (source_x0, source_y0)

    def _ink_evidence_sampling_plan(self, read_extent, raster_extent):
        """Use v2 only when one cache pixel is exactly one source pixel.

        Ink v2's 9/15/31px filters and 128px normalization tiles are defined
        in the source raster grid. Passing a downsampled 1000px viewport with
        a source-pixel tile origin would silently mix two coordinate systems.
        Keep the normal bounded v1 cache for wide views, but opt into v2 only
        when its halo-expanded source window itself fits the memory boundary.
        """

        source_width = max(1, int(self.raster_layer.width()))
        source_height = max(1, int(self.raster_layer.height()))
        source_pixel_width = raster_extent.width() / source_width
        source_pixel_height = raster_extent.height() / source_height
        exact_width = int(round(read_extent.width() / source_pixel_width))
        exact_height = int(round(read_extent.height() / source_pixel_height))
        exact_width = max(1, exact_width)
        exact_height = max(1, exact_height)

        source_grid_fits = (
            exact_width <= self.CACHE_MAX_DIMENSION
            and exact_height <= self.CACHE_MAX_DIMENSION
            and exact_width * exact_height
            <= self.CACHE_MAX_DIMENSION * self.CACHE_MAX_DIMENSION
        )
        if source_grid_fits:
            # This equality is the detector contract: array indices, filter
            # scales, tile origin, tile size and halo all share source pixels.
            return (exact_width, exact_height), True, ""

        output_size = compute_resampled_dimensions(
            raster_extent.width(),
            raster_extent.height(),
            source_width,
            source_height,
            read_extent.width(),
            read_extent.height(),
            self.CACHE_MAX_DIMENSION,
            min_dimension=1,
        )
        reason = (
            "Zoom in to enable continuous Ink evidence at source resolution "
            f"({exact_width}x{exact_height} source pixels exceeds "
            f"{self.CACHE_MAX_DIMENSION}x{self.CACHE_MAX_DIMENSION}); "
            "Ink v1 remains active."
        )
        return output_size, False, reason

    def update_edge_cache(self):
        """Cache edge detection for current view."""
        self._edge_cache_timer.stop()
        if not self._needs_edge_cache():
            self._clear_edge_cache()
            return
        if not self.cache_dirty and self.cached_edges is not None:
            return
        if self._ink_evidence_task is not None:
            return
        try:
            if self.edge_detector is None:
                self._clear_edge_cache()
                return

            extent = self._canvas_extent_in_raster_crs()
            if extent is None:
                self._clear_edge_cache()
                return

            provider = self.raster_layer.dataProvider()
            raster_ext = self.raster_layer.extent()
            visible_ext = extent.intersect(raster_ext)
            if visible_ext.isEmpty():
                self._clear_edge_cache()
                return
            use_ink_evidence = self.edge_method == EdgeDetector.METHOD_INK
            if use_ink_evidence:
                evidence_read_ext, tile_origin = (
                    self._ink_evidence_extent_and_origin(
                        extent,
                        raster_ext,
                    )
                )
                (
                    (evidence_width, evidence_height),
                    enable_ink_evidence,
                    evidence_disabled_reason,
                ) = self._ink_evidence_sampling_plan(
                    evidence_read_ext,
                    raster_ext,
                )
                if (
                    evidence_width < self.CACHE_MIN_DIMENSION
                    or evidence_height < self.CACHE_MIN_DIMENSION
                ):
                    enable_ink_evidence = False
                    evidence_disabled_reason = (
                        "Continuous Ink evidence needs at least "
                        f"{self.CACHE_MIN_DIMENSION} source pixels per axis; "
                        "Ink v1 remains active."
                    )

                # This is the frozen 0.1.5 cache boundary.  It intentionally
                # remains the unexpanded visible extent with the historical
                # resampling and per-block uint8 conversion.  The wider v2
                # source context must never leak into a fallback result.
                fallback_width, fallback_height = compute_resampled_dimensions(
                    raster_ext.width(),
                    raster_ext.height(),
                    self.raster_layer.width(),
                    self.raster_layer.height(),
                    visible_ext.width(),
                    visible_ext.height(),
                    self.CACHE_MAX_DIMENSION,
                    min_dimension=1,
                )
                if (
                    fallback_width < self.CACHE_MIN_DIMENSION
                    or fallback_height < self.CACHE_MIN_DIMENSION
                ):
                    self._clear_edge_cache()
                    return
                fallback_bands = read_raster_bands(
                    provider,
                    visible_ext,
                    fallback_width,
                    fallback_height,
                    max_bands=self.CACHE_MAX_BANDS_FOR_RGB,
                )
                if not fallback_bands:
                    self._clear_edge_cache()
                    return
                fallback_rgb_image = self._build_cached_rgb_image(
                    fallback_bands
                )
                fallback_image = (
                    fallback_rgb_image
                    if len(fallback_bands) >= 3
                    else fallback_bands[0]
                )
                fallback_cache_identity = self._cache_request_identity(
                    visible_ext,
                    (fallback_width, fallback_height),
                    (0, 0),
                )

                evidence_image = fallback_image
                recovery_image = fallback_rgb_image
                recovery_compatible = False
                recovery_disabled_reason = (
                    "Smart Recovery requires native Byte raster samples; "
                    "Ink remains active."
                )
                if enable_ink_evidence:
                    (
                        _evidence_display_bands,
                        native_bands,
                        native_integer_stable,
                    ) = read_raster_bands_with_native(
                        provider,
                        evidence_read_ext,
                        evidence_width,
                        evidence_height,
                        max_bands=self.CACHE_MAX_BANDS_FOR_RGB,
                    )
                    if not native_integer_stable:
                        enable_ink_evidence = False
                        evidence_disabled_reason = (
                            "Continuous Ink evidence needs native integer "
                            "raster DNs for pan-stable normalization; this "
                            "source uses an unsupported or floating sample "
                            "type. Ink v1 remains active."
                        )
                    else:
                        if len(native_bands) >= 3:
                            evidence_image = np.ascontiguousarray(
                                np.stack(native_bands[:3], axis=-1)
                            )
                        else:
                            evidence_image = np.ascontiguousarray(
                                native_bands[0]
                            )
                        recovery_image = self._build_stable_integer_rgb_image(
                            native_bands
                        )
                        recovery_compatible = all(
                            np.asarray(band).dtype == np.dtype(np.uint8)
                            for band in native_bands
                        )
                        if not recovery_compatible:
                            recovery_disabled_reason = (
                                "Smart Recovery currently supports only native "
                                "Byte raster samples; Ink v2 remains active."
                            )

                request_identity = self._cache_request_identity(
                    evidence_read_ext,
                    (evidence_width, evidence_height),
                    tile_origin,
                )
                self._cancel_ink_evidence_task()
                self._cancel_livewire_task()
                self._livewire_generation += 1
                self._livewire_tree = None
                self._cancel_recovery_task(clear_request=True)
                self._recovery_generation += 1
                self._recovery_encoding = None
                self._recovery_encoding_cache_generation = None
                self._cache_generation += 1
                generation = self._ink_evidence_generation
                self._pending_cache_identity = request_identity
                task = _InkEvidenceTask(
                    detector=self.edge_detector,
                    fallback_image=fallback_image,
                    fallback_rgb_image=fallback_rgb_image,
                    fallback_cache_identity=fallback_cache_identity,
                    fallback_cache_extent=(
                        visible_ext.xMinimum(),
                        visible_ext.yMinimum(),
                        visible_ext.xMaximum(),
                        visible_ext.yMaximum(),
                    ),
                    fallback_output_size=(fallback_width, fallback_height),
                    evidence_image=evidence_image,
                    recovery_image=recovery_image,
                    recovery_compatible=recovery_compatible,
                    recovery_disabled_reason=recovery_disabled_reason,
                    tile_origin=tile_origin,
                    enable_evidence=enable_ink_evidence,
                    evidence_disabled_reason=evidence_disabled_reason,
                    build_cost_map=self.auto_path and self.cv2 is not None,
                    edge_weight=self.edge_weight,
                    generation=generation,
                    cache_identity=request_identity,
                    cache_extent=(
                        evidence_read_ext.xMinimum(),
                        evidence_read_ext.yMinimum(),
                        evidence_read_ext.xMaximum(),
                        evidence_read_ext.yMaximum(),
                    ),
                    output_size=(evidence_width, evidence_height),
                    callback=self._on_ink_evidence_finished,
                )
                self._ink_evidence_task = task
                QgsApplication.taskManager().addTask(task)
                return

            read_ext = visible_ext
            tile_origin = (0, 0)
            out_w, out_h = compute_resampled_dimensions(
                raster_ext.width(),
                raster_ext.height(),
                self.raster_layer.width(),
                self.raster_layer.height(),
                read_ext.width(),
                read_ext.height(),
                self.CACHE_MAX_DIMENSION,
                min_dimension=1,
            )
            if out_w < self.CACHE_MIN_DIMENSION or out_h < self.CACHE_MIN_DIMENSION:
                self._clear_edge_cache()
                return
            bands = read_raster_bands(
                provider,
                read_ext,
                out_w,
                out_h,
                max_bands=self.CACHE_MAX_BANDS_FOR_RGB,
            )
            if not bands:
                self._clear_edge_cache()
                return
            rgb_image = self._build_cached_rgb_image(bands)
            image = rgb_image if len(bands) >= 3 else bands[0]
            cache_identity = self._cache_request_identity(
                read_ext,
                (out_w, out_h),
                tile_origin,
            )
            edges = self.edge_detector.detect_edges(image)
            cost_map = None
            if self.auto_path and self.cv2 is not None:
                cost_map = self.edge_detector.get_edge_cost_map(
                    edges,
                    self.edge_weight,
                )
            self._publish_edge_cache(
                rgb_image=rgb_image,
                edges=edges,
                evidence=None,
                cost_map=cost_map,
                read_extent=read_ext,
                output_size=(out_w, out_h),
                tile_origin=tile_origin,
                cache_identity=cache_identity,
            )

        except Exception as e:
            print(f"Edge cache error: {e}")
            self._clear_edge_cache()

    def map_to_pixel(self, map_point):
        """Convert map coordinates to pixel coordinates."""
        px, py = self.map_to_pixel_float(map_point)
        return int(px), int(py)

    def map_to_pixel_float(self, map_point):
        """Convert map coordinates without losing sub-pixel cursor motion."""
        if self.cache_transform is None:
            raise ValueError("Edge cache is not initialized.")

        raster_point = self._map_point_to_raster(map_point)
        t = self.cache_transform
        px = (raster_point.x() - t['x_min']) / t['px_w']
        py = (t['y_max'] - raster_point.y()) / t['px_h']
        return float(px), float(py)

    def pixel_to_map(self, px, py):
        """Convert pixel coordinates to map coordinates."""
        if self.cache_transform is None:
            raise ValueError("Edge cache is not initialized.")

        t = self.cache_transform
        raster_point = QgsPointXY(
            t['x_min'] + px * t['px_w'],
            t['y_max'] - py * t['px_h'],
        )
        return self._raster_point_to_map(raster_point)

    def is_near_start(self, point):
        """Check if point is near start point for polygon close."""
        if not self.start_point:
            return False

        # If we only have the start point (Spot Height candidate), use larger tolerance
        is_spot_candidate = (len(self.path_points) == 1)

        dx = point.x() - self.start_point.x()
        dy = point.y() - self.start_point.y()
        dist = np.sqrt(dx*dx + dy*dy)

        base_tol = self.CLOSE_TOLERANCE_BASE_PIXELS
        if is_spot_candidate:
            base_tol = self.CLOSE_TOLERANCE_SPOT_PIXELS

        close_threshold = self.canvas.mapUnitsPerPixel() * base_tol
        return dist < close_threshold

    def redraw_confirmed_path(self):
        """Redraw the confirmed path."""
        self.confirm_band.reset(LINE_GEOMETRY)
        for pt in self.path_points:
            self.confirm_band.addPoint(pt)

    def save_to_layer(self, closed=False, elevation=None):
        """Save path to vector layer with Bézier smoothing."""
        if len(self.path_points) < 2 or not self.vector_layer:
            return False

        if self.vector_layer.readOnly():
            self._push_message(
                self._tr("출력 레이어가 읽기 전용입니다.", "Output layer is read-only."),
                MESSAGE_CRITICAL,
            )
            return False
        if self.unsupported_output_reason(self.vector_layer):
            self._push_message(
                self._tr(
                    "Z/M 라인은 차원값 손실 위험 때문에 저장하지 않았습니다. 2D 라인 레이어를 사용하세요.",
                    "The Z/M line was not saved because dimensional values could be lost. Use a 2D line layer.",
                ),
                MESSAGE_CRITICAL,
            )
            return False

        # Disable extra smoothing to match Green Preview exactly
        # The points are already smoothed by 5-point Moving Average in find_optimal_path
        smoothed = list(self.path_points)

        # ALWAYS use LineString. For closed loops, just make start==end.
        # Allow 2 points + close = 3 points (Triangle/Flat Loop)
        if closed and len(smoothed) >= 2:
            # Add first point to end to close the loop
            # ONLY if not already closed
            if smoothed[-1] != smoothed[0]:
                smoothed.append(smoothed[0])

        # Prepare geometry
        map_geometry = QgsGeometry.fromPolylineXY(smoothed)

        # MERGE LOGIC
        if self.resume_feature_id is not None and not closed:
            # We are extending an existing feature
            existing_feat = self.vector_layer.getFeature(self.resume_feature_id)
            if existing_feat.isValid() and existing_feat.geometry():
                try:
                    existing_geom = self._layer_geometry_to_map(
                        existing_feat.geometry(),
                        self.vector_layer,
                    )
                except Exception as exc:
                    self._push_message(
                        self._tr(
                            f"기존 선의 좌표계 변환에 실패했습니다: {exc}",
                            f"Failed to transform the existing line: {exc}",
                        ),
                        MESSAGE_CRITICAL,
                    )
                    return False
                existing_lines = None

                # Prevent crash on Multipart: Cannot simple-merge without knowing which part
                if existing_geom.isMultipart():
                    self.resume_feature_id = None
                    self._push_message(
                        self._tr(
                            "멀티파트 라인은 새 피처로 저장합니다.",
                            "Multipart lines are saved as a new feature.",
                        ),
                        MESSAGE_INFO,
                    )
                else:
                    existing_lines = existing_geom.asPolyline()

                if existing_lines and self.resume_at_start:
                    # We snapped to START. We are drawing AWAY from start.
                    # So new line ends at old start.
                    # Merged = (New Reversed) + Existing
                    # BUT: self.path_points[0] IS the snap point (Old Start).
                    # So self.path_points starts at Old Start and goes away.
                    # So we should Reverse New and Append Existing.

                    # current path: [Start(Snap), P1, P2 ...]
                    # reversed: [..., P2, P1, Start(Snap)]
                    # existing: [Start(Snap), E1, E2 ...]
                    # Combined: [..., P2, P1, Start(Snap), E1, E2 ...]

                    new_part = smoothed[::-1]  # Reverse
                    merged_points = new_part[:-1] + existing_lines  # Skip duplicate join
                elif existing_lines:
                    # Snapped to END. Drawing away.
                    # Existing: [..., End(Snap)]
                    # New: [End(Snap), P1, P2 ...]
                    # Combined: [..., End(Snap), P1, P2 ...]
                    merged_points = existing_lines + smoothed[1:]

                if existing_lines:
                    merged_map_geometry = QgsGeometry.fromPolylineXY(merged_points)
                    try:
                        layer_geometry = self._map_geometry_to_layer(
                            merged_map_geometry,
                            self.vector_layer,
                        )
                    except Exception as exc:
                        self._push_message(
                            self._tr(
                                f"병합한 선의 좌표계 변환에 실패했습니다: {exc}",
                                f"Failed to transform the merged line: {exc}",
                            ),
                            MESSAGE_CRITICAL,
                        )
                        return False

                    def update_in_edit_buffer():
                        attribute_changes = {}
                        if elevation is not None:
                            elev_idx = self._ensure_field_in_edit_buffer(
                                self.vector_layer,
                                FIELD_ELEVATION,
                                _field_type("Double"),
                            )
                            if elev_idx < 0:
                                return False
                            attribute_changes[elev_idx] = float(elevation)
                        return self._update_feature_in_edit_buffer(
                            self.vector_layer,
                            self.resume_feature_id,
                            layer_geometry,
                            attribute_changes,
                        )

                    if not self._run_edit_command(
                        self.vector_layer,
                        "ArchaeoTrace extend feature",
                        update_in_edit_buffer,
                    ):
                        try:
                            self.vector_layer.updateFields()
                        # Preserve the primary edit failure.
                        except Exception:  # nosec B110
                            pass
                        self._push_message(
                            self._tr("기존 선 갱신에 실패했습니다.", "Failed to update existing line."),
                            MESSAGE_CRITICAL,
                        )
                        return False

                    self.vector_layer.updateExtents()
                    self.vector_layer.triggerRepaint()
                    self.resume_feature_id = None
                    self.resume_at_start = False
                    return True

        try:
            layer_geometry = self._map_geometry_to_layer(map_geometry, self.vector_layer)
        except Exception as exc:
            self._push_message(
                self._tr(
                    f"등고선 좌표계 변환에 실패했습니다: {exc}",
                    f"Failed to transform contour coordinates: {exc}",
                ),
                MESSAGE_CRITICAL,
            )
            return False
        return self.save_geometry(layer_geometry, elevation)

    def save_geometry(self, geometry, elevation=None):
        """Helper to save a generic geometry to the layer."""
        if not self.vector_layer:
            return False
        if self.unsupported_output_reason(self.vector_layer):
            self._push_message(
                self._tr(
                    "2D 라인 출력 레이어만 지원합니다.",
                    "Only 2D line output layers are supported.",
                ),
                MESSAGE_CRITICAL,
            )
            return False

        if not self._add_geometry_feature(
            self.vector_layer,
            geometry,
            elevation,
        ):
            self._push_message(
                self._tr("피처 저장에 실패했습니다.", "Failed to save feature."),
                MESSAGE_CRITICAL,
            )
            return False

        self.vector_layer.triggerRepaint()
        self.resume_feature_id = None
        self.resume_at_start = False
        return True

    def ask_elevation(self):
        """Show dialog to input elevation value."""
        from qgis.PyQt.QtWidgets import QInputDialog

        value, ok = QInputDialog.getDouble(
            None,
            self._tr("등고선 해발값", "Contour Elevation"),
            self._tr("해발고도 (m):", "Elevation (m):"),
            self.ELEVATION_DEFAULT,
            self.ELEVATION_MIN,
            self.ELEVATION_MAX,
            self.ELEVATION_DECIMALS,
        )

        if ok:
            return value
        return None

    def smooth_bezier(self, points, closed=False):
        """
        Smooth points using Bézier-like curve fitting (Chaikin).
        Handles closed polygons correctly to prevent flattened ends.
        """
        if len(points) < 3:
            return points

        if not closed:
            smoothed = chaikin_smooth_path(
                ((point.x(), point.y()) for point in points),
                iterations=self.CHAIKIN_ITERATIONS,
                q_weight=self.CHAIKIN_Q_WEIGHT,
                r_weight=self.CHAIKIN_R_WEIGHT,
            )
            return [QgsPointXY(x, y) for x, y in smoothed]

        # Convert to numpy for easier math
        pts = np.array([[p.x(), p.y()] for p in points])

        for _ in range(self.CHAIKIN_ITERATIONS):
            if len(pts) < 3:
                break

            new_pts = []

            # If NOT closed, keep first point
            if not closed:
                new_pts.append(pts[0])

            # Loop segments
            count = len(pts) if closed else len(pts) - 1

            for i in range(count):
                p0 = pts[i]
                p1 = pts[(i + 1) % len(pts)]

                q = p0 * self.CHAIKIN_Q_WEIGHT + p1 * self.CHAIKIN_R_WEIGHT
                r = p0 * self.CHAIKIN_R_WEIGHT + p1 * self.CHAIKIN_Q_WEIGHT
                new_pts.extend([q, r])

            # If NOT closed, keep last point
            if not closed:
                new_pts.append(pts[-1])

            pts = np.array(new_pts)

        return [QgsPointXY(p[0], p[1]) for p in pts]

    def reset_tracing(self):
        """Reset all tracing state."""
        self._cancel_recovery_task(clear_request=True)
        self._recovery_generation += 1
        self._recovery_preview_identity = None
        self.is_tracing = False
        self.path_points = []
        self.preview_path = []
        self.preview_is_global = False
        self.preview_target = None
        self._proposal_timer.stop()
        self._cancel_proposal_task()
        self._proposal_generation += 1
        self._proposal_request_point = None
        self._cancel_livewire_task()
        self._livewire_generation += 1
        self._livewire_tree = None
        self._livewire_anchor_pixel = None
        self._livewire_request_point = None
        self._livewire_failed_anchor = None
        self.checkpoints = []
        self.start_point = None
        self.last_map_point = None
        self.last_input_point = None
        self.last_hover_pos = None  # Reset stabilizer
        self.last_sample_pos = None
        self.last_preview_pos = None
        self.resume_feature_id = None
        self.resume_at_start = False
        self.preview_band.reset(LINE_GEOMETRY)
        self.confirm_band.reset(LINE_GEOMETRY)
        self.start_marker.reset(POINT_GEOMETRY)
        self.close_indicator.reset(POINT_GEOMETRY)
        self.checkpoint_markers.reset(POINT_GEOMETRY)
        self.snap_marker.reset(POINT_GEOMETRY)
        if self.smart_recovery_requested:
            state = (
                RECOVERY_STATE_INK
                if self.smart_recovery_enabled
                else RECOVERY_STATE_INK_FALLBACK
            )
            self._emit_recovery_state(state)

    def dispose(self):
        """Release canvas-owned graphics and asynchronous work permanently."""

        if getattr(self, "_disposed", False):
            return
        self._disposed = True
        self._edge_cache_timer.stop()
        self._proposal_timer.stop()
        self._cancel_ink_evidence_task()
        self._set_extent_cache_listener(False)
        self._set_coordinate_crs_listeners(False)
        self._set_source_lifecycle_listeners(False)
        try:
            self.reset_tracing()
        except RuntimeError:
            # A source may already have been deleted while deactivating.
            pass

        # QgsRubberBand is a canvas scene item, not a QObject child of this
        # map tool. Deleting the canvas-parented tool alone leaves every band
        # in the scene until QGIS exits, so detach and drop them explicitly.
        scene = self.canvas.scene()
        for attribute_name in (
            "preview_band",
            "confirm_band",
            "start_marker",
            "close_indicator",
            "checkpoint_markers",
            "snap_marker",
        ):
            item = getattr(self, attribute_name, None)
            if item is None:
                continue
            try:
                scene.removeItem(item)
            except RuntimeError:
                pass
            setattr(self, attribute_name, None)

    def activate(self):
        """Called when tool is activated."""
        self._is_active = True
        self._refresh_crs_transforms()
        self._set_coordinate_crs_listeners(True)
        self._set_source_lifecycle_listeners(True)
        if self._needs_edge_cache():
            self.update_edge_cache()
        else:
            self._clear_edge_cache()
        self._set_extent_cache_listener(True)

        super().activate()

    def deactivate(self):
        """Called when tool is deactivated."""
        # Flip the activity gate before cancelling tasks so a finished()
        # callback queued in the same event-loop turn cannot republish.
        self._is_active = False
        self._edge_cache_timer.stop()
        self._proposal_timer.stop()
        self._set_extent_cache_listener(False)
        self._set_coordinate_crs_listeners(False)
        self._set_source_lifecycle_listeners(False)

        self.reset_tracing()
        self._clear_edge_cache()
        super().deactivate()
