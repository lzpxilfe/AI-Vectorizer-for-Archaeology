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
    QgsVectorLayer, QgsField, QgsApplication, QgsTask, Qgis
)
from qgis.PyQt.QtCore import Qt, QVariant, QTimer, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import QAction

from ..core.dependencies import get_cv2, require_cv2
from ..core.edge_detector import EdgeDetector
from ..core.raster_utils import compute_resampled_dimensions, read_raster_bands
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
    blend_path_with_cursor,
    build_livewire_tree,
    is_livewire_available,
)
from ..core.sam_trace_kernel import (
    DEFAULT_CONFIG as DEFAULT_SAM_TRACE_CONFIG,
    SamTraceConfig,
    build_cost_map as build_sam_cost_map,
    nearest_active_pixel as find_nearest_active_pixel,
    postprocess_mask as postprocess_sam_mask,
    trace_mask as trace_sam_mask,
)
from ..core.trace_kernel import (
    TraceConfig,
    chaikin_smooth_path,
    find_path,
    smooth_pixel_path,
)
from ..config import (
    DEFAULT_EDGE_METHOD,
    DEFAULT_OUTPUT_LAYER_NAME,
    DEFAULT_SPOT_LAYER_NAME,
    FIELD_ELEVATION,
    FIELD_ID,
    PLUGIN_NAME,
)


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
        super().__init__("ArchaeoTrace live path preview", QgsTask.CanCancel)
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


class _LiveWireTreeTask(QgsTask):
    """Build one anchor-rooted Live-Wire tree away from the UI thread."""

    def __init__(
        self,
        *,
        image,
        edges,
        anchor_pixel,
        incoming_direction,
        strength,
        generation,
        config,
        callback,
    ):
        super().__init__("ArchaeoTrace Live-Wire tree", QgsTask.CanCancel)
        self.image = image
        self.edges = edges
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
            )
            return not self.isCanceled()
        except LiveWireCancelled:
            return False
        except Exception as exc:
            self.error = exc
            return False

    def finished(self, result):
        self.callback(self, bool(result), self.tree, self.error)


class SmartTraceTool(QgsMapToolEmitPoint):
    deactivated = pyqtSignal()
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

    UNDO_MESSAGE_SECONDS = 2
    UNDO_ACTION_OBJECT_NAME = 'mActionUndo'
    PREVIEW_BAND_COLOR = (0, 180, 0, 180)
    PREVIEW_BAND_WIDTH = 8
    PREVIEW_BAND_LINE_STYLE = Qt.DashLine
    # Keep every uncommitted suggestion in the same visual language. The
    # distinction is interaction state, not another competing line color.
    PROPOSAL_BAND_COLOR = (0, 180, 0, 180)
    PROPOSAL_BAND_WIDTH = 8
    PROPOSAL_BAND_LINE_STYLE = Qt.DashLine
    CONFIRM_BAND_COLOR = (255, 50, 50, 255)
    CONFIRM_BAND_WIDTH = 3
    START_MARKER_COLOR = (255, 255, 0, 255)
    START_MARKER_WIDTH = 12
    START_MARKER_ICON = QgsRubberBand.ICON_CIRCLE
    CLOSE_INDICATOR_COLOR = (0, 255, 255, 200)
    CLOSE_INDICATOR_WIDTH = 16
    CLOSE_INDICATOR_ICON = QgsRubberBand.ICON_CIRCLE
    CHECKPOINT_MARKER_COLOR = (50, 150, 255, 255)
    CHECKPOINT_MARKER_WIDTH = 10
    CHECKPOINT_MARKER_ICON = QgsRubberBand.ICON_BOX
    SNAP_MARKER_COLOR = (255, 0, 255, 200)
    SNAP_MARKER_WIDTH = 15
    SNAP_MARKER_ICON = QgsRubberBand.ICON_X
    A_STAR_NEIGHBORS = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]

    def _tr(self, ko_text, en_text):
        return en_text if getattr(self, "language", "ko") == "en" else ko_text

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
        self.preview_band.reset(QgsWkbTypes.LineGeometry)
        self.last_sample_pos = None
        self.last_preview_pos = None
        self._edge_cache_timer.start()

    def _set_undo_enabled(self, enabled):
        if not self.iface:
            return

        try:
            undo_action = self.iface.actionUndo()
            if undo_action is not None:
                undo_action.setEnabled(enabled)

            main_window = self.iface.mainWindow()
            fallback_action = None
            if main_window is not None:
                fallback_action = main_window.findChild(
                    QAction,
                    self.UNDO_ACTION_OBJECT_NAME,
                )
            if fallback_action is not None and fallback_action is not undo_action:
                fallback_action.setEnabled(enabled)
        except Exception as exc:
            action_name = "enable" if enabled else "disable"
            print(f"Failed to {action_name} undo action: {exc}")

    def __init__(self, canvas, raster_layer, vector_layer, model_type=0,
                 sam_engine=None, edge_weight=0.5, freehand=False, edge_method=DEFAULT_EDGE_METHOD,
                 iface=None, language="ko", auto_path=False):
        self.canvas = canvas
        super().__init__(self.canvas)
        self.iface = iface
        self.language = language

        self.raster_layer = raster_layer
        self.vector_layer = vector_layer
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
        self.preview_band = QgsRubberBand(self.canvas, QgsWkbTypes.LineGeometry)
        self._preview_style_is_global = None
        self._configure_band(
            self.preview_band,
            self.PREVIEW_BAND_COLOR,
            self.PREVIEW_BAND_WIDTH,
            line_style=self.PREVIEW_BAND_LINE_STYLE,
        )

        self.confirm_band = QgsRubberBand(self.canvas, QgsWkbTypes.LineGeometry)
        self._configure_band(
            self.confirm_band,
            self.CONFIRM_BAND_COLOR,
            self.CONFIRM_BAND_WIDTH,
        )

        self.start_marker = QgsRubberBand(self.canvas, QgsWkbTypes.PointGeometry)
        self._configure_band(
            self.start_marker,
            self.START_MARKER_COLOR,
            self.START_MARKER_WIDTH,
            icon=self.START_MARKER_ICON,
        )

        self.close_indicator = QgsRubberBand(self.canvas, QgsWkbTypes.PointGeometry)
        self._configure_band(
            self.close_indicator,
            self.CLOSE_INDICATOR_COLOR,
            self.CLOSE_INDICATOR_WIDTH,
            icon=self.CLOSE_INDICATOR_ICON,
        )

        # Checkpoint markers (blue diamonds)
        self.checkpoint_markers = QgsRubberBand(self.canvas, QgsWkbTypes.PointGeometry)
        self._configure_band(
            self.checkpoint_markers,
            self.CHECKPOINT_MARKER_COLOR,
            self.CHECKPOINT_MARKER_WIDTH,
            icon=self.CHECKPOINT_MARKER_ICON,
        )

        # Checkpoints: list of point indices where user clicked
        self.checkpoints = []

        # Snap marker (for resuming drawing)
        self.snap_marker = QgsRubberBand(self.canvas, QgsWkbTypes.PointGeometry)
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
        self.cache_transform = None  # Pixel <-> Map transform
        self.cached_rgb_image = None
        self.sam_image_ready = False
        self.sam_warning_emitted = False
        self.cache_dirty = True

        self._edge_cache_timer = QTimer(self.canvas)
        self._edge_cache_timer.setSingleShot(True)
        self._edge_cache_timer.setInterval(self.CACHE_DEBOUNCE_MS)
        self._edge_cache_timer.timeout.connect(self.update_edge_cache)
        self._extent_cache_listener_connected = False

        # Auto Path/SAM proposals are debounced so the expensive route is
        # calculated after the cursor pauses, not for every mouse event.
        self._proposal_timer = QTimer(self.canvas)
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

        # CRS transforms
        self.to_raster_transform = QgsCoordinateTransform(
            self.canvas.mapSettings().destinationCrs(),
            self.raster_layer.crs(),
            QgsProject.instance()
        )
        self.to_map_transform = QgsCoordinateTransform(
            self.raster_layer.crs(),
            self.canvas.mapSettings().destinationCrs(),
            QgsProject.instance()
        )

        # Resume/Merge State
        self.resume_feature_id = None
        self.resume_at_start = False  # True if appending to Start of existing line

        # Stability (Anti-Pulse)
        self.last_hover_pos = None
        self.last_sample_pos = None

        # Auto-create output layer if needed
        if not self.vector_layer:
            self.vector_layer = self.create_output_layer()

    def create_output_layer(self):
        crs = self.canvas.mapSettings().destinationCrs().authid()
        layer = QgsVectorLayer(f"LineString?crs={crs}", DEFAULT_OUTPUT_LAYER_NAME, "memory")
        pr = layer.dataProvider()
        pr.addAttributes([QgsField(FIELD_ID, QVariant.Int)])
        layer.updateFields()
        QgsProject.instance().addMapLayer(layer)
        return layer

    def get_or_create_spot_layer(self):
        """Get or create the Spot Heights (Point) layer."""
        if self.spot_height_layer and not self.spot_height_layer.isValid():
            self.spot_height_layer = None

        target_crs = (
            self.vector_layer.crs()
            if self.vector_layer and self.vector_layer.crs().isValid()
            else self.canvas.mapSettings().destinationCrs()
        )

        if self.spot_height_layer is None:
            # Check if exists in project
            for layer in QgsProject.instance().mapLayers().values():
                if (
                    layer.name() == DEFAULT_SPOT_LAYER_NAME
                    and layer.geometryType() == QgsWkbTypes.PointGeometry
                    and layer.crs() == target_crs
                ):
                    self.spot_height_layer = layer
                    break

        if self.spot_height_layer is None:
            crs = target_crs.authid()
            self.spot_height_layer = QgsVectorLayer(f"Point?crs={crs}", DEFAULT_SPOT_LAYER_NAME, "memory")
            pr = self.spot_height_layer.dataProvider()
            pr.addAttributes([QgsField(FIELD_ELEVATION, QVariant.Double)])
            self.spot_height_layer.updateFields()
            QgsProject.instance().addMapLayer(self.spot_height_layer)

        return self.spot_height_layer

    def _push_message(self, text, level=Qgis.Warning, duration=4):
        if self.iface:
            self.iface.messageBar().pushMessage(PLUGIN_NAME, text, level, duration)
        else:
            print(text)

    def _clear_edge_cache(self):
        self._cancel_livewire_task()
        self._livewire_generation += 1
        self._livewire_tree = None
        self._livewire_anchor_pixel = None
        self._livewire_request_point = None
        self.cached_edges = None
        self.cached_cost = None
        self.cache_extent = None
        self.cache_transform = None
        self.cached_rgb_image = None
        self.sam_image_ready = False
        self.sam_warning_emitted = False
        self.cache_dirty = True

    @staticmethod
    def _provider_result_ok(result):
        if isinstance(result, tuple):
            return bool(result[0])
        return bool(result)

    @staticmethod
    def _ensure_edit_session(layer):
        if layer.isEditable():
            return True
        try:
            return bool(layer.startEditing())
        except Exception:
            return False

    def _ensure_field(self, layer, field_name, field_type):
        fields = layer.fields()
        field_idx = fields.indexOf(field_name)
        if field_idx >= 0:
            return field_idx

        field = QgsField(field_name, field_type)
        if layer.isEditable():
            ok = layer.addAttribute(field)
        else:
            ok = self._provider_result_ok(layer.dataProvider().addAttributes([field]))
            if not ok and self._ensure_edit_session(layer):
                ok = layer.addAttribute(field)

        if not ok:
            self._push_message(
                self._tr(
                    f"필드 '{field_name}' 추가에 실패했습니다.",
                    f"Failed to add field '{field_name}'.",
                ),
                Qgis.Critical,
            )
            return -1

        layer.updateFields()
        return layer.fields().indexOf(field_name)

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
        feature = QgsFeature()
        feature.setFields(layer.fields())
        feature.setGeometry(geometry)

        attrs = [None] * len(layer.fields())
        id_idx = layer.fields().indexOf(FIELD_ID)
        if id_idx >= 0:
            attrs[id_idx] = self._next_feature_id_value(layer)

        elev_idx = layer.fields().indexOf(FIELD_ELEVATION)
        if elev_idx >= 0 and elevation is not None:
            attrs[elev_idx] = float(elevation)

        feature.setAttributes(attrs)
        return feature

    def _add_feature(self, layer, feature):
        if layer.isEditable():
            return bool(layer.addFeature(feature))

        ok = self._provider_result_ok(layer.dataProvider().addFeatures([feature]))
        if not ok and self._ensure_edit_session(layer):
            ok = bool(layer.addFeature(feature))
        if ok:
            layer.updateExtents()
        return ok

    def _update_geometry(self, layer, feature_id, geometry):
        if layer.isEditable():
            return bool(layer.changeGeometry(feature_id, geometry))
        ok = self._provider_result_ok(
            layer.dataProvider().changeGeometryValues({feature_id: geometry})
        )
        if not ok and self._ensure_edit_session(layer):
            ok = bool(layer.changeGeometry(feature_id, geometry))
        return ok

    def _canvas_extent_in_raster_crs(self):
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
                Qgis.Warning,
            )
            return None

    def _map_point_to_raster(self, map_point):
        if self.canvas.mapSettings().destinationCrs() == self.raster_layer.crs():
            return QgsPointXY(map_point.x(), map_point.y())

        transformed = self.to_raster_transform.transform(map_point)
        return QgsPointXY(transformed.x(), transformed.y())

    def _raster_point_to_map(self, point):
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
                    Qgis.Warning,
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
            self.freehand
            or self.use_sam
            or self.edge_weight <= 0.0
            or self.cached_edges is None
            or self.cached_rgb_image is None
            or not self.path_points
        ):
            return False

        if not is_livewire_available():
            if not self._livewire_warning_emitted:
                self._push_message(
                    self._tr(
                        "SciPy가 없어 방향 인식 Live-Wire 대신 가까운 선 스냅을 사용합니다.",
                        "SciPy is unavailable; using nearby-edge snapping instead of Live-Wire.",
                    ),
                    Qgis.Warning,
                    5,
                )
                self._livewire_warning_emitted = True
            return False

        anchor_pixel = self._current_livewire_anchor_pixel()
        if anchor_pixel is None:
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
            request_point = self._livewire_request_point
            if request_point is not None and not self.use_sam:
                self._present_livewire_cursor_preview(
                    request_point,
                    global_mode=self.auto_path,
                    request_tree=False,
                )
            return

        if error is not None:
            print(f"Live-Wire tree build failed: {error}")

        # A drag or click may have advanced the accepted anchor while the old
        # tree was building. Coalesce that state into one fresh build.
        if self.is_tracing and current_anchor is not None:
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
        self.preview_band.reset(QgsWkbTypes.LineGeometry)
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
                    Qgis.Warning,
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
            Qgis.Info,
            3,
        )
        return False

    def canvasPressEvent(self, event):
        if event.button() == Qt.RightButton:
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

        if event.button() != Qt.LeftButton:
            return

        point = self.toMapCoordinates(event.pos())

        if not self.is_tracing:
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
            self.start_marker.reset(QgsWkbTypes.PointGeometry)
            self.start_marker.addPoint(place_point)
            self.snap_marker.reset(QgsWkbTypes.PointGeometry)  # Hide snap marker

            # Reset checkpoint markers
            self.checkpoint_markers.reset(QgsWkbTypes.PointGeometry)

            # Update edge cache
            if self._needs_edge_cache():
                self.update_edge_cache()
                self._request_livewire_tree(force=False)

            self.confirm_band.reset(QgsWkbTypes.LineGeometry)
            self.confirm_band.addPoint(place_point)
            self.preview_band.reset(QgsWkbTypes.LineGeometry)
        else:
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
            self.snap_marker.reset(QgsWkbTypes.PointGeometry)
            if snapped:
                self.snap_marker.addPoint(snapped)
            return

        # 2. TRACING ACTIVE

        # Check close indicator
        if self.is_near_start(current_point):
            self.close_indicator.reset(QgsWkbTypes.PointGeometry)
            self.close_indicator.addPoint(self.start_point)
        else:
            self.close_indicator.reset(QgsWkbTypes.PointGeometry)

        if self.last_map_point is None:
            self.last_map_point = current_point
            self.last_input_point = current_point
            return

        # MODE CHECK: Dragging vs Hovering
        is_manual_mode = (event.modifiers() & (Qt.ShiftModifier | Qt.ControlModifier))
        interaction_mode = resolve_interaction_mode(
            freehand=self.freehand,
            auto_path=self.auto_path,
            manual_override=bool(is_manual_mode),
        )

        is_dragging = bool(event.buttons() & Qt.LeftButton)

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
            self.preview_band.reset(QgsWkbTypes.LineGeometry)
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
                    self.snap_marker.reset(QgsWkbTypes.PointGeometry)
                    self.snap_marker.addPoint(snap_pt)
                    if self.iface:
                        self.iface.mapCanvas().setCursor(Qt.PointingHandCursor)
                else:
                    self.snap_marker.reset(QgsWkbTypes.PointGeometry)
                    if self.iface:
                        self.iface.mapCanvas().setCursor(Qt.CrossCursor)
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

        # GLOBAL UNDO BLOCKER:
        # Prevent QGIS from consuming Ctrl+Z and deleting committed features
        # CRITICAL: This must be handled BEFORE the is_tracing check to protect idle state
        if (event.key() == Qt.Key_Z and event.modifiers() & Qt.ControlModifier) or event.key() == Qt.Key_Backspace:
            if self.is_tracing:
                self.undo_to_checkpoint()
            else:
                # Inform user that global undo is blocked here for safety
                if self.iface:
                    self.iface.messageBar().pushMessage(
                        PLUGIN_NAME,
                        self._tr(
                            "완료된 선 보호를 위해 Undo가 비활성화되어 있습니다. 피처 삭제는 Delete 키를 사용하세요.",
                            "Undo is disabled to protect finished lines. Use Delete key to remove features.",
                        ),
                        Qgis.Info,
                        self.UNDO_MESSAGE_SECONDS,
                    )

            # CRITICAL: Always accept event to stop propagation
            event.accept()
            return

        if not self.is_tracing:
            return

        # Esc: Remove last 10 points (quick undo)

        # Esc: Cancel entire line (Reset Tracing)
        if event.key() == Qt.Key_Escape:
            self.reset_tracing()
            return

        # Delete: Cancel entire line
        if event.key() == Qt.Key_Delete:
            self.reset_tracing()
            return

        # Enter: Save current line (Capture PREVIEW if exists)
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
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
                        Qgis.Warning,
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
        self.checkpoint_markers.reset(QgsWkbTypes.PointGeometry)
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
        self.checkpoint_markers.reset(QgsWkbTypes.PointGeometry)
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
            if geom.type() != QgsWkbTypes.LineGeometry:
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
                Qgis.Critical,
            )
            return False

        elev_idx = self._ensure_field(layer, FIELD_ELEVATION, QVariant.Double)
        if elev_idx < 0:
            return False

        feat = QgsFeature()
        feat.setFields(layer.fields())
        try:
            geometry = self._map_geometry_to_layer(QgsGeometry.fromPointXY(point), layer)
        except Exception as exc:
            self._push_message(
                self._tr(
                    f"Spot Height 좌표계 변환에 실패했습니다: {exc}",
                    f"Failed to transform Spot Height coordinates: {exc}",
                ),
                Qgis.Critical,
            )
            return False
        feat.setGeometry(geometry)
        attrs = [None] * len(layer.fields())
        attrs[elev_idx] = float(elevation)
        feat.setAttributes(attrs)

        if not self._add_feature(layer, feat):
            self._push_message(
                self._tr("Spot Height 저장에 실패했습니다.", "Failed to save spot height."),
                Qgis.Critical,
            )
            return False

        layer.triggerRepaint()
        return True

    def update_edge_cache(self):
        """Cache edge detection for current view."""
        self._edge_cache_timer.stop()
        if not self._needs_edge_cache():
            self._clear_edge_cache()
            return
        if not self.cache_dirty and self.cached_edges is not None:
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
            read_ext = extent.intersect(raster_ext)

            if read_ext.isEmpty():
                self._clear_edge_cache()
                return

            # Determine output size using the source raster resolution on each axis.
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

            # Read bands
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

            self.cached_rgb_image = self._build_cached_rgb_image(bands)
            self.sam_image_ready = False
            self.sam_warning_emitted = False

            # Convert to grayscale
            if len(bands) >= 3:
                image = self.cached_rgb_image
            else:
                image = bands[0]

            # Detect edges
            self.cached_edges = self.edge_detector.detect_edges(image)
            self.cache_extent = read_ext

            # Store transform info
            self.cache_transform = {
                'x_min': read_ext.xMinimum(),
                'y_max': read_ext.yMaximum(),
                'px_w': read_ext.width() / out_w,
                'px_h': read_ext.height() / out_h,
                'width': out_w,
                'height': out_h
            }

            # Mouse-led tracing only needs the binary edge mask.  Distance
            # transforms are reserved for explicit Auto Path mode, where A*
            # actually consumes the cost map.
            if self.auto_path and self.cv2 is not None:
                self.cached_cost = self.edge_detector.get_edge_cost_map(
                    self.cached_edges,
                    self.edge_weight,
                )
            else:
                self.cached_cost = None
            self.cache_dirty = False
            if self.is_tracing and self.path_points:
                self._request_livewire_tree(force=True)

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
        self.confirm_band.reset(QgsWkbTypes.LineGeometry)
        for pt in self.path_points:
            self.confirm_band.addPoint(pt)

    def save_to_layer(self, closed=False, elevation=None):
        """Save path to vector layer with Bézier smoothing."""
        if len(self.path_points) < 2 or not self.vector_layer:
            return False

        if self.vector_layer.readOnly():
            self._push_message(
                self._tr("출력 레이어가 읽기 전용입니다.", "Output layer is read-only."),
                Qgis.Critical,
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
                        Qgis.Critical,
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
                        Qgis.Info,
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
                            Qgis.Critical,
                        )
                        return False
                    if not self._update_geometry(
                        self.vector_layer,
                        self.resume_feature_id,
                        layer_geometry,
                    ):
                        self._push_message(
                            self._tr("기존 선 갱신에 실패했습니다.", "Failed to update existing line."),
                            Qgis.Critical,
                        )
                        return False

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
                Qgis.Critical,
            )
            return False
        return self.save_geometry(layer_geometry, elevation)

    def save_geometry(self, geometry, elevation=None):
        """Helper to save a generic geometry to the layer."""
        if not self.vector_layer:
            return False

        if elevation is not None:
            elev_idx = self._ensure_field(self.vector_layer, FIELD_ELEVATION, QVariant.Double)
            if elev_idx < 0:
                return False

        feature = self._build_feature(self.vector_layer, geometry, elevation)
        if not self._add_feature(self.vector_layer, feature):
            self._push_message(
                self._tr("피처 저장에 실패했습니다.", "Failed to save feature."),
                Qgis.Critical,
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
        self.checkpoints = []
        self.start_point = None
        self.last_map_point = None
        self.last_input_point = None
        self.last_hover_pos = None  # Reset stabilizer
        self.last_sample_pos = None
        self.last_preview_pos = None
        self.resume_feature_id = None
        self.resume_at_start = False
        self.preview_band.reset(QgsWkbTypes.LineGeometry)
        self.confirm_band.reset(QgsWkbTypes.LineGeometry)
        self.start_marker.reset(QgsWkbTypes.PointGeometry)
        self.close_indicator.reset(QgsWkbTypes.PointGeometry)
        self.checkpoint_markers.reset(QgsWkbTypes.PointGeometry)
        self.snap_marker.reset(QgsWkbTypes.PointGeometry)

    def activate(self):
        """Called when tool is activated."""
        if self._needs_edge_cache():
            self.update_edge_cache()
        else:
            self._clear_edge_cache()
        self._set_extent_cache_listener(True)

        # NUCLEAR UNDO BLOCK: Disable QGIS Undo Action
        self._set_undo_enabled(False)

        super().activate()

    def deactivate(self):
        """Called when tool is deactivated."""
        self._edge_cache_timer.stop()
        self._proposal_timer.stop()
        self._set_extent_cache_listener(False)

        # RESTORE UNDO ACTION
        self._set_undo_enabled(True)

        self.reset_tracing()
        super().deactivate()
        self.deactivated.emit()
