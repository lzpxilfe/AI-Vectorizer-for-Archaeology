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
    QgsVectorLayer, QgsField, Qgis
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
    resolve_interaction_mode,
    uses_global_path_search,
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
    DEFAULT_OUTPUT_LAYER_NAME,
    DEFAULT_SPOT_LAYER_NAME,
    FIELD_ELEVATION,
    FIELD_ID,
    PLUGIN_NAME,
)


class SmartTraceTool(QgsMapToolEmitPoint):
    deactivated = pyqtSignal()
    SNAP_RADIUS_BASE = 15
    SNAP_RADIUS_EDGE_WEIGHT_FACTOR = 0.7
    SAMPLE_INTERVAL_PIXELS = 3
    AUTO_PATH_SAMPLE_INTERVAL_PIXELS = 12

    EDGE_BLEND_FACTOR = 0.3
    EDGE_PIXEL_THRESHOLD = 128

    ANGLE_CONSTRAINED_SNAP_RADIUS = 6
    GENTLE_SNAP_RADIUS = 5
    MAX_TURN_ANGLE_DEGREES = 60

    ENDPOINT_SNAP_TOLERANCE_PIXELS = 10
    CLOSE_TOLERANCE_BASE_PIXELS = 20
    CLOSE_TOLERANCE_SPOT_PIXELS = 30

    CACHE_MAX_DIMENSION = 1000
    CACHE_MIN_DIMENSION = 10
    CACHE_MAX_BANDS_FOR_RGB = 3
    CACHE_DEBOUNCE_MS = 150

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
        self.preview_band.reset(QgsWkbTypes.LineGeometry)
        self.last_sample_pos = None
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
                 sam_engine=None, edge_weight=0.5, freehand=False, edge_method='canny',
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
        self.edge_weight = float(edge_weight)
        self.auto_path = bool(auto_path) and not self.freehand

        # Snap radius (pixels) - higher = more magnetic
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
        self.is_tracing = False
        self.start_point = None
        self.last_map_point = None
        self.last_input_point = None

        # RubberBands for visualization
        self.preview_band = QgsRubberBand(self.canvas, QgsWkbTypes.LineGeometry)
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
        if not self.freehand or self.use_sam:
            self.cv2 = require_cv2("OpenCV tracing")

        # Edge detector
        self.edge_detector = None
        if not self.freehand or self.use_sam:
            self.edge_detector = EdgeDetector(method=self.edge_method)

        # Edge cache
        self.cached_edges = None
        self.cached_cost = None
        self.cache_extent = None
        self.cache_transform = None  # Pixel <-> Map transform
        self.cached_rgb_image = None
        self.sam_image_ready = False
        self.sam_warning_emitted = False

        self._edge_cache_timer = QTimer(self.canvas)
        self._edge_cache_timer.setSingleShot(True)
        self._edge_cache_timer.setInterval(self.CACHE_DEBOUNCE_MS)
        self._edge_cache_timer.timeout.connect(self.update_edge_cache)
        self._extent_cache_listener_connected = False

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
        self.cached_edges = None
        self.cached_cost = None
        self.cache_extent = None
        self.cache_transform = None
        self.cached_rgb_image = None
        self.sam_image_ready = False
        self.sam_warning_emitted = False

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

    def _run_a_star_path(self, cost_map, start_px, start_py, end_px, end_py, allow_partial=True):
        config = TraceConfig(
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
        result = find_path(
            cost_map,
            (start_px, start_py),
            (end_px, end_py),
            allow_partial=allow_partial,
            config=config,
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
            if not self.freehand:
                self.update_edge_cache()

            self.confirm_band.reset(QgsWkbTypes.LineGeometry)
            self.confirm_band.addPoint(place_point)
            self.preview_band.reset(QgsWkbTypes.LineGeometry)
        else:
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

                # Normal Polygon Close.  Only the deliberate Auto Path mode
                # may invent a route for the closing segment; the default
                # mouse-led mode closes directly to the user's start point.
                if self.auto_path:
                    closing_path = self.find_optimal_path(self.start_point)
                    if len(closing_path) > 2:
                        closing_path = self.smooth_bezier(closing_path, closed=False)
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

            # ADD CHECKPOINT: Save current position as checkpoint
            if self.preview_path:
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

    def canvasMoveEvent(self, event):
        current_point = self.toMapCoordinates(event.pos())
        # Keep the raw cursor as the source of truth.  The previous 70/30 EMA
        # introduced visible cursor lag and leaked pre-trace hover state into
        # the first segment.  Local snapping below is enough to tame detector
        # jitter without delaying the user's input.
        self.last_hover_pos = current_point
        ai_target_point = current_point

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

        # Throttle by screen distance, not map distance.  This stays stable
        # when the user zooms during a trace and avoids over-sampling at a
        # different scale.
        if self.last_sample_pos is None:
            self.last_sample_pos = event.pos()
        sample_pixels = (
            self.AUTO_PATH_SAMPLE_INTERVAL_PIXELS
            if uses_global_path_search(interaction_mode)
            else self.SAMPLE_INTERVAL_PIXELS
        )
        screen_dx = event.pos().x() - self.last_sample_pos.x()
        screen_dy = event.pos().y() - self.last_sample_pos.y()
        if screen_dx * screen_dx + screen_dy * screen_dy < sample_pixels * sample_pixels:
            return
        self.last_sample_pos = event.pos()

        if event.buttons() & Qt.LeftButton:
            # DRAGGING: Manual Draw (Mouse Following + Gentle Snap)
            self.preview_path = []

            # If Manual Mode (Shift/Ctrl): No snapping, just exact mouse pos
            if interaction_mode == MODE_AUTO_PATH and self.cached_edges is not None:
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
                smoothed_preview = [current_point]
            elif uses_global_path_search(interaction_mode):
                # AI Auto-Path Preview (Bunting Style)
                # Calculate A* path from last point to mouse
                ai_path = self.find_optimal_path(ai_target_point)

                # Apply Smoothing to Preview
                if len(ai_path) > 2:
                    smoothed_preview = self.smooth_bezier(ai_path, closed=False)
                else:
                    smoothed_preview = ai_path
            else:
                # Mouse-led assist: one nearby, direction-compatible snap
                # point.  This never replaces the user's segment with a
                # global route and never invokes SAM per mouse event.
                smoothed_preview = [
                    self.angle_constrained_snap(current_point)
                    if self.cached_edges is not None
                    else current_point
                ]

            self.preview_path = smoothed_preview
            # Advance the throttle anchor after each accepted preview.  The
            # old hover branch kept measuring from the last committed point,
            # which made it effectively run on every mouse event.
            self.last_input_point = current_point

            # Draw preview (Green line)
            self.preview_band.reset(QgsWkbTypes.LineGeometry)
            if self.path_points:
                self.preview_band.addPoint(self.path_points[-1])
            for pt in smoothed_preview:
                self.preview_band.addPoint(pt)

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

            # 1. Check if we have history to determine direction
            has_history = len(self.path_points) >= 2
            last_angle = 0
            if has_history:
                p1 = self.path_points[-2]
                p2 = self.path_points[-1]
                last_angle = math.atan2(p2.y() - p1.y(), p2.x() - p1.x())

            # 2. Search for nearby edge pixels
            snap_radius = max(self.ANGLE_CONSTRAINED_SNAP_RADIUS, self.snap_radius)
            best_dist = snap_radius + 1
            best_px, best_py = px, py
            found = False

            for dy in range(-snap_radius, snap_radius + 1):
                for dx in range(-snap_radius, snap_radius + 1):
                    nx, ny = int(px + dx), int(py + dy)
                    if 0 <= nx < w and 0 <= ny < h:
                        if self.cached_edges[ny, nx] > self.EDGE_PIXEL_THRESHOLD:
                            # 3. Angle Filter: Check if this point causes sharp turn
                            if has_history:
                                edge_pt = self.pixel_to_map(nx, ny)
                                last_pt = self.path_points[-1]
                                new_angle = math.atan2(edge_pt.y() - last_pt.y(), edge_pt.x() - last_pt.x())
                                angle_diff = abs(new_angle - last_angle)
                                while angle_diff > math.pi:
                                    angle_diff -= 2*math.pi
                                while angle_diff < -math.pi:
                                    angle_diff += 2*math.pi

                                # If turn is sharper than 60 degrees, ignore this edge (it's noise/hairline)
                                if abs(angle_diff) > math.radians(self.MAX_TURN_ANGLE_DEGREES):
                                    continue

                            dist = abs(dx) + abs(dy)
                            if dist < best_dist:
                                best_dist = dist
                                best_px, best_py = nx, ny
                                found = True

            if found:
                edge_point = self.pixel_to_map(best_px, best_py)
                # Gentle blend: 30% edge, 70% mouse
                blend = self.EDGE_BLEND_FACTOR
                result_x = map_point.x() * (1 - blend) + edge_point.x() * blend
                result_y = map_point.y() * (1 - blend) + edge_point.y() * blend
                return QgsPointXY(result_x, result_y)

            return map_point

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
            if self.auto_path:
                self.cached_cost = self.edge_detector.get_edge_cost_map(
                    self.cached_edges,
                    self.edge_weight,
                )
            else:
                self.cached_cost = None

        except Exception as e:
            print(f"Edge cache error: {e}")
            self._clear_edge_cache()

    def map_to_pixel(self, map_point):
        """Convert map coordinates to pixel coordinates."""
        if self.cache_transform is None:
            raise ValueError("Edge cache is not initialized.")

        raster_point = self._map_point_to_raster(map_point)
        t = self.cache_transform
        px = (raster_point.x() - t['x_min']) / t['px_w']
        py = (t['y_max'] - raster_point.y()) / t['px_h']
        return int(px), int(py)

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
        self.checkpoints = []
        self.start_point = None
        self.last_map_point = None
        self.last_input_point = None
        self.last_hover_pos = None  # Reset stabilizer
        self.last_sample_pos = None
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
        self.update_edge_cache()
        self._set_extent_cache_listener(True)

        # NUCLEAR UNDO BLOCK: Disable QGIS Undo Action
        self._set_undo_enabled(False)

        super().activate()

    def deactivate(self):
        """Called when tool is deactivated."""
        self._edge_cache_timer.stop()
        self._set_extent_cache_listener(False)

        # RESTORE UNDO ACTION
        self._set_undo_enabled(True)

        self.reset_tracing()
        super().deactivate()
        self.deactivated.emit()
