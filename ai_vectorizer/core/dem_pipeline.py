"""Validated QGIS Processing pipeline for contours-to-DEM generation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import uuid

from qgis.PyQt.QtCore import QObject, QTimer, pyqtSignal
from qgis.PyQt.QtCore import QVariant
from qgis.core import (
    Qgis,
    QgsApplication,
    QgsProcessingAlgRunnerTask,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
    QgsUnitTypes,
    QgsWkbTypes,
)

from .dem_spec import (
    DEFAULT_MAX_GRID_CELLS,
    DemSpecificationError,
    GridEstimate,
    default_hillshade_path,
    estimate_grid,
    extent_parameter,
    interpolation_data,
    interpolation_entry,
    is_tiff_file,
    normalize_tiff_path,
    paths_refer_to_same_file,
    publish_output_files,
)


TIN_ALGORITHM_ID = "qgis:tininterpolation"
TRANSLATE_ALGORITHM_ID = "gdal:translate"
STATISTICS_ALGORITHM_ID = "native:rasterlayerstatistics"
HILLSHADE_ALGORITHM_ID = "gdal:hillshade"

# QGIS interpolation serialization values. These numeric values are stable in
# QGIS 3.22-3.44 even though the Python enum names became scoped in 3.44.
VALUE_SOURCE_ATTRIBUTE = 0
SOURCE_TYPE_POINTS = 0
SOURCE_TYPE_STRUCTURE_LINES = 1


class DemInputError(ValueError):
    """Raised when terrain inputs cannot safely produce a DEM."""


@dataclass(frozen=True)
class ElevationStats:
    """Validation summary for one elevation vector layer."""

    feature_count: int
    vertex_count: int
    invalid_geometry_count: int
    invalid_elevation_count: int
    minimum: float
    maximum: float
    unique_values: tuple[float, ...]


@dataclass(frozen=True)
class DemBuildRequest:
    """Fully validated inputs for the asynchronous terrain pipeline."""

    interpolation_data: str
    extent: str
    crs_authid: str
    pixel_size: float
    grid: GridEstimate
    dem_path: str
    hillshade_path: str
    contour_stats: ElevationStats
    spot_stats: ElevationStats | None
    dependent_layers: tuple[object, ...]


def _geometry_type_value(name: str, modern_name: str):
    legacy = getattr(QgsWkbTypes, name, None)
    if legacy is not None:
        return legacy
    return getattr(Qgis.GeometryType, modern_name)


def _meter_unit():
    legacy = getattr(QgsUnitTypes, "DistanceMeters", None)
    if legacy is not None:
        return legacy
    return Qgis.DistanceUnit.Meters


def _is_numeric_field(field) -> bool:
    if hasattr(field, "isNumeric"):
        return bool(field.isNumeric())
    numeric_types = {
        QVariant.Int,
        QVariant.UInt,
        QVariant.LongLong,
        QVariant.ULongLong,
        QVariant.Double,
    }
    return field.type() in numeric_types


def _layer_reference(layer) -> str:
    if layer.providerType() == "memory":
        return layer.id()
    return layer.source() or layer.id()


def inspect_elevation_layer(layer, field_name: str) -> ElevationStats:
    """Inspect geometries and numeric elevation values on the main thread."""

    field_index = layer.fields().indexOf(field_name)
    if field_index < 0:
        raise DemInputError(f"Elevation field '{field_name}' does not exist on {layer.name()}.")
    field = layer.fields().at(field_index)
    if not _is_numeric_field(field):
        raise DemInputError(f"Elevation field '{field_name}' on {layer.name()} must be numeric.")

    feature_count = 0
    vertex_count = 0
    invalid_geometry_count = 0
    invalid_elevation_count = 0
    elevations = []

    for feature in layer.getFeatures():
        feature_count += 1
        geometry = feature.geometry()
        if geometry is None or geometry.isEmpty():
            invalid_geometry_count += 1
            continue
        try:
            vertices = sum(1 for _point in geometry.vertices())
        except Exception:
            vertices = 0
        if vertices == 0:
            invalid_geometry_count += 1
            continue
        vertex_count += vertices

        try:
            elevation = float(feature[field_index])
        except (TypeError, ValueError):
            invalid_elevation_count += 1
            continue
        if not math.isfinite(elevation):
            invalid_elevation_count += 1
            continue
        elevations.append(elevation)

    if elevations:
        minimum = min(elevations)
        maximum = max(elevations)
        unique_values = tuple(sorted(set(elevations)))
    else:
        minimum = math.nan
        maximum = math.nan
        unique_values = ()

    return ElevationStats(
        feature_count=feature_count,
        vertex_count=vertex_count,
        invalid_geometry_count=invalid_geometry_count,
        invalid_elevation_count=invalid_elevation_count,
        minimum=minimum,
        maximum=maximum,
        unique_values=unique_values,
    )


def _validate_layer(layer, expected_geometry, label: str) -> None:
    if layer is None or not layer.isValid():
        raise DemInputError(f"Select a valid {label} layer.")
    if layer.geometryType() != expected_geometry:
        raise DemInputError(f"{label.capitalize()} layer has the wrong geometry type.")
    if layer.featureCount() == 0:
        raise DemInputError(f"{label.capitalize()} layer contains no features.")
    if layer.isModified():
        raise DemInputError(f"Save or roll back edits on {layer.name()} before building a DEM.")


def _validate_stats(stats: ElevationStats, label: str, minimum_vertices: int) -> None:
    if stats.invalid_geometry_count:
        raise DemInputError(
            f"{label.capitalize()} layer contains {stats.invalid_geometry_count} empty geometries."
        )
    if stats.invalid_elevation_count:
        raise DemInputError(
            f"{label.capitalize()} layer contains {stats.invalid_elevation_count} missing or invalid elevations."
        )
    if stats.vertex_count < minimum_vertices:
        raise DemInputError(
            f"{label.capitalize()} layer needs at least {minimum_vertices} valid "
            f"{'vertex' if minimum_vertices == 1 else 'vertices'}."
        )


def _validate_output_parent(path: str) -> None:
    parent = Path(path).parent
    if not parent.exists() or not parent.is_dir():
        raise DemInputError(f"Output folder does not exist: {parent}")
    if not os.access(str(parent), os.W_OK):
        raise DemInputError(f"Output folder is not writable: {parent}")


def _has_non_collinear_vertices(layers) -> bool:
    """Return whether any three input vertices can define a TIN triangle."""

    anchor = None
    baseline = None
    for layer in layers:
        for feature in layer.getFeatures():
            geometry = feature.geometry()
            if geometry is None or geometry.isEmpty():
                continue
            for point in geometry.vertices():
                coordinates = (float(point.x()), float(point.y()))
                if anchor is None:
                    anchor = coordinates
                    continue
                dx = coordinates[0] - anchor[0]
                dy = coordinates[1] - anchor[1]
                if baseline is None:
                    if dx != 0 or dy != 0:
                        baseline = (dx, dy)
                    continue
                cross_product = baseline[0] * dy - baseline[1] * dx
                scale = max(
                    math.hypot(*baseline),
                    math.hypot(dx, dy),
                    1.0,
                ) ** 2
                if abs(cross_product) > scale * 1e-12:
                    return True
    return False


def build_dem_request(
    contour_layer,
    contour_field: str,
    pixel_size: float,
    dem_path: str,
    hillshade_path: str | None = None,
    spot_layer=None,
    spot_field: str | None = None,
    max_cells: int = DEFAULT_MAX_GRID_CELLS,
) -> DemBuildRequest:
    """Validate QGIS layers and construct immutable Processing parameters."""

    line_type = _geometry_type_value("LineGeometry", "Line")
    point_type = _geometry_type_value("PointGeometry", "Point")
    _validate_layer(contour_layer, line_type, "contour")

    crs = contour_layer.crs()
    if not crs.isValid() or not crs.authid():
        raise DemInputError("Contour layer needs a valid, identified CRS.")
    if crs.isGeographic() or crs.mapUnits() != _meter_unit():
        raise DemInputError("DEM generation currently requires a projected CRS with metre units.")

    contour_stats = inspect_elevation_layer(contour_layer, contour_field)
    _validate_stats(contour_stats, "contour", minimum_vertices=3)
    contour_field_index = contour_layer.fields().indexOf(contour_field)
    entries = [
        interpolation_entry(
            _layer_reference(contour_layer),
            VALUE_SOURCE_ATTRIBUTE,
            contour_field_index,
            SOURCE_TYPE_STRUCTURE_LINES,
        )
    ]
    dependent_layers = [contour_layer]
    combined_extent = QgsRectangle(contour_layer.extent())

    spot_stats = None
    if spot_layer is not None:
        _validate_layer(spot_layer, point_type, "spot height")
        if spot_layer.crs() != crs:
            raise DemInputError("Contour and spot-height layers must use the same CRS.")
        if not spot_field:
            raise DemInputError("Select a numeric spot-height elevation field.")

        spot_stats = inspect_elevation_layer(spot_layer, spot_field)
        _validate_stats(spot_stats, "spot height", minimum_vertices=1)
        spot_field_index = spot_layer.fields().indexOf(spot_field)
        entries.append(
            interpolation_entry(
                _layer_reference(spot_layer),
                VALUE_SOURCE_ATTRIBUTE,
                spot_field_index,
                SOURCE_TYPE_POINTS,
            )
        )
        dependent_layers.append(spot_layer)
        combined_extent.combineExtentWith(spot_layer.extent())

    if not _has_non_collinear_vertices(dependent_layers):
        raise DemInputError(
            "Terrain inputs are collinear; at least three non-collinear vertices "
            "are required to form a TIN."
        )

    unique_elevations = set(contour_stats.unique_values)
    if spot_stats is not None:
        unique_elevations.update(spot_stats.unique_values)
    if len(unique_elevations) < 2:
        raise DemInputError("At least two distinct elevation values are required to build terrain.")

    if combined_extent.isEmpty() or combined_extent.width() <= 0 or combined_extent.height() <= 0:
        raise DemInputError("Terrain input extent is empty.")
    grid = estimate_grid(combined_extent.width(), combined_extent.height(), pixel_size, max_cells=max_cells)

    dem_path = normalize_tiff_path(dem_path)
    hillshade_path = normalize_tiff_path(hillshade_path or default_hillshade_path(dem_path))
    if paths_refer_to_same_file(dem_path, hillshade_path):
        raise DemInputError("DEM and hillshade outputs must use different files.")
    _validate_output_parent(dem_path)
    _validate_output_parent(hillshade_path)

    return DemBuildRequest(
        interpolation_data=interpolation_data(entries),
        extent=extent_parameter(
            combined_extent.xMinimum(),
            combined_extent.xMaximum(),
            combined_extent.yMinimum(),
            combined_extent.yMaximum(),
            crs.authid(),
        ),
        crs_authid=crs.authid(),
        pixel_size=float(pixel_size),
        grid=grid,
        dem_path=dem_path,
        hillshade_path=hillshade_path,
        contour_stats=contour_stats,
        spot_stats=spot_stats,
        dependent_layers=tuple(dependent_layers),
    )


class _CollectingFeedback(QgsProcessingFeedback):
    def __init__(self):
        super().__init__()
        self.errors = []

    def reportError(self, error, fatalError=False):
        self.errors.append(str(error))
        super().reportError(error, fatalError)


class DemPipelineRunner(QObject):
    """Run TIN, GeoTIFF conversion, and hillshade without blocking QGIS."""

    stageChanged = pyqtSignal(str)
    progressChanged = pyqtSignal(float)
    succeeded = pyqtSignal(str, str)
    failed = pyqtSignal(str)
    canceled = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._active_task = None
        self._request = None
        self._resources = []
        self._cancel_requested = False
        self._continuation_pending = False
        self._raw_dem_path = None
        self._dem_work_path = None
        self._hillshade_work_path = None

    @property
    def is_running(self) -> bool:
        return self._active_task is not None or self._continuation_pending

    def _algorithm(self, algorithm_id: str):
        algorithm = QgsApplication.processingRegistry().algorithmById(algorithm_id)
        if algorithm is None:
            raise DemInputError(f"Required QGIS Processing algorithm is unavailable: {algorithm_id}")
        return algorithm

    @staticmethod
    def _supported_parameters(algorithm, candidates: dict) -> dict:
        names = {definition.name() for definition in algorithm.parameterDefinitions()}
        return {name: value for name, value in candidates.items() if name in names}

    def _new_task(self, algorithm, parameters: dict):
        context = QgsProcessingContext()
        context.setProject(QgsProject.instance())
        feedback = _CollectingFeedback()
        task = QgsProcessingAlgRunnerTask(algorithm, parameters, context, feedback)
        task.setDependentLayers(list(self._request.dependent_layers))
        self._resources.append((task, context, feedback))
        task.destroyed.connect(
            lambda _object=None, completed_feedback=feedback: self._release_resources(
                completed_feedback
            )
        )
        return task, feedback

    def _release_resources(self, completed_feedback) -> None:
        """Release context only after its owning C++ task is destroyed."""

        self._resources = [
            resources
            for resources in self._resources
            if resources[2] is not completed_feedback
        ]

    def _continue_after_task_destroyed(
        self,
        task,
        callback,
        successful: bool,
        results: dict,
        feedback,
    ) -> None:
        """Resume the pipeline after providers owned by a task are released."""

        def schedule_continuation(_object=None):
            self._active_task = None
            self._continuation_pending = True
            QTimer.singleShot(
                0,
                lambda: self._resume_task_result(
                    callback,
                    successful,
                    results,
                    feedback,
                ),
            )

        task.destroyed.connect(schedule_continuation)

    def _resume_task_result(self, callback, successful, results, feedback) -> None:
        self._continuation_pending = False
        callback(successful, results, feedback)

    @staticmethod
    def _work_path(final_path: str, token: str, label: str, suffix: str) -> str:
        final = Path(final_path)
        return str(
            final.with_name(f".{final.stem}.archaeotrace-{token}-{label}{suffix}")
        )

    def _prepare_work_paths(self, request: DemBuildRequest) -> None:
        token = uuid.uuid4().hex
        # Older QGIS writes ASCII Grid content regardless of extension, while
        # QGIS 3.42+ requires a GDAL driver which supports direct creation.
        # A .tif staging name works in both cases; gdal:translate below then
        # guarantees that the published file is an actual GeoTIFF.
        self._raw_dem_path = self._work_path(request.dem_path, token, "tin", ".tif")
        self._dem_work_path = self._work_path(request.dem_path, token, "dem", ".tif")
        self._hillshade_work_path = self._work_path(
            request.hillshade_path,
            token,
            "hillshade",
            ".tif",
        )

    def _cleanup_work_files(self) -> None:
        primary_paths = [
            self._raw_dem_path,
            self._dem_work_path,
            self._hillshade_work_path,
        ]
        paths = list(primary_paths)
        for primary_path in primary_paths:
            if primary_path:
                paths.extend((f"{primary_path}.aux.xml", f"{primary_path}.ovr"))
        if self._raw_dem_path:
            raw = Path(self._raw_dem_path)
            paths.extend((f"{self._raw_dem_path}.prj", str(raw.with_suffix(".prj"))))
        for path in paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    @staticmethod
    def _validate_raster(
        path: str,
        expected_crs_authid: str,
        expected_dimensions: tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        if not is_tiff_file(path):
            raise DemInputError(f"Processing output is not a TIFF raster: {path}")

        layer = QgsRasterLayer(path, "ArchaeoTrace validation", "gdal")
        if not layer.isValid():
            raise DemInputError(f"Processing created an invalid raster: {path}")
        if layer.bandCount() < 1 or layer.width() <= 0 or layer.height() <= 0:
            raise DemInputError(f"Processing created an empty raster: {path}")
        if layer.crs().authid() != expected_crs_authid:
            raise DemInputError(
                "Processing output CRS does not match the terrain inputs: "
                f"{layer.crs().authid() or 'unknown'} != {expected_crs_authid}."
            )
        dimensions = (layer.width(), layer.height())
        if expected_dimensions is not None and dimensions != expected_dimensions:
            raise DemInputError(
                "Hillshade grid does not match the DEM grid: "
                f"{dimensions[0]} x {dimensions[1]} != "
                f"{expected_dimensions[0]} x {expected_dimensions[1]}."
            )
        return dimensions

    @staticmethod
    def _publish_outputs(pairs: tuple[tuple[str, str], ...]) -> None:
        """Replace final rasters while restoring previous files on failure."""

        try:
            publish_output_files(pairs)
        except DemSpecificationError as exc:
            raise DemInputError(str(exc)) from exc

    def start(self, request: DemBuildRequest) -> None:
        if self.is_running:
            raise DemInputError("A terrain task is already running.")

        self._request = request
        self._cancel_requested = False
        self._prepare_work_paths(request)
        try:
            dem_algorithm = self._algorithm(TIN_ALGORITHM_ID)
            parameters = {
                "INTERPOLATION_DATA": request.interpolation_data,
                "METHOD": 0,
                "EXTENT": request.extent,
                "PIXEL_SIZE": request.pixel_size,
                # QGIS 3.22-3.38 may put ASCII Grid data in this .tif staging
                # path; gdal:translate normalizes it before publication.
                "OUTPUT": self._raw_dem_path,
            }
            task, feedback = self._new_task(dem_algorithm, parameters)
            self._active_task = task
            task.progressChanged.connect(
                lambda progress: self.progressChanged.emit(progress * 0.65)
            )
            task.executed.connect(
                lambda successful, results: self._continue_after_task_destroyed(
                    task,
                    self._on_dem_finished,
                    successful,
                    results,
                    feedback,
                )
            )
            self.stageChanged.emit("dem")
            QgsApplication.taskManager().addTask(task)
        except DemInputError:
            self._active_task = None
            self._cleanup_work_files()
            raise
        except Exception as exc:
            self._active_task = None
            self._cleanup_work_files()
            raise DemInputError(f"Could not start TIN interpolation: {exc}") from exc

    def cancel(self) -> None:
        if self._active_task is None:
            if self._continuation_pending:
                self._cancel_requested = True
            return
        self._cancel_requested = True
        self._active_task.cancel()

    @staticmethod
    def _feedback_message(feedback, fallback: str) -> str:
        return feedback.errors[-1] if feedback.errors else fallback

    def _on_dem_finished(self, successful: bool, _results: dict, feedback) -> None:
        self._active_task = None
        if not successful:
            self._cleanup_work_files()
            if self._cancel_requested:
                self.canceled.emit()
            else:
                self.failed.emit(self._feedback_message(feedback, "TIN interpolation failed."))
            return
        if not os.path.exists(self._raw_dem_path):
            self._cleanup_work_files()
            self.failed.emit("TIN interpolation reported success but no grid was created.")
            return
        if self._cancel_requested:
            self._cleanup_work_files()
            self.canceled.emit()
            return

        try:
            translate_algorithm = self._algorithm(TRANSLATE_ALGORITHM_ID)
            candidates = {
                "INPUT": self._raw_dem_path,
                # Float32 preserves interpolated elevations consistently when
                # older QGIS releases expose the TIN as an ASCII grid.
                "DATA_TYPE": 6,
                "EXTRA": "",
                "OUTPUT": self._dem_work_path,
            }
            parameters = self._supported_parameters(translate_algorithm, candidates)
            task, translate_feedback = self._new_task(translate_algorithm, parameters)
            self._active_task = task
            task.progressChanged.connect(
                lambda progress: self.progressChanged.emit(65.0 + progress * 0.1)
            )
            task.executed.connect(
                lambda ok, results: self._continue_after_task_destroyed(
                    task,
                    self._on_translate_finished,
                    ok,
                    results,
                    translate_feedback,
                )
            )
            self.stageChanged.emit("translate")
            QgsApplication.taskManager().addTask(task)
        except Exception as exc:
            self._active_task = None
            self._cleanup_work_files()
            self.failed.emit(f"Could not start GeoTIFF conversion: {exc}")

    def _on_translate_finished(
        self,
        successful: bool,
        _results: dict,
        feedback,
    ) -> None:
        self._active_task = None
        if not successful:
            self._cleanup_work_files()
            if self._cancel_requested:
                self.canceled.emit()
            else:
                self.failed.emit(
                    self._feedback_message(feedback, "GeoTIFF conversion failed.")
                )
            return
        if not os.path.exists(self._dem_work_path):
            self._cleanup_work_files()
            self.failed.emit(
                "GeoTIFF conversion reported success but no DEM file was created."
            )
            return
        try:
            self._dem_dimensions = self._validate_raster(
                self._dem_work_path,
                self._request.crs_authid,
            )
        except DemInputError as exc:
            self._cleanup_work_files()
            self.failed.emit(str(exc))
            return
        if self._cancel_requested:
            self._cleanup_work_files()
            self.canceled.emit()
            return
        self._start_statistics()

    def _start_statistics(self) -> None:
        try:
            statistics_algorithm = self._algorithm(STATISTICS_ALGORITHM_ID)
            candidates = {
                "INPUT": self._dem_work_path,
                "BAND": 1,
                "OUTPUT_HTML_FILE": "TEMPORARY_OUTPUT",
            }
            parameters = self._supported_parameters(statistics_algorithm, candidates)
            task, statistics_feedback = self._new_task(
                statistics_algorithm,
                parameters,
            )
            self._active_task = task
            task.progressChanged.connect(
                lambda progress: self.progressChanged.emit(75.0 + progress * 0.1)
            )
            task.executed.connect(
                lambda ok, results: self._continue_after_task_destroyed(
                    task,
                    self._on_statistics_finished,
                    ok,
                    results,
                    statistics_feedback,
                )
            )
            self.stageChanged.emit("validate")
            QgsApplication.taskManager().addTask(task)
        except Exception as exc:
            self._active_task = None
            self._cleanup_work_files()
            self.failed.emit(f"Could not start DEM value validation: {exc}")

    def _on_statistics_finished(
        self,
        successful: bool,
        results: dict,
        feedback,
    ) -> None:
        self._active_task = None
        if not successful:
            self._cleanup_work_files()
            if self._cancel_requested:
                self.canceled.emit()
            else:
                self.failed.emit(
                    self._feedback_message(feedback, "DEM value validation failed.")
                )
            return
        try:
            minimum = float(results.get("MIN"))
            maximum = float(results.get("MAX"))
        except (TypeError, ValueError):
            minimum = math.nan
            maximum = math.nan
        if not math.isfinite(minimum) or not math.isfinite(maximum):
            self._cleanup_work_files()
            self.failed.emit("The DEM contains no finite terrain cells.")
            return
        if maximum <= minimum:
            self._cleanup_work_files()
            self.failed.emit("The DEM has no elevation range.")
            return
        if self._cancel_requested:
            self._cleanup_work_files()
            self.canceled.emit()
            return
        self._start_hillshade()

    def _start_hillshade(self) -> None:
        try:
            hillshade_algorithm = self._algorithm(HILLSHADE_ALGORITHM_ID)
            candidates = {
                "INPUT": self._dem_work_path,
                "BAND": 1,
                "Z_FACTOR": 1.0,
                "SCALE": 1.0,
                "AZIMUTH": 315.0,
                "ALTITUDE": 45.0,
                "COMPUTE_EDGES": True,
                "ZEVENBERGEN": False,
                "COMBINED": False,
                "MULTIDIRECTIONAL": False,
                "EXTRA": "",
                "OUTPUT": self._hillshade_work_path,
            }
            parameters = self._supported_parameters(hillshade_algorithm, candidates)
            task, hillshade_feedback = self._new_task(hillshade_algorithm, parameters)
            self._active_task = task
            task.progressChanged.connect(
                lambda progress: self.progressChanged.emit(85.0 + progress * 0.15)
            )
            task.executed.connect(
                lambda ok, results: self._continue_after_task_destroyed(
                    task,
                    self._on_hillshade_finished,
                    ok,
                    results,
                    hillshade_feedback,
                )
            )
            self.stageChanged.emit("hillshade")
            QgsApplication.taskManager().addTask(task)
        except Exception as exc:
            self._active_task = None
            self._cleanup_work_files()
            self.failed.emit(f"Could not start hillshade generation: {exc}")

    def _on_hillshade_finished(self, successful: bool, _results: dict, feedback) -> None:
        self._active_task = None
        if not successful:
            self._cleanup_work_files()
            if self._cancel_requested:
                self.canceled.emit()
            else:
                self.failed.emit(self._feedback_message(feedback, "Hillshade generation failed."))
            return
        if not os.path.exists(self._hillshade_work_path):
            self._cleanup_work_files()
            self.failed.emit("Hillshade processing reported success but no output file was created.")
            return
        if self._cancel_requested:
            self._cleanup_work_files()
            self.canceled.emit()
            return
        try:
            self._validate_raster(
                self._hillshade_work_path,
                self._request.crs_authid,
                expected_dimensions=self._dem_dimensions,
            )
            self._publish_outputs(
                (
                    (self._dem_work_path, self._request.dem_path),
                    (self._hillshade_work_path, self._request.hillshade_path),
                )
            )
        except DemInputError as exc:
            self._cleanup_work_files()
            self.failed.emit(str(exc))
            return
        self._cleanup_work_files()
        self.progressChanged.emit(100.0)
        self.succeeded.emit(self._request.dem_path, self._request.hillshade_path)
