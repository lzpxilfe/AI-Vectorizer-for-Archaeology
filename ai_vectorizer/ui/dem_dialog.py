# -*- coding: utf-8 -*-
"""Terrain reconstruction dialog for contours and spot heights."""

import os
from pathlib import Path

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)
from qgis.core import (
    Qgis,
    QgsFieldProxyModel,
    QgsMapLayerProxyModel,
    QgsProject,
    QgsRasterLayer,
)
from qgis.gui import QgsFieldComboBox, QgsMapLayerComboBox

from ..config import FIELD_ELEVATION
from ..core.dem_pipeline import (
    DemInputError,
    DemPipelineRunner,
    build_dem_request,
    loaded_project_paths,
)
from ..core.dem_spec import (
    DemSpecificationError,
    default_hillshade_path,
    estimate_grid,
    suggest_pixel_size,
)


LANG_EN = "en"


def _proxy_filter(owner, legacy_name, modern_name, qgis_enum_name):
    """Return a QGIS proxy enum across unscoped/scoped enum releases."""

    legacy = getattr(owner, legacy_name, None)
    if legacy is not None:
        return legacy
    scoped = getattr(owner, "Filter", None)
    if scoped is not None and hasattr(scoped, modern_name):
        return getattr(scoped, modern_name)
    return getattr(getattr(Qgis, qgis_enum_name), modern_name)


def _qt_value(legacy_name, scope_name):
    legacy = getattr(Qt, legacy_name, None)
    if legacy is not None:
        return legacy
    return getattr(getattr(Qt, scope_name), legacy_name)


def _message_box_button(name):
    legacy = getattr(QMessageBox, name, None)
    if legacy is not None:
        return legacy
    return getattr(QMessageBox.StandardButton, name)


class DemBuildDialog(QDialog):
    """Collect validated terrain inputs and run QGIS Processing in tasks."""

    def __init__(self, iface, contour_layer=None, raster_layer=None, language="ko", parent=None):
        super().__init__(parent or iface.mainWindow())
        self.iface = iface
        self.language = language
        self._last_contour_id = None
        self._user_changed_pixel_size = False
        self._shutting_down = False
        self._grid_is_valid = False
        self._delete_when_idle = False

        self.runner = DemPipelineRunner(self)
        self.runner.stageChanged.connect(self._on_stage_changed)

        self._setup_ui()
        # Signals which require widgets are connected after construction.
        self.runner.progressChanged.connect(lambda value: self.progress.setValue(round(value)))
        self.runner.succeeded.connect(self._on_succeeded)
        self.runner.failed.connect(self._on_failed)
        self.runner.canceled.connect(self._on_canceled)

        self.set_inputs(contour_layer, raster_layer, language, reset_outputs=True)

    def _tr(self, ko, en):
        return en if self.language == LANG_EN else ko

    def _setup_ui(self):
        self.setMinimumWidth(560)
        self.setWindowModality(_qt_value("WindowModal", "WindowModality"))

        root = QVBoxLayout(self)
        self.intro_label = QLabel()
        self.intro_label.setWordWrap(True)
        root.addWidget(self.intro_label)

        input_group = QGroupBox()
        input_form = QFormLayout(input_group)

        self.contour_combo = QgsMapLayerComboBox()
        self.contour_combo.setFilters(
            _proxy_filter(QgsMapLayerProxyModel, "LineLayer", "LineLayer", "LayerFilter")
        )
        self.contour_label = QLabel()
        input_form.addRow(self.contour_label, self.contour_combo)

        self.contour_field = QgsFieldComboBox()
        self.contour_field.setFilters(
            _proxy_filter(QgsFieldProxyModel, "Numeric", "Numeric", "FieldFilter")
        )
        self.contour_field_label = QLabel()
        input_form.addRow(self.contour_field_label, self.contour_field)

        self.spot_combo = QgsMapLayerComboBox()
        self.spot_combo.setFilters(
            _proxy_filter(QgsMapLayerProxyModel, "PointLayer", "PointLayer", "LayerFilter")
        )
        self.spot_combo.setAllowEmptyLayer(True)
        self.spot_label = QLabel()
        input_form.addRow(self.spot_label, self.spot_combo)

        self.spot_field = QgsFieldComboBox()
        self.spot_field.setFilters(
            _proxy_filter(QgsFieldProxyModel, "Numeric", "Numeric", "FieldFilter")
        )
        self.spot_field_label = QLabel()
        input_form.addRow(self.spot_field_label, self.spot_field)
        root.addWidget(input_group)
        self.input_group = input_group

        grid_group = QGroupBox()
        grid_form = QFormLayout(grid_group)
        self.pixel_size = QDoubleSpinBox()
        self.pixel_size.setDecimals(6)
        self.pixel_size.setRange(0.000001, 1_000_000_000.0)
        self.pixel_size.setValue(1.0)
        self.pixel_size.setSuffix(" m")
        self.pixel_label = QLabel()
        grid_form.addRow(self.pixel_label, self.pixel_size)
        self.grid_summary = QLabel()
        self.grid_summary.setWordWrap(True)
        self.grid_summary_label = QLabel()
        grid_form.addRow(self.grid_summary_label, self.grid_summary)
        root.addWidget(grid_group)
        self.grid_group = grid_group

        output_group = QGroupBox()
        output_form = QFormLayout(output_group)
        self.dem_path, self.dem_browse, dem_row = self._path_row(self._browse_dem)
        self.dem_path_label = QLabel()
        output_form.addRow(self.dem_path_label, dem_row)
        self.hillshade_path, self.hillshade_browse, hillshade_row = self._path_row(
            self._browse_hillshade
        )
        self.hillshade_path_label = QLabel()
        output_form.addRow(self.hillshade_path_label, hillshade_row)
        root.addWidget(output_group)
        self.output_group = output_group

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        root.addWidget(self.progress)

        button_row = QHBoxLayout()
        button_row.addStretch()
        self.build_button = QPushButton()
        self.build_button.clicked.connect(self._start_build)
        button_row.addWidget(self.build_button)
        self.cancel_button = QPushButton()
        self.cancel_button.clicked.connect(self.runner.cancel)
        self.cancel_button.setEnabled(False)
        button_row.addWidget(self.cancel_button)
        self.close_button = QPushButton()
        self.close_button.clicked.connect(self.close)
        button_row.addWidget(self.close_button)
        root.addLayout(button_row)

        self.contour_combo.layerChanged.connect(self._on_contour_changed)
        self.spot_combo.layerChanged.connect(self._on_spot_changed)
        self.pixel_size.valueChanged.connect(self._on_pixel_size_changed)

        self.apply_language()

    @staticmethod
    def _path_row(callback):
        line_edit = QLineEdit()
        browse = QPushButton("📂")
        browse.setFixedWidth(36)
        browse.clicked.connect(callback)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(line_edit)
        row.addWidget(browse)
        return line_edit, browse, row

    def apply_language(self):
        self.setWindowTitle(self._tr("지형 복원: DEM 생성", "Terrain Reconstruction: Build DEM"))
        self.intro_label.setText(
            self._tr(
                "고도값을 가진 등고선을 구조선으로 사용해 선형 TIN DEM과 hillshade를 생성합니다. "
                "결과는 입력 피처의 범위와 정확도에 의존합니다.",
                "Build a linear-TIN DEM and hillshade using elevated contours as structure lines. "
                "The result depends on the coverage and accuracy of the input features.",
            )
        )
        self.input_group.setTitle(self._tr("1. 고도 데이터", "1. Elevation data"))
        self.contour_label.setText(self._tr("등고선 레이어:", "Contour layer:"))
        self.contour_field_label.setText(self._tr("고도 필드:", "Elevation field:"))
        self.spot_label.setText(
            self._tr("표고점 레이어 (선택):", "Spot-height layer (optional):")
        )
        self.spot_field_label.setText(self._tr("표고점 필드:", "Spot-height field:"))
        self.grid_group.setTitle(self._tr("2. DEM 격자", "2. DEM grid"))
        self.pixel_label.setText(self._tr("격자 크기:", "Pixel size:"))
        self.grid_summary_label.setText(self._tr("예상 출력:", "Estimated output:"))
        self.output_group.setTitle(self._tr("3. 출력", "3. Outputs"))
        self.dem_path_label.setText("DEM GeoTIFF:")
        self.hillshade_path_label.setText("Hillshade GeoTIFF:")
        self.build_button.setText(self._tr("⏵ DEM 생성", "⏵ Build DEM"))
        self.cancel_button.setText(self._tr("취소", "Cancel"))
        self.close_button.setText(self._tr("닫기", "Close"))
        if not self.runner.is_running:
            self.status_label.setText(
                self._tr(
                    "투영 좌표계(m)와 서로 다른 두 개 이상의 고도값이 필요합니다.",
                    "A projected CRS in metres and at least two distinct elevation values are required.",
                )
            )

    def set_inputs(self, contour_layer, raster_layer=None, language=None, reset_outputs=False):
        """Refresh defaults from the main dock without interrupting a task."""

        self._shutting_down = False
        if language is not None:
            self.language = language
            self.apply_language()
        if self.runner.is_running:
            return

        contour_id = contour_layer.id() if contour_layer is not None else None
        changed = contour_id != self._last_contour_id
        self._last_contour_id = contour_id
        self.contour_combo.setLayer(contour_layer)
        self.contour_field.setLayer(contour_layer)
        if contour_layer is not None and contour_layer.fields().indexOf(FIELD_ELEVATION) >= 0:
            self.contour_field.setField(FIELD_ELEVATION)

        if contour_layer is not None and (changed or not self._user_changed_pixel_size):
            suggestion = self._suggest_pixel_size(contour_layer, raster_layer)
            if suggestion is not None:
                self.pixel_size.blockSignals(True)
                self.pixel_size.setValue(suggestion)
                self.pixel_size.blockSignals(False)

        if contour_layer is not None and (changed or reset_outputs):
            dem_path = self._default_dem_path(contour_layer)
            self.dem_path.setText(dem_path)
            self.hillshade_path.setText(default_hillshade_path(dem_path))
        self._update_grid_summary()

    def _suggest_pixel_size(self, contour_layer, raster_layer):
        if (
            raster_layer is not None
            and raster_layer.isValid()
            and raster_layer.crs() == contour_layer.crs()
        ):
            values = (
                abs(float(raster_layer.rasterUnitsPerPixelX())),
                abs(float(raster_layer.rasterUnitsPerPixelY())),
            )
            if all(value > 0 for value in values):
                extent = contour_layer.extent()
                if not extent.isEmpty() and extent.width() > 0 and extent.height() > 0:
                    safe_default = suggest_pixel_size(extent.width(), extent.height())
                    return max(max(values), safe_default)
                return max(values)
        extent = contour_layer.extent()
        if not extent.isEmpty() and extent.width() > 0 and extent.height() > 0:
            try:
                return suggest_pixel_size(extent.width(), extent.height())
            except DemSpecificationError:
                return None
        return None

    @staticmethod
    def _default_dem_path(contour_layer):
        source_path = str(contour_layer.source() or "").split("|", 1)[0]
        source = Path(source_path)
        if contour_layer.providerType() != "memory" and source.is_file():
            folder = source.parent
            stem = source.stem
        else:
            project_folder = QgsProject.instance().homePath()
            folder = Path(project_folder) if project_folder else Path(os.getcwd())
            stem = contour_layer.name() or "terrain"
        safe_stem = "".join(char if char.isalnum() or char in "-_" else "_" for char in stem)
        return str(folder / f"{safe_stem}_dem.tif")

    def _on_contour_changed(self, layer):
        self.contour_field.setLayer(layer)
        if layer is not None and layer.fields().indexOf(FIELD_ELEVATION) >= 0:
            self.contour_field.setField(FIELD_ELEVATION)
        self._last_contour_id = layer.id() if layer is not None else None
        self._user_changed_pixel_size = False
        if layer is not None:
            suggestion = self._suggest_pixel_size(layer, None)
            if suggestion is not None:
                self.pixel_size.blockSignals(True)
                self.pixel_size.setValue(suggestion)
                self.pixel_size.blockSignals(False)
            dem_path = self._default_dem_path(layer)
            self.dem_path.setText(dem_path)
            self.hillshade_path.setText(default_hillshade_path(dem_path))
        self._update_grid_summary()

    def _on_spot_changed(self, layer):
        self.spot_field.setLayer(layer)
        if layer is not None and layer.fields().indexOf(FIELD_ELEVATION) >= 0:
            self.spot_field.setField(FIELD_ELEVATION)

    def _on_pixel_size_changed(self, _value):
        self._user_changed_pixel_size = True
        self._update_grid_summary()

    def _update_grid_summary(self):
        layer = self.contour_combo.currentLayer()
        if layer is None or layer.extent().isEmpty():
            self.grid_summary.setText(self._tr("입력 범위 없음", "No input extent"))
            self._grid_is_valid = False
            self._refresh_build_enabled()
            return
        try:
            grid = estimate_grid(
                layer.extent().width(),
                layer.extent().height(),
                self.pixel_size.value(),
            )
            self.grid_summary.setText(
                self._tr(
                    "{columns:,} × {rows:,} ({cells:,} 셀)",
                    "{columns:,} × {rows:,} ({cells:,} cells)",
                ).format(columns=grid.columns, rows=grid.rows, cells=grid.cells)
            )
            self._grid_is_valid = True
        except DemSpecificationError as exc:
            self.grid_summary.setText(str(exc))
            self._grid_is_valid = False
        self._refresh_build_enabled()

    def _refresh_build_enabled(self):
        if hasattr(self, "build_button"):
            self.build_button.setEnabled(
                self._grid_is_valid and not self.runner.is_running
            )

    def _browse_dem(self):
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            self._tr("DEM GeoTIFF 저장", "Save DEM GeoTIFF"),
            self.dem_path.text(),
            "GeoTIFF (*.tif *.tiff)",
        )
        if path:
            self.dem_path.setText(path)
            self.hillshade_path.setText(default_hillshade_path(path))

    def _browse_hillshade(self):
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            self._tr("Hillshade GeoTIFF 저장", "Save Hillshade GeoTIFF"),
            self.hillshade_path.text(),
            "GeoTIFF (*.tif *.tiff)",
        )
        if path:
            self.hillshade_path.setText(path)

    def _start_build(self):
        try:
            request = build_dem_request(
                contour_layer=self.contour_combo.currentLayer(),
                contour_field=self.contour_field.currentField(),
                pixel_size=self.pixel_size.value(),
                dem_path=self.dem_path.text(),
                hillshade_path=self.hillshade_path.text(),
                spot_layer=self.spot_combo.currentLayer(),
                spot_field=self.spot_field.currentField(),
            )
        except (DemInputError, DemSpecificationError) as exc:
            QMessageBox.warning(self, self._tr("입력 확인", "Check Inputs"), str(exc))
            return

        existing = [
            path
            for path in (request.dem_path, request.hillshade_path)
            if os.path.exists(path)
        ]
        loaded = self._loaded_output_paths(
            (request.dem_path, request.hillshade_path)
        )
        if loaded:
            QMessageBox.warning(
                self,
                self._tr("출력 레이어 사용 중", "Output Layer Is Loaded"),
                self._tr(
                    "안전한 덮어쓰기를 위해 다음 레이어를 QGIS에서 제거한 뒤 다시 실행하세요:\n{paths}",
                    "Remove these output layers from QGIS before overwriting them safely:\n{paths}",
                ).format(paths="\n".join(loaded)),
            )
            return
        if existing:
            answer = QMessageBox.question(
                self,
                self._tr("기존 파일 덮어쓰기", "Overwrite Existing Files"),
                self._tr(
                    "다음 파일을 덮어쓸까요?\n{paths}",
                    "Overwrite these files?\n{paths}",
                ).format(paths="\n".join(existing)),
                _message_box_button("Yes") | _message_box_button("No"),
                _message_box_button("No"),
            )
            if answer != _message_box_button("Yes"):
                return

        self.dem_path.setText(request.dem_path)
        self.hillshade_path.setText(request.hillshade_path)
        self._set_running(True)
        self.progress.setValue(0)
        try:
            self.runner.start(request)
        except (DemInputError, DemSpecificationError) as exc:
            self._set_running(False)
            QMessageBox.critical(self, self._tr("실행 오류", "Processing Error"), str(exc))

    def _set_running(self, running):
        for widget in (
            self.contour_combo,
            self.contour_field,
            self.spot_combo,
            self.spot_field,
            self.pixel_size,
            self.dem_path,
            self.hillshade_path,
            self.dem_browse,
            self.hillshade_browse,
            self.build_button,
        ):
            widget.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.close_button.setEnabled(not running)
        if not running:
            self._refresh_build_enabled()

    @staticmethod
    def _loaded_output_paths(paths):
        return loaded_project_paths(paths)

    def _on_stage_changed(self, stage):
        if stage == "dem":
            self.status_label.setText(self._tr("선형 TIN DEM 생성 중…", "Building linear-TIN DEM…"))
        elif stage == "translate":
            self.status_label.setText(
                self._tr(
                    "TIN 격자를 GeoTIFF로 변환 중…",
                    "Converting the TIN grid to GeoTIFF…",
                )
            )
        elif stage == "validate":
            self.status_label.setText(
                self._tr(
                    "DEM 유효 값을 검증 중…",
                    "Validating finite DEM values…",
                )
            )
        else:
            self.status_label.setText(self._tr("Hillshade 생성 중…", "Building hillshade…"))

    def _on_succeeded(self, dem_path, hillshade_path):
        self._set_running(False)
        if self._delete_when_idle:
            self._finish_shutdown_if_needed()
            return
        added = []
        failed = []
        for path, suffix in ((dem_path, "DEM"), (hillshade_path, "Hillshade")):
            layer = QgsRasterLayer(path, f"{Path(path).stem} ({suffix})")
            if layer.isValid():
                QgsProject.instance().addMapLayer(layer)
                added.append(path)
            else:
                failed.append(path)

        if failed:
            self.status_label.setText(
                self._tr(
                    "파일은 생성됐지만 일부 결과를 QGIS에 로드하지 못했습니다.",
                    "Files were created, but QGIS could not load every result.",
                )
            )
            QMessageBox.warning(
                self,
                self._tr("결과 로드 경고", "Result Load Warning"),
                "\n".join(failed),
            )
        else:
            self.status_label.setText(
                self._tr(
                    "DEM과 hillshade 생성을 완료하고 QGIS에 추가했습니다.",
                    "DEM and hillshade were created and added to QGIS.",
                )
            )
        self.progress.setValue(100)
        self._finish_shutdown_if_needed()

    def _on_failed(self, message):
        self._set_running(False)
        if self._delete_when_idle:
            self._finish_shutdown_if_needed()
            return
        self.status_label.setText(self._tr("DEM 생성 실패", "DEM build failed"))
        QMessageBox.critical(self, self._tr("처리 실패", "Processing Failed"), message)
        self._finish_shutdown_if_needed()

    def _on_canceled(self):
        self._set_running(False)
        if self._delete_when_idle:
            self._finish_shutdown_if_needed()
            return
        self.status_label.setText(self._tr("작업을 취소했습니다.", "Task canceled."))
        self._finish_shutdown_if_needed()

    def _finish_shutdown_if_needed(self):
        if self._delete_when_idle and not self.runner.is_running:
            self.deleteLater()

    def shutdown(self, permanent=False):
        """Stop accepting UI work while the plugin dock is being closed."""

        self._shutting_down = True
        self._delete_when_idle = self._delete_when_idle or permanent
        if self.runner.is_running:
            self.runner.cancel()
        self.hide()
        self._finish_shutdown_if_needed()

    def closeEvent(self, event):
        if self.runner.is_running and not self._shutting_down:
            QMessageBox.information(
                self,
                self._tr("작업 진행 중", "Task In Progress"),
                self._tr("진행 중인 작업을 먼저 취소하세요.", "Cancel the running task before closing."),
            )
            event.ignore()
            return
        super().closeEvent(event)
