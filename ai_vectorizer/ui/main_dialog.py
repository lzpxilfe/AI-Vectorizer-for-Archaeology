# -*- coding: utf-8 -*-
"""
ArchaeoTrace - AI-assisted contour tracing for historical maps
Dockable panel with guided workflow and tooltips
"""

import os
import importlib.util
import json
import shutil
import tempfile
import traceback
from datetime import datetime, timezone
from qgis.PyQt.QtWidgets import (
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QCheckBox,
    QPushButton,
    QGroupBox,
    QFileDialog,
    QLineEdit,
    QSlider,
    QMessageBox,
    QApplication,
)
from qgis.core import (
    QgsProject,
    QgsMapLayerProxyModel,
    QgsVectorLayer,
    QgsField,
    QgsVectorFileWriter,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsApplication,
    QgsTask,
    QgsSymbol,
    QgsSingleSymbolRenderer,
    Qgis,
)
from qgis.gui import QgsMapLayerComboBox
from qgis.PyQt.QtCore import Qt, QSettings, QStandardPaths, QTimer
try:
    from qgis.PyQt.QtCore import QVariant
except ImportError:  # PyQt6/QGIS 4
    QVariant = None
try:
    from qgis.PyQt.QtCore import QMetaType
except ImportError:
    QMetaType = None
from qgis.PyQt.QtGui import QColor

from ..core.dependencies import get_cv2_error_text, get_opencv_install_command, is_cv2_available
from ..core.livewire import is_livewire_available
from ..core.raster_utils import compute_resampled_dimensions, read_raster_bands
from ..core.dem_pipeline import (
    layer_file_path as _layer_file_path,
    project_layers_using_path as _project_layers_using_path,
)
from ..recovery import (
    RECOVERY_STATE_ENHANCED,
    RECOVERY_STATE_INK,
    RECOVERY_STATE_INK_FALLBACK,
    RECOVERY_STATE_RECOVERING,
    require_recovery_state,
)
from ..config import (
    DEFAULT_CRS_AUTHID,
    DEFAULT_EDGE_METHOD,
    DEFAULT_FREEDOM_SLIDER_VALUE,
    DEFAULT_OUTPUT_LAYER_NAME,
    DEFAULT_VECTOR_FILE_ENCODING,
    EDGE_METHOD_BY_MODEL,
    FIELD_ELEVATION,
    FIELD_ID,
    MAX_RASTER_BANDS_FOR_RGB,
    MOBILE_SAM_INSTALL_COMMAND,
    MODE_NAME_BY_MODEL,
    MODEL_MENU_LABELS,
    MODEL_IDX_INK,
    MODEL_IDX_HED,
    MODEL_IDX_LEGACY_CANNY,
    MODEL_IDX_LSD,
    MODEL_IDX_MOBILE_SAM,
    MODEL_IDX_SAM,
    PLUGIN_NAME,
    PREVIEW_EDGE_MAX_DIMENSION,
    SAM_INSTALL_COMMAND,
    SAM_ASSIST_EDGE_METHOD,
    SAM_ENGINE_SPEC_BY_MODEL,
    SAM_MODEL_INDICES,
    SAM_REPORT_FILENAME,
    SETTINGS_LANG_KEY,
    STATUS_STYLE_ERROR,
    STATUS_STYLE_INFO,
    STATUS_STYLE_NEUTRAL,
    STATUS_STYLE_READY,
    STATUS_STYLE_WARNING,
    TRACE_BUTTON_ACTIVE_STYLE,
    TRACE_BUTTON_IDLE_STYLE,
)


LANG_KO = "ko"
LANG_EN = "en"
RECOVERY_RUNTIME_INSTALL_COMMAND = (
    'python -m pip install "onnxruntime>=1.17,<2"'
)


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


def _map_layer_filter(name):
    legacy = getattr(QgsMapLayerProxyModel, name, None)
    if legacy is not None:
        return legacy
    scoped = getattr(QgsMapLayerProxyModel, "Filter", None)
    if scoped is not None and hasattr(scoped, name):
        return getattr(scoped, name)
    return getattr(Qgis.LayerFilter, name)


def _field_type(name):
    # QGIS 3.22 exposes QMetaType.Type through PyQt5, but QgsField still
    # requires QVariant.Type there. Prefer that binding whenever it exists;
    # PyQt6/QGIS 4 falls through to the scoped QMetaType enum.
    if QVariant is not None and hasattr(QVariant, name):
        return getattr(QVariant, name)
    meta_name = "QString" if name == "String" else name
    meta_types = getattr(QMetaType, "Type", None)
    if meta_types is not None and hasattr(meta_types, meta_name):
        return getattr(meta_types, meta_name)
    raise RuntimeError(f"Qt field type is unavailable: {name}")


def _standard_location(name):
    legacy = getattr(QStandardPaths, name, None)
    if legacy is not None:
        return legacy
    return getattr(QStandardPaths.StandardLocation, name)


def _writer_no_error():
    legacy = getattr(QgsVectorFileWriter, "NoError", None)
    if legacy is not None:
        return legacy
    return QgsVectorFileWriter.WriterError.NoError


def _task_can_cancel():
    legacy = getattr(QgsTask, "CanCancel", None)
    if legacy is not None:
        return legacy
    scoped = getattr(QgsTask, "Flag", None)
    if scoped is not None and hasattr(scoped, "CanCancel"):
        return scoped.CanCancel
    return Qgis.TaskFlag.CanCancel


def _write_vector_layer(layer, path, crs):
    modern = getattr(QgsVectorFileWriter, "writeAsVectorFormatV3", None)
    if modern is not None:
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "ESRI Shapefile"
        options.fileEncoding = DEFAULT_VECTOR_FILE_ENCODING
        return modern(
            layer,
            path,
            QgsProject.instance().transformContext(),
            options,
        )
    return QgsVectorFileWriter.writeAsVectorFormat(
        layer,
        path,
        DEFAULT_VECTOR_FILE_ENCODING,
        crs,
        "ESRI Shapefile",
    )


def _exec_dialog(dialog):
    execute = getattr(dialog, "exec", None)
    if execute is None:
        execute = dialog.exec_
    return execute()


class _RecoveryInstallTask(QgsTask):
    """Install or repair the pinned bundle after one explicit button press."""

    def __init__(self, cache_root, callback, *, repair_corrupt=False):
        super().__init__("ArchaeoTrace install recovery model", _task_can_cancel())
        self.cache_root = cache_root
        self.callback = callback
        self.repair_corrupt = bool(repair_corrupt)
        self.bundle = None
        self.error = None

    def run(self):
        if self.isCanceled():
            return False
        try:
            # This is the only Smart Recovery path allowed to open the
            # network. Inspection, tracing, and retry remain offline.
            from ..core.model_store import (
                ModelDownloadCancelled,
                fetch_bundle,
                repair_bundle,
            )

            try:
                install = repair_bundle if self.repair_corrupt else fetch_bundle
                self.bundle = install(
                    self.cache_root,
                    cancel_check=self.isCanceled,
                )
            except ModelDownloadCancelled:
                return False
            # The verified store transaction is the commit point. A cancel
            # flag arriving after it returns must not report a false failure.
            return True
        except Exception as exc:
            self.error = exc
            return False

    def finished(self, result):
        self.callback(self, bool(result), self.bundle, self.error)


class _RecoveryPrepareTask(QgsTask):
    """Verify and initialize an already-installed bundle off the UI thread."""

    def __init__(self, cache_root, generation, runtime_available, callback):
        super().__init__("ArchaeoTrace prepare recovery model", _task_can_cancel())
        self.cache_root = cache_root
        self.generation = int(generation)
        self.runtime_available = bool(runtime_available)
        self.callback = callback
        self.status = None
        self.engine = None
        self.error = None

    def run(self):
        if self.isCanceled():
            return False
        try:
            from ..core.efficientsam_recovery import EfficientSAMRecoveryEngine

            # Both SHA-256 verification and ONNX session construction can be
            # expensive for the pinned 41 MB bundle. They belong in QgsTask,
            # never in a checkbox or trace-button callback.
            self.status = EfficientSAMRecoveryEngine.inspect(self.cache_root)
            if self.isCanceled():
                return False
            if self.status.ready and self.runtime_available:
                self.engine = EfficientSAMRecoveryEngine(self.cache_root)
            return not self.isCanceled()
        except Exception as exc:
            self.error = exc
            return False

    def finished(self, result):
        self.callback(
            self,
            bool(result),
            self.status,
            self.engine,
            self.error,
        )


class _TemporaryPreviewStore:
    """Tie private preview files to their corresponding project layers."""

    def __init__(self, project):
        self._project = project
        self._directories = {}
        self._connected = True
        project.layerRemoved.connect(self._on_layer_removed)

    def track(self, layer, temporary_directory):
        layer_id = layer.id()
        previous = self._directories.pop(layer_id, None)
        if previous is not None:
            self._cleanup_directory(previous)
        self._directories[layer_id] = temporary_directory

    @staticmethod
    def _cleanup_directory(temporary_directory):
        try:
            temporary_directory.cleanup()
        except OSError as exc:
            # A provider on Windows may release its final handle one event
            # later. cleanup() has already detached TemporaryDirectory's own
            # finalizer, so explicitly retry after pending deletions run.
            print(f"Failed to remove edge-preview files: {exc}")
            path = temporary_directory.name
            QTimer.singleShot(
                0,
                lambda: _TemporaryPreviewStore._retry_directory_cleanup(path),
            )

    @staticmethod
    def _retry_directory_cleanup(path):
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"Failed to remove edge-preview files after retry: {exc}")

    def _on_layer_removed(self, layer_id):
        temporary_directory = self._directories.pop(str(layer_id), None)
        if temporary_directory is not None:
            self._cleanup_directory(temporary_directory)

    def clear(self):
        for layer_id in tuple(self._directories):
            try:
                if self._project.mapLayer(layer_id) is not None:
                    self._project.removeMapLayer(layer_id)
            except RuntimeError:
                pass
            # layerRemoved is synchronous, but retain this fallback for a
            # project which is already shutting down.
            temporary_directory = self._directories.pop(layer_id, None)
            if temporary_directory is not None:
                self._cleanup_directory(temporary_directory)

    def shutdown(self):
        self.clear()
        if not self._connected:
            return
        try:
            self._project.layerRemoved.disconnect(self._on_layer_removed)
        except (RuntimeError, TypeError):
            pass
        self._connected = False


class AIVectorizerDock(QDockWidget):
    """Dockable panel for ArchaeoTrace plugin."""

    def __init__(self, iface, parent=None):
        super().__init__(PLUGIN_NAME, parent)
        self.iface = iface
        self.setAllowedAreas(
            _qt_value("LeftDockWidgetArea", "DockWidgetArea")
            | _qt_value("RightDockWidgetArea", "DockWidgetArea")
        )

        self.active_tool = None
        self.output_layer = None
        self.dem_dialog = None
        self.sam_engine = None
        self.sam_engine_key = None
        self.recovery_engine = None
        self.recovery_install_task = None
        self.recovery_prepare_task = None
        self._recovery_prepare_generation = 0
        self._recovery_model_status = None
        self._recovery_prepare_error = ""
        self._shutting_down = False
        self._recovery_state = RECOVERY_STATE_INK
        self._recovery_detail = ""
        self._preview_store = _TemporaryPreviewStore(QgsProject.instance())
        self.current_language = self._load_language()
        self._configure_hed_storage()

        main_widget = QWidget()
        self.layout = QVBoxLayout()
        main_widget.setLayout(self.layout)
        self.setWidget(main_widget)

        self.setup_ui()

    def _tr(self, ko, en):
        return en if self.current_language == LANG_EN else ko

    def _load_language(self):
        settings = QSettings()
        value = settings.value(SETTINGS_LANG_KEY, None)
        if value is None:
            locale = str(settings.value("locale/userLocale", "ko"))
            return LANG_EN if locale.lower().startswith("en") else LANG_KO
        lang = str(value)
        return lang if lang in (LANG_KO, LANG_EN) else LANG_KO

    def _save_language(self):
        QSettings().setValue(SETTINGS_LANG_KEY, self.current_language)

    @staticmethod
    def _log_nonfatal_ui_error(context, exc):
        print(f"{context}: {exc}")

    @staticmethod
    def _same_layer(first, second):
        if first is second:
            return True
        if first is None or second is None:
            return False
        try:
            return first.id() == second.id()
        except RuntimeError:
            return False

    def _model_items(self):
        return [
            MODEL_MENU_LABELS[idx][self.current_language]
            for idx in (
                MODEL_IDX_INK,
                MODEL_IDX_LSD,
                MODEL_IDX_HED,
                MODEL_IDX_MOBILE_SAM,
                MODEL_IDX_SAM,
                MODEL_IDX_LEGACY_CANNY,
            )
        ]

    def _mode_name(self, idx):
        return MODE_NAME_BY_MODEL.get(idx, "OpenCV")

    def _set_status_label(self, text, tone="neutral"):
        style_by_tone = {
            "neutral": STATUS_STYLE_NEUTRAL,
            "ready": STATUS_STYLE_READY,
            "info": STATUS_STYLE_INFO,
            "warning": STATUS_STYLE_WARNING,
            "error": STATUS_STYLE_ERROR,
        }
        self.status_label.setText(text)
        self.status_label.setStyleSheet(style_by_tone.get(tone, STATUS_STYLE_NEUTRAL))

    def _set_trace_button_idle(self):
        self.trace_btn.setChecked(False)
        self.trace_btn.setText(self._tr("🖊️ 트레이싱 시작", "🖊️ Start Tracing"))
        self.trace_btn.setStyleSheet(TRACE_BUTTON_IDLE_STYLE)

    def _set_trace_button_active(self):
        self.trace_btn.setText(self._tr("⏹️ 중지", "⏹️ Stop"))
        self.trace_btn.setStyleSheet(TRACE_BUTTON_ACTIVE_STYLE)

    def _set_ready_state(self, prompt=False):
        text = self._tr("✅ 준비 완료! 트레이싱을 시작하세요", "✅ Ready! Start tracing") if prompt else self._tr("✅ 준비 완료", "✅ Ready")
        self._set_status_label(text, "ready" if prompt else "neutral")

    def _set_tracing_state(self, mode_name):
        self._set_status_label(
            self._tr("🖊️ [{mode}] 등고선을 클릭하세요", "🖊️ [{mode}] Click on contours").format(mode=mode_name),
            "neutral",
        )
        self._set_trace_button_active()

    def _set_trace_configuration_enabled(self, enabled):
        """Keep the active map tool and the dock's visible model in sync."""

        for widget in (
            self.lang_combo,
            self.layer_combo,
            self.shp_path,
            self.browse_btn,
            self.create_shp_btn,
            self.vector_combo,
            self.smart_recovery_check,
            self.recovery_install_btn,
            self.advanced_check,
            self.model_combo,
            self.sam_check_btn,
            self.sam_report_btn,
            self.sam_download_btn,
            self.freehand_check,
            self.auto_path_check,
            self.freedom_slider,
            self.preview_edge_btn,
        ):
            widget.setEnabled(enabled)
        self.recovery_install_btn.setEnabled(
            enabled
            and self.recovery_install_task is None
            and self.recovery_prepare_task is None
        )
        self.recovery_retry_btn.setEnabled(
            not enabled
            and self.active_tool is not None
            and bool(
                getattr(self.active_tool, "smart_recovery_enabled", False)
            )
        )
        self._update_dem_button_for_tracing(not enabled)

    def _set_idle_ui(self, prompt=False):
        self._set_trace_configuration_enabled(True)
        self._set_trace_button_idle()
        self._set_ready_state(prompt=prompt)
        self._refresh_recovery_availability()

    def _set_sam_status(self, text, tone="neutral"):
        style_by_tone = {
            "neutral": "font-size: 10px;",
            "info": STATUS_STYLE_INFO,
            "warning": STATUS_STYLE_WARNING,
            "error": STATUS_STYLE_ERROR,
        }
        self.sam_status.setText(text)
        self.sam_status.setStyleSheet(style_by_tone.get(tone, "font-size: 10px;"))

    def _set_model_aux_visibility(self, show_check=False, show_report=False, show_download=False, show_install=False):
        self.sam_check_btn.setVisible(show_check)
        self.sam_report_btn.setVisible(show_report)
        self.sam_download_btn.setVisible(show_download)
        self.install_guide.setVisible(show_install)
        self.install_cmd.setVisible(show_install)

    def _set_install_hint(self, label, command):
        self.install_guide.setText(label)
        self.install_cmd.setText(command)

    def _opencv_detail_text(self):
        detail = get_cv2_error_text()
        if not detail:
            return ""
        return self._tr(f"\n원인: {detail}", f"\nDetail: {detail}")

    def _show_opencv_warning(self, feature_name):
        command = get_opencv_install_command()
        QMessageBox.warning(
            self,
            self._tr("OpenCV 필요", "OpenCV Required"),
            self._tr(
                "{feature} 기능에는 OpenCV(`cv2`)가 필요합니다.\nQGIS Python 환경에 아래 명령으로 설치하세요:\n{cmd}{detail}",
                "{feature} requires OpenCV (`cv2`).\nInstall it into the QGIS Python environment with:\n{cmd}{detail}",
            ).format(
                feature=feature_name,
                cmd=command,
                detail=self._opencv_detail_text(),
            ),
        )

    def _download_button_text(self, model_idx=None):
        idx = self.model_combo.currentIndex() if model_idx is None else model_idx
        if idx == MODEL_IDX_HED:
            return self._tr("📥 HED 다운로드", "📥 Download HED")
        if idx not in SAM_MODEL_INDICES:
            return self._tr("⬇️ 모델 다운로드", "⬇️ Download Model")
        model_name = self._mode_name(idx)
        size_hint = self._sam_size_hint_mb(idx)
        size_text_ko = f" (~{size_hint}MB)" if size_hint else ""
        size_text_en = f" (~{size_hint}MB)" if size_hint else ""
        return self._tr(
            f"⬇️ {model_name} 다운로드{size_text_ko}",
            f"⬇️ Download {model_name}{size_text_en}",
        )

    @staticmethod
    def _hed_size_hint_mb():
        from ..core.edge_detector import EdgeDetector
        return getattr(EdgeDetector, "HED_MODEL_SIZE_MB", 56)

    def _sam_engine_spec(self, model_idx=None):
        idx = self.model_combo.currentIndex() if model_idx is None else model_idx
        return SAM_ENGINE_SPEC_BY_MODEL.get(idx)

    def _is_sam_model(self, model_idx=None):
        idx = self.model_combo.currentIndex() if model_idx is None else model_idx
        return idx in SAM_MODEL_INDICES

    def _sam_size_hint_mb(self, model_idx=None):
        spec = self._sam_engine_spec(model_idx)
        if spec is None:
            return None
        sam_engine_cls = self._import_sam_engine()
        return sam_engine_cls.size_hint_mb_for_backend(spec["backend"], spec["model_type"])

    def _install_command_for_model(self, model_idx=None):
        idx = self.model_combo.currentIndex() if model_idx is None else model_idx
        if idx == MODEL_IDX_MOBILE_SAM:
            return MOBILE_SAM_INSTALL_COMMAND
        if idx == MODEL_IDX_SAM:
            return SAM_INSTALL_COMMAND
        return MOBILE_SAM_INSTALL_COMMAND

    def _sam_display_name(self, model_idx=None):
        idx = self.model_combo.currentIndex() if model_idx is None else model_idx
        if not self._is_sam_model(idx):
            return self._mode_name(idx)
        engine = self._get_or_create_sam_engine(idx)
        return getattr(engine, "display_name", self._mode_name(idx))

    def _sam_backend_available(self, model_idx=None):
        spec = self._sam_engine_spec(model_idx)
        if spec is None:
            return False
        sam_engine_cls = self._import_sam_engine()
        return sam_engine_cls.is_backend_available(spec["backend"])

    @staticmethod
    def _import_sam_engine():
        from ..core.sam_engine import SAMEngine
        return SAMEngine

    @staticmethod
    def _sam_models_dir():
        """Return a model directory that survives plugin ZIP upgrades."""
        settings_dir = QgsApplication.qgisSettingsDirPath()
        if not settings_dir:
            settings_dir = os.path.join(
                QStandardPaths.writableLocation(_standard_location("AppDataLocation")),
                "QGIS",
                "QGIS3",
                "profiles",
                "default",
            )
        return os.path.join(
            settings_dir,
            "ai_vectorizer",
            "models",
        )

    def _set_recovery_state(self, state, detail=""):
        """Render the four-state recovery contract without hiding Ink."""

        state = require_recovery_state(state)
        self._recovery_state = state
        self._recovery_detail = str(detail or "")
        if not hasattr(self, "recovery_status"):
            return
        tone_by_state = {
            RECOVERY_STATE_INK: STATUS_STYLE_INFO,
            RECOVERY_STATE_RECOVERING: STATUS_STYLE_WARNING,
            RECOVERY_STATE_ENHANCED: STATUS_STYLE_READY,
            RECOVERY_STATE_INK_FALLBACK: STATUS_STYLE_WARNING,
        }
        suffix = f" — {self._recovery_detail}" if self._recovery_detail else ""
        self.recovery_status.setText(f"{state}{suffix}")
        self.recovery_status.setStyleSheet(
            tone_by_state.get(state, STATUS_STYLE_NEUTRAL)
        )

    def _on_recovery_state_changed(self, state, detail=""):
        """Receive current-segment state from SmartTraceTool's main thread."""

        self._set_recovery_state(state, detail)

    def _load_recovery_engine_offline(self):
        """Return a prepared engine without doing file or ONNX work here."""

        if self.recovery_engine is not None:
            return self.recovery_engine, ""
        self._refresh_recovery_availability()
        if self.recovery_prepare_task is not None:
            return None, "Recovery model verification is still running."
        if self._recovery_prepare_error:
            return None, self._recovery_prepare_error
        return None, "Recovery model is not prepared; Ink remains active."

    def _cancel_recovery_prepare(self):
        task = self.recovery_prepare_task
        self.recovery_prepare_task = None
        self._recovery_prepare_generation += 1
        if task is not None:
            try:
                task.cancel()
            except RuntimeError:
                pass
        return task is not None

    def _start_recovery_prepare(self, runtime_available):
        if self.recovery_prepare_task is not None or self._shutting_down:
            return
        self._recovery_prepare_generation += 1
        generation = self._recovery_prepare_generation
        task = _RecoveryPrepareTask(
            self._sam_models_dir(),
            generation,
            runtime_available,
            self._on_recovery_prepare_finished,
        )
        self.recovery_prepare_task = task
        self.recovery_install_btn.setEnabled(False)
        self._set_recovery_state(
            RECOVERY_STATE_INK_FALLBACK,
            self._tr(
                "복구 모델 검증·준비 중; Ink만 사용",
                "Verifying and preparing recovery model; using Ink only",
            ),
        )
        try:
            QgsApplication.taskManager().addTask(task)
        except Exception as exc:
            self.recovery_prepare_task = None
            self._recovery_prepare_error = str(exc)
            self.recovery_install_btn.setEnabled(
                self.recovery_install_task is None and self.active_tool is None
            )
            self._set_recovery_state(
                RECOVERY_STATE_INK_FALLBACK,
                self._tr(
                    f"복구 준비 작업 시작 실패; Ink 유지: {exc}",
                    f"Could not start recovery preparation; Ink kept: {exc}",
                ),
            )

    def _on_recovery_prepare_finished(
        self,
        task,
        succeeded,
        status,
        engine,
        error,
    ):
        if (
            self.recovery_prepare_task is not task
            or task.generation != self._recovery_prepare_generation
        ):
            return
        self.recovery_prepare_task = None
        if self._shutting_down:
            return
        self.recovery_install_btn.setEnabled(
            self.recovery_install_task is None and self.active_tool is None
        )
        self._recovery_model_status = status
        if not succeeded and error is None:
            self._recovery_prepare_error = "Recovery model preparation was cancelled."
        else:
            self._recovery_prepare_error = "" if error is None else str(error)
        self.recovery_engine = engine if succeeded and error is None else None
        active_tool = self.active_tool
        if (
            self.recovery_engine is not None
            and active_tool is not None
            and bool(getattr(active_tool, "smart_recovery_requested", False))
        ):
            setter = getattr(active_tool, "set_recovery_engine", None)
            if callable(setter):
                ready_for_retry = bool(setter(self.recovery_engine))
                if hasattr(self, "recovery_retry_btn"):
                    self.recovery_retry_btn.setEnabled(ready_for_retry)
        self._refresh_recovery_availability()

    def _release_recovery_engine(self):
        engine = self.recovery_engine
        self.recovery_engine = None
        if engine is None:
            return
        clear_image = getattr(engine, "clear_image", None)
        if callable(clear_image):
            try:
                clear_image()
            except Exception as exc:
                self._log_nonfatal_ui_error(
                    "Failed to clear Smart Recovery image",
                    exc,
                )

    @staticmethod
    def _recovery_runtime_available():
        try:
            return importlib.util.find_spec("onnxruntime") is not None
        except (ImportError, AttributeError, ValueError):
            return False

    def _refresh_recovery_availability(self):
        if self._shutting_down or not hasattr(self, "smart_recovery_check"):
            return
        enabled = self.smart_recovery_check.isChecked()
        self.recovery_install_btn.setVisible(False)
        self.recovery_retry_btn.setVisible(enabled)
        self.recovery_runtime_guide.setVisible(False)
        self.recovery_runtime_cmd.setVisible(False)
        if not enabled:
            self._set_recovery_state(
                RECOVERY_STATE_INK,
                self._tr(
                    "기본 중심선 추적기",
                    "Primary centerline tracer",
                ),
            )
            return
        if (
            hasattr(self, "freedom_slider")
            and self.freedom_slider.value() <= 0
        ):
            self._set_recovery_state(
                RECOVERY_STATE_INK_FALLBACK,
                self._tr(
                    "0% 보조에서는 모델·증거 계산 없이 정확한 커서만 사용",
                    "At 0% assist, exact cursor input runs without model or evidence work",
                ),
            )
            return
        runtime_ready = self._recovery_runtime_available()
        if not runtime_ready:
            self.recovery_runtime_guide.setVisible(True)
            self.recovery_runtime_cmd.setVisible(True)
        if self.recovery_engine is not None and runtime_ready:
            self._set_recovery_state(
                RECOVERY_STATE_INK,
                self._tr(
                    "복구 모델 준비됨; 약한 구간에서만 실행",
                    "Recovery model ready; runs only on weak segments",
                ),
            )
            return
        if self.recovery_prepare_task is not None:
            self._set_recovery_state(
                RECOVERY_STATE_INK_FALLBACK,
                self._tr(
                    "복구 모델 검증·준비 중; Ink만 사용",
                    "Verifying and preparing recovery model; using Ink only",
                ),
            )
            return
        status = self._recovery_model_status
        if status is None and not self._recovery_prepare_error:
            self._start_recovery_prepare(runtime_ready)
            return
        if status is None:
            self.recovery_install_btn.setVisible(False)
            self._set_recovery_state(
                RECOVERY_STATE_INK_FALLBACK,
                self._tr(
                    f"모델 확인 실패; 설치를 중단하고 Ink 유지: {self._recovery_prepare_error}",
                    f"Model inspection failed; installation disabled and Ink kept: {self._recovery_prepare_error}",
                ),
            )
            return
        if status.ready and runtime_ready and not self._recovery_prepare_error:
            # A prior 0%/Freehand lifecycle may have deliberately released
            # the session while retaining the cheap inspection result.
            self._recovery_model_status = None
            self._start_recovery_prepare(runtime_ready)
            return
        corrupt = any(
            artifact.state == "corrupt" for artifact in status.artifacts
        )
        unsafe = any(
            artifact.state == "unsafe" for artifact in status.artifacts
        )
        self.recovery_install_btn.setText(
            self._tr(
                "🧰 Repair Recovery Model",
                "🧰 Repair Recovery Model",
            )
            if corrupt
            else self._tr(
                "⬇️ Install Recovery Model",
                "⬇️ Install Recovery Model",
            )
        )
        self.recovery_install_btn.setVisible(not status.ready and not unsafe)
        states = ", ".join(
            f"{artifact.spec.identifier}:{artifact.state}"
            for artifact in status.artifacts
            if not artifact.ready
        )
        self._set_recovery_state(
            RECOVERY_STATE_INK_FALLBACK,
            (
                self._tr(
                    "ONNX Runtime 없음; 아래 명령을 QGIS Python 환경에서 실행하세요. Ink만 사용합니다.",
                    "ONNX Runtime is missing; run the command below in QGIS Python. Using Ink only.",
                )
                if status.ready and not runtime_ready
                else self._tr(
                    f"복구 준비 실패; Ink 유지: {self._recovery_prepare_error}",
                    f"Recovery preparation failed; Ink kept: {self._recovery_prepare_error}",
                )
                if status.ready and self._recovery_prepare_error
                else self._tr(
                    f"안전하지 않은 모델 캐시 ({states}); 자동 변경을 거부하고 Ink만 사용",
                    f"Unsafe model cache ({states}); automatic changes disabled, using Ink only",
                )
                if unsafe
                else self._tr(
                    f"복구 모델 없음 ({states}); Ink만 사용",
                    f"Recovery model unavailable ({states}); using Ink only",
                )
            ),
        )

    def _on_smart_recovery_toggled(self, checked):
        """Opt in without ever turning model discovery into a download."""

        if not checked:
            self._cancel_recovery_prepare()
            self._release_recovery_engine()
        else:
            self._recovery_model_status = None
            self._recovery_prepare_error = ""
        if checked and self.freehand_check.isChecked():
            self.freehand_check.setChecked(False)
        if checked and self.model_combo.currentIndex() != MODEL_IDX_INK:
            self.model_combo.setCurrentIndex(MODEL_IDX_INK)
        self._refresh_recovery_availability()

    def _on_assist_strength_changed(self, value):
        """Keep literal 0% free of evidence and model preparation work."""

        self.freedom_label.setText(f"{int(value)}%")
        if int(value) <= 0:
            self._cancel_recovery_prepare()
            self._release_recovery_engine()
        self._refresh_recovery_availability()

    def _on_freehand_toggled(self, checked):
        """Keep literal Freehand and model-assisted Recovery unambiguous."""

        if checked and self.smart_recovery_check.isChecked():
            # setChecked(False) routes through the normal engine release and
            # offline availability refresh; no special lifecycle path exists.
            self.smart_recovery_check.setChecked(False)

    def install_recovery_model(self):
        """Start the sole, explicit install or corrupt-object repair action."""

        if self.recovery_install_task is not None:
            return
        status = self._recovery_model_status
        repair_corrupt = bool(
            status is not None
            and any(
                artifact.state == "corrupt"
                for artifact in status.artifacts
            )
        )
        unsafe = bool(
            status is not None
            and any(
                artifact.state == "unsafe"
                for artifact in status.artifacts
            )
        )
        if unsafe:
            self._set_recovery_state(
                RECOVERY_STATE_INK_FALLBACK,
                self._tr(
                    "안전하지 않은 모델 캐시는 자동 변경하지 않습니다. Ink를 유지합니다.",
                    "Unsafe model cache is never changed automatically. Ink remains active.",
                ),
            )
            return
        self._cancel_recovery_prepare()
        self._release_recovery_engine()
        self._recovery_model_status = None
        self._recovery_prepare_error = ""
        task = _RecoveryInstallTask(
            self._sam_models_dir(),
            self._on_recovery_install_finished,
            repair_corrupt=repair_corrupt,
        )
        self.recovery_install_task = task
        self.recovery_install_btn.setEnabled(False)
        self._set_recovery_state(
            RECOVERY_STATE_INK_FALLBACK,
            self._tr(
                "손상 모델을 격리·복구 중; 완료 전까지 Ink 유지"
                if repair_corrupt
                else "복구 모델 설치 중; 완료 전까지 Ink 유지",
                "Quarantining and repairing the corrupt model; Ink remains active"
                if repair_corrupt
                else "Installing recovery model; Ink remains active",
            ),
        )
        try:
            QgsApplication.taskManager().addTask(task)
        except Exception as exc:
            self.recovery_install_task = None
            self.recovery_install_btn.setEnabled(True)
            self._recovery_model_status = status
            self._recovery_prepare_error = ""
            self._set_recovery_state(
                RECOVERY_STATE_INK_FALLBACK,
                self._tr(
                    f"복구 설치 작업을 시작하지 못함; Ink 유지: {exc}",
                    f"Could not start recovery installation; Ink kept: {exc}",
                ),
            )
            self._refresh_recovery_availability()

    def _on_recovery_install_finished(self, task, succeeded, _bundle, error):
        if self.recovery_install_task is not task:
            return
        self.recovery_install_task = None
        if self._shutting_down:
            return
        self.recovery_install_btn.setEnabled(True)
        if succeeded and error is None:
            self._release_recovery_engine()
            self._recovery_model_status = None
            self._recovery_prepare_error = ""
            self._refresh_recovery_availability()
            return
        detail = (
            self._tr(
                "설치 취소; Ink 유지",
                "Installation cancelled; Ink kept",
            )
            if error is None
            else self._tr(
                f"설치 실패; Ink 유지: {error}",
                f"Installation failed; Ink kept: {error}",
            )
        )
        # A repair rollback restores the corrupt object by design. Reinspect
        # off the UI thread before exposing another action so the next click
        # remains Repair rather than degrading to a dead plain-install retry.
        self._recovery_model_status = None
        self._recovery_prepare_error = ""
        self.recovery_install_btn.setVisible(False)
        self._set_recovery_state(RECOVERY_STATE_INK_FALLBACK, detail)
        self._refresh_recovery_availability()

    def retry_current_segment(self):
        retry = getattr(self.active_tool, "retry_current_segment", None)
        if not callable(retry):
            self._set_recovery_state(
                RECOVERY_STATE_INK_FALLBACK,
                self._tr(
                    "먼저 Ink 트레이싱을 시작하세요",
                    "Start Ink tracing before retrying a segment",
                ),
            )
            return False
        return bool(retry())

    @classmethod
    def _configure_hed_storage(cls):
        """Keep HED assets beside SAM weights so ZIP upgrades preserve them."""
        from ..core.edge_detector import EdgeDetector

        EdgeDetector.configure_hed_storage(cls._sam_models_dir())

    def _get_or_create_sam_engine(self, model_idx=None):
        spec = self._sam_engine_spec(model_idx)
        if spec is None:
            self._release_sam_engine()
            return None

        SAMEngine = self._import_sam_engine()
        cache_key = (spec["backend"], spec["model_type"])
        if self.sam_engine is None or self.sam_engine_key != cache_key:
            self._release_sam_engine()
            self.sam_engine = SAMEngine(
                backend=spec["backend"],
                model_type=spec["model_type"],
                models_dir=self._sam_models_dir(),
            )
            self.sam_engine_key = cache_key
        return self.sam_engine

    def _release_sam_engine(self):
        engine = self.sam_engine
        self.sam_engine = None
        self.sam_engine_key = None
        if engine is not None:
            unload = getattr(engine, "unload_model", None)
            if unload is not None:
                unload()

    def _canvas_extent_in_layer_crs(self, layer):
        extent = self.iface.mapCanvas().extent()
        canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
        if canvas_crs == layer.crs():
            return extent

        transform = QgsCoordinateTransform(
            canvas_crs,
            layer.crs(),
            QgsProject.instance(),
        )
        return transform.transformBoundingBox(extent)

    def cleanup(self, permanent=False):
        if permanent:
            self._shutting_down = True
        install_task = self.recovery_install_task
        if install_task is not None:
            try:
                install_task.cancel()
            except RuntimeError:
                pass
        self._cancel_recovery_prepare()
        if (
            hasattr(self, "smart_recovery_check")
            and self.smart_recovery_check.isChecked()
        ):
            # Closing the dock returns the experimental opt-in to its safe
            # default and prevents a refresh from starting a new task.
            self.smart_recovery_check.setChecked(False)
        if permanent:
            self._preview_store.shutdown()
        else:
            self._preview_store.clear()
        if self.dem_dialog:
            try:
                self.dem_dialog.shutdown(permanent=permanent)
            except Exception as exc:
                self._log_nonfatal_ui_error("Failed to close DEM dialog", exc)
            if permanent:
                self.dem_dialog = None
        if self.active_tool:
            try:
                self.iface.mapCanvas().unsetMapTool(self.active_tool)
            except Exception as exc:
                self._log_nonfatal_ui_error("Failed to unset active tool", exc)
        self.active_tool = None
        self._release_sam_engine()
        self._release_recovery_engine()
        self._set_idle_ui()

    def closeEvent(self, event):
        self.cleanup()
        super().closeEvent(event)

    def setup_ui(self):
        self.header_label = QLabel()
        self.header_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; padding: 5px; "
            "background: #2c3e50; color: white; border-radius: 3px;"
        )
        self.layout.addWidget(self.header_label)

        lang_layout = QHBoxLayout()
        self.lang_label = QLabel()
        self.lang_combo = QComboBox()
        self.lang_combo.addItem("한국어", LANG_KO)
        self.lang_combo.addItem("English", LANG_EN)
        idx = self.lang_combo.findData(self.current_language)
        self.lang_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.lang_combo.currentIndexChanged.connect(self.on_language_changed)
        lang_layout.addWidget(self.lang_label)
        lang_layout.addWidget(self.lang_combo)
        lang_layout.addStretch()
        self.layout.addLayout(lang_layout)

        self.step1_group = QGroupBox()
        step1_layout = QVBoxLayout()
        self.step1_desc = QLabel()
        self.step1_desc.setStyleSheet("color: gray; font-size: 10px;")
        step1_layout.addWidget(self.step1_desc)
        self.layer_combo = QgsMapLayerComboBox()
        self.layer_combo.setFilters(_map_layer_filter("RasterLayer"))
        self.layer_combo.layerChanged.connect(self.on_raster_layer_selected)
        step1_layout.addWidget(self.layer_combo)
        self.step1_group.setLayout(step1_layout)
        self.layout.addWidget(self.step1_group)

        self.step2_group = QGroupBox()
        step2_layout = QVBoxLayout()
        self.step2_desc = QLabel()
        self.step2_desc.setStyleSheet("color: gray; font-size: 10px;")
        step2_layout.addWidget(self.step2_desc)

        path_layout = QHBoxLayout()
        self.shp_path = QLineEdit()
        self.browse_btn = QPushButton("📂")
        self.browse_btn.setFixedWidth(30)
        self.browse_btn.clicked.connect(self.browse_shp)
        path_layout.addWidget(self.shp_path)
        path_layout.addWidget(self.browse_btn)
        step2_layout.addLayout(path_layout)

        self.create_shp_btn = QPushButton()
        self.create_shp_btn.clicked.connect(self.create_shp_layer)
        step2_layout.addWidget(self.create_shp_btn)

        self.existing_layer_label = QLabel()
        step2_layout.addWidget(self.existing_layer_label)
        self.vector_combo = QgsMapLayerComboBox()
        self.vector_combo.setFilters(_map_layer_filter("LineLayer"))
        self.vector_combo.layerChanged.connect(self.on_layer_selected)
        step2_layout.addWidget(self.vector_combo)
        self.step2_group.setLayout(step2_layout)
        self.layout.addWidget(self.step2_group)

        self.step3_group = QGroupBox()
        step3_layout = QVBoxLayout()

        self.primary_mode_label = QLabel()
        self.primary_mode_label.setWordWrap(True)
        self.primary_mode_label.setStyleSheet("color: gray; font-size: 10px;")
        step3_layout.addWidget(self.primary_mode_label)

        self.freehand_check = QCheckBox()
        step3_layout.addWidget(self.freehand_check)

        edge_layout = QHBoxLayout()
        self.edge_strength_label = QLabel()
        edge_layout.addWidget(self.edge_strength_label)
        self.freedom_slider = QSlider(_qt_value("Horizontal", "Orientation"))
        self.freedom_slider.setMinimum(0)
        self.freedom_slider.setMaximum(100)
        self.freedom_slider.setValue(DEFAULT_FREEDOM_SLIDER_VALUE)
        edge_layout.addWidget(self.freedom_slider)
        self.freedom_label = QLabel(f"{DEFAULT_FREEDOM_SLIDER_VALUE}%")
        self.freedom_slider.valueChanged.connect(self._on_assist_strength_changed)
        edge_layout.addWidget(self.freedom_label)
        step3_layout.addLayout(edge_layout)

        self.smart_recovery_check = QCheckBox()
        self.smart_recovery_check.setChecked(False)
        self.smart_recovery_check.toggled.connect(
            self._on_smart_recovery_toggled
        )
        self.freehand_check.toggled.connect(self._on_freehand_toggled)
        step3_layout.addWidget(self.smart_recovery_check)

        self.recovery_status = QLabel()
        self.recovery_status.setWordWrap(True)
        step3_layout.addWidget(self.recovery_status)

        recovery_actions = QHBoxLayout()
        self.recovery_install_btn = QPushButton()
        self.recovery_install_btn.clicked.connect(self.install_recovery_model)
        self.recovery_install_btn.setVisible(False)
        recovery_actions.addWidget(self.recovery_install_btn)
        self.recovery_retry_btn = QPushButton()
        self.recovery_retry_btn.clicked.connect(self.retry_current_segment)
        self.recovery_retry_btn.setEnabled(False)
        self.recovery_retry_btn.setVisible(False)
        recovery_actions.addWidget(self.recovery_retry_btn)
        step3_layout.addLayout(recovery_actions)

        self.recovery_runtime_guide = QLabel()
        self.recovery_runtime_guide.setWordWrap(True)
        self.recovery_runtime_guide.setStyleSheet(
            "color: #e67e22; font-size: 9px;"
        )
        self.recovery_runtime_guide.setVisible(False)
        step3_layout.addWidget(self.recovery_runtime_guide)
        self.recovery_runtime_cmd = QLineEdit()
        self.recovery_runtime_cmd.setText(RECOVERY_RUNTIME_INSTALL_COMMAND)
        self.recovery_runtime_cmd.setReadOnly(True)
        self.recovery_runtime_cmd.setStyleSheet(
            "background: #fff3e0; font-size: 9px; padding: 3px;"
        )
        self.recovery_runtime_cmd.setVisible(False)
        step3_layout.addWidget(self.recovery_runtime_cmd)

        self.advanced_check = QCheckBox()
        self.advanced_check.setChecked(False)
        step3_layout.addWidget(self.advanced_check)

        self.advanced_group = QGroupBox()
        advanced_layout = QVBoxLayout()
        self.model_desc_label = QLabel()
        self.model_desc_label.setStyleSheet("color: gray; font-size: 10px;")
        advanced_layout.addWidget(self.model_desc_label)

        model_layout = QHBoxLayout()
        self.model_label = QLabel()
        model_layout.addWidget(self.model_label)
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_combo)
        advanced_layout.addLayout(model_layout)

        self.sam_status = QLabel("")
        self.sam_status.setStyleSheet("font-size: 10px;")
        advanced_layout.addWidget(self.sam_status)

        self.sam_check_btn = QPushButton()
        self.sam_check_btn.clicked.connect(self.check_sam_update)
        self.sam_check_btn.setVisible(False)
        advanced_layout.addWidget(self.sam_check_btn)

        self.sam_report_btn = QPushButton()
        self.sam_report_btn.clicked.connect(self.export_sam_report)
        self.sam_report_btn.setVisible(False)
        advanced_layout.addWidget(self.sam_report_btn)

        self.sam_download_btn = QPushButton()
        self.sam_download_btn.clicked.connect(self.download_sam)
        self.sam_download_btn.setVisible(False)
        advanced_layout.addWidget(self.sam_download_btn)

        self.install_guide = QLabel()
        self.install_guide.setStyleSheet("color: #e67e22; font-size: 9px;")
        self.install_guide.setVisible(False)
        advanced_layout.addWidget(self.install_guide)

        self.install_cmd = QLineEdit()
        self.install_cmd.setText(self._install_command_for_model())
        self.install_cmd.setReadOnly(True)
        self.install_cmd.setStyleSheet("background: #fff3e0; font-size: 9px; padding: 3px;")
        self.install_cmd.setVisible(False)
        advanced_layout.addWidget(self.install_cmd)

        self.auto_path_check = QCheckBox()
        self.auto_path_check.toggled.connect(self._on_auto_path_toggled)
        advanced_layout.addWidget(self.auto_path_check)

        self.advanced_group.setLayout(advanced_layout)
        self.advanced_group.setVisible(False)
        self.advanced_check.toggled.connect(self.advanced_group.setVisible)
        step3_layout.addWidget(self.advanced_group)

        self.trace_btn = QPushButton()
        self.trace_btn.setCheckable(True)
        self.trace_btn.clicked.connect(self.toggle_trace_tool)
        self.trace_btn.setStyleSheet(TRACE_BUTTON_IDLE_STYLE)
        self.trace_btn.setEnabled(False)
        step3_layout.addWidget(self.trace_btn)

        self.step3_group.setLayout(step3_layout)
        self.layout.addWidget(self.step3_group)

        self.step4_group = QGroupBox()
        step4_layout = QVBoxLayout()
        self.step4_desc = QLabel()
        self.step4_desc.setWordWrap(True)
        self.step4_desc.setStyleSheet("color: gray; font-size: 10px;")
        step4_layout.addWidget(self.step4_desc)
        self.dem_btn = QPushButton()
        self.dem_btn.setEnabled(False)
        self.dem_btn.clicked.connect(self.open_dem_dialog)
        self.trace_btn.toggled.connect(self._update_dem_button_for_tracing)
        step4_layout.addWidget(self.dem_btn)
        self.step4_group.setLayout(step4_layout)
        self.layout.addWidget(self.step4_group)

        self.status_box = QGroupBox()
        status_layout = QVBoxLayout()
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        self.controls_title_label = QLabel()
        self.controls_title_label.setStyleSheet("font-weight: bold; color: #333; margin-top: 5px;")
        status_layout.addWidget(self.controls_title_label)
        self.controls_label = QLabel()
        self.controls_label.setStyleSheet(
            "color: #555; font-size: 9px; background: #f8f9fa; "
            "padding: 8px; border-radius: 4px; line-height: 1.4;"
        )
        status_layout.addWidget(self.controls_label)
        self.status_box.setLayout(status_layout)
        self.layout.addWidget(self.status_box)

        self.debug_box = QGroupBox()
        debug_layout = QVBoxLayout()
        self.preview_edge_btn = QPushButton()
        self.preview_edge_btn.clicked.connect(self.preview_edges)
        debug_layout.addWidget(self.preview_edge_btn)
        self.help_btn = QPushButton()
        self.help_btn.clicked.connect(self.show_help)
        debug_layout.addWidget(self.help_btn)
        self.debug_box.setLayout(debug_layout)
        self.layout.addWidget(self.debug_box)

        self.layout.addStretch()

        self.apply_language()
        self.on_model_changed(self.model_combo.currentIndex())
        self._refresh_recovery_availability()
        self.on_layer_selected(self.vector_combo.currentLayer())

    def apply_language(self):
        current_idx = self.model_combo.currentIndex()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(self._model_items())
        self.model_combo.setCurrentIndex(max(0, min(current_idx, self.model_combo.count() - 1)))
        self.model_combo.blockSignals(False)

        self.setWindowTitle(PLUGIN_NAME)
        self.header_label.setText(
            self._tr(
                f"🏛️ {PLUGIN_NAME} - 고지도 등고선 벡터화",
                f"🏛️ {PLUGIN_NAME} - Historical Map Contour Vectorization",
            )
        )
        self.lang_label.setText(self._tr("언어:", "Language:"))

        self.step1_group.setTitle(self._tr("1️⃣ 입력 지도", "1️⃣ Input Map"))
        self.step1_group.setToolTip(self._tr("벡터화할 래스터 지도를 선택하세요", "Select a raster map to vectorize"))
        self.step1_desc.setText(self._tr("💡 등고선이 있는 스캔 지도 선택", "💡 Select a scanned map with contours"))
        self.layer_combo.setToolTip(self._tr("QGIS에 로드된 래스터 레이어 중 선택", "Choose from raster layers loaded in QGIS"))

        self.step2_group.setTitle(self._tr("2️⃣ 출력 파일", "2️⃣ Output File"))
        self.step2_group.setToolTip(self._tr("등고선을 저장할 Shapefile 생성 또는 선택", "Create or select a Shapefile for output"))
        self.step2_desc.setText(self._tr("💡 새 SHP 생성 또는 기존 레이어 선택", "💡 Create a new SHP or select an existing line layer"))
        self.shp_path.setPlaceholderText(self._tr("저장할 SHP 파일 경로...", "Output SHP file path..."))
        self.browse_btn.setToolTip(self._tr("파일 위치 찾기", "Browse file location"))
        self.create_shp_btn.setText(self._tr("📁 새 SHP 생성", "📁 Create New SHP"))
        self.create_shp_btn.setToolTip(self._tr("지정한 경로에 새 Shapefile을 생성합니다", "Create a new Shapefile at the selected path"))
        self.existing_layer_label.setText(self._tr("또는 기존 라인 레이어:", "Or existing line layer:"))
        self.vector_combo.setToolTip(self._tr("이미 있는 라인 레이어에 추가", "Append to an existing line layer"))

        self.step3_group.setTitle(self._tr("3️⃣ 트레이싱 설정", "3️⃣ Tracing Options"))
        self.step3_group.setToolTip(self._tr("Ink 중심선과 선택적 복구 설정", "Ink centerline and optional recovery settings"))
        self.primary_mode_label.setText(
            self._tr(
                "🖋 기본은 Ink Centerline입니다. 모델 없이 항상 먼저 실행됩니다.",
                "🖋 Ink Centerline is the default and always runs first without a model.",
            )
        )
        self.smart_recovery_check.setText(
            self._tr(
                "🛟 Smart Recovery (실험적, 기본 꺼짐)",
                "🛟 Smart Recovery (Experimental, default OFF)",
            )
        )
        self.smart_recovery_check.setToolTip(
            self._tr(
                "Ink 증거가 약한 현재 구간만 로컬 EfficientSAM으로 재검토합니다. 모델 결과가 안전하게 개선될 때만 초록 미리보기를 바꿉니다.",
                "Re-check only weak current Ink segments with local EfficientSAM. The green preview changes only when the challenger is a safe improvement.",
            )
        )
        self.recovery_install_btn.setText(
            self._tr(
                "⬇️ Install Recovery Model",
                "⬇️ Install Recovery Model",
            )
        )
        self.recovery_install_btn.setToolTip(
            self._tr(
                "명시적으로 누를 때만 고정 URL에서 모델을 다운로드하고 SHA-256을 검증합니다.",
                "Downloads from the pinned URL and verifies SHA-256 only after this explicit click.",
            )
        )
        self.recovery_retry_btn.setText(
            self._tr(
                "↻ Retry current segment",
                "↻ Retry current segment",
            )
        )
        self.recovery_retry_btn.setToolTip(
            self._tr(
                "확정되지 않은 현재 Ink 구간에 복구를 한 번 다시 요청합니다.",
                "Explicitly retry recovery for the current uncommitted Ink segment.",
            )
        )
        self.recovery_runtime_guide.setText(
            self._tr(
                "📦 Recovery Runtime 설치 (자동 실행되지 않음, 복사 가능):",
                "📦 Install Recovery Runtime (never run automatically; copy this):",
            )
        )
        self.recovery_runtime_cmd.setText(RECOVERY_RUNTIME_INSTALL_COMMAND)
        self.advanced_check.setText(
            self._tr(
                "▸ Advanced: 기존 추적 모델",
                "▸ Advanced: legacy tracing models",
            )
        )
        self.advanced_group.setTitle(self._tr("고급 모델 선택", "Advanced model selection"))
        self.model_desc_label.setText(self._tr("기존 모델 인덱스 0–5 호환 영역", "Compatibility area preserving legacy model indices 0–5"))
        self.model_label.setText(self._tr("AI 모델:", "AI Model:"))
        self.model_label.setToolTip(
            self._tr(
                (
                    "각 추적 방식의 역할:\n"
                    "• Ink Centerline: 다중 스케일 검은색·유색 선 증거 + 방향 인식 Live-Wire\n"
                    "• LSD: 선분 지도를 이용한 Live-Wire\n"
                    f"• HED: 학습된 엣지 지도 (~{self._hed_size_hint_mb()}MB)\n"
                    f"• MobileSAM: 프롬프트 마스크 + edge/A* (~{self._sam_size_hint_mb(MODEL_IDX_MOBILE_SAM)}MB)\n"
                    f"• SAM: ViT-B 프롬프트 마스크 + edge/A* (~{self._sam_size_hint_mb(MODEL_IDX_SAM)}MB)\n"
                    "• Legacy Canny: 기존 경계 검출 호환 모드"
                ),
                (
                    "Tracing-mode roles:\n"
                    "• Ink Centerline: multi-scale dark/coloured line evidence + direction-aware Live-Wire\n"
                    "• LSD: Live-Wire over line-segment evidence\n"
                    f"• HED: learned edge map (~{self._hed_size_hint_mb()}MB)\n"
                    f"• MobileSAM: prompt mask + edge/A* (~{self._sam_size_hint_mb(MODEL_IDX_MOBILE_SAM)}MB)\n"
                    f"• SAM: ViT-B prompt mask + edge/A* (~{self._sam_size_hint_mb(MODEL_IDX_SAM)}MB)\n"
                    "• Legacy Canny: compatibility edge detector"
                ),
            )
        )
        self.model_combo.setToolTip(
            self._tr(
                "Ink Centerline: 다중 스케일 검은색·유색 선 증거\nLSD: 선분 보조\nHED: 학습된 엣지 지도\nMobileSAM: 프롬프트 마스크\nSAM: ViT-B 프롬프트 마스크\nLegacy Canny: 기존 경계 검출",
                "Ink Centerline: multi-scale dark/coloured line evidence\n"
                "LSD: line-segment assistance\n"
                "HED: learned edge map\n"
                "MobileSAM: prompt mask\n"
                "SAM: ViT-B prompt mask\n"
                "Legacy Canny: compatibility edge detector",
            )
        )
        self.sam_check_btn.setText(self._tr("🔎 선택 SAM 모델 검증", "🔎 Verify Selected SAM Model"))
        self.sam_check_btn.setToolTip(
            self._tr(
                "현재 선택된 SAM 계열 모델을 고정된 크기와 SHA-256 정체성으로 검증합니다",
                "Verify the selected SAM-family model against its pinned size and SHA-256 identity",
            )
        )
        self.sam_report_btn.setText(self._tr("📄 SAM 상태 리포트", "📄 SAM Status Report"))
        self.sam_report_btn.setToolTip(
            self._tr(
                "선택 모델의 고정된 파일 정체성을 검증한 뒤\nSAM 환경/버전/모델 상태를 JSON으로 저장하고 클립보드에 복사합니다",
                "Verify the selected model's pinned file identity,\nthen export SAM environment/version/model status as JSON and copy it to the clipboard",
            )
        )
        self.sam_download_btn.setToolTip(self._tr("인터넷 연결 필요. 최초 1회만 다운로드", "Internet required. Download once on first use"))
        self.install_guide.setText(self._tr("📦 선택 모델 설치 (복사 가능):", "📦 Selected Model Install (copy this):"))
        self.install_cmd.setText(self._install_command_for_model())
        self.freehand_check.setText(self._tr("✏️ 프리핸드 (AI 비활성)", "✏️ Freehand (AI Off)"))
        self.freehand_check.setToolTip(self._tr("체크: AI 없이 순수 마우스 추적", "Checked: pure mouse tracing without AI"))
        self.edge_strength_label.setText(self._tr("AI 개입 강도:", "AI Assist Strength:"))
        self.freedom_slider.setToolTip(
            self._tr(
                "0%: 커서 그대로\n1~99%: 커서 경로와 Live-Wire를 실제 비율로 혼합\n100%: 방향 인식 경로 전체 적용\nSAM은 Auto Path에서만 사용",
                "0%: exact cursor\n1-99%: literal geometry blend between cursor and Live-Wire\n100%: full direction-aware route\nSAM is used only in Auto Path",
            )
        )
        if self.trace_btn.isChecked():
            self._set_trace_button_active()
        else:
            self._set_trace_button_idle()
        self.trace_btn.setToolTip(self._tr("클릭하여 트레이싱 시작", "Click to start tracing"))

        self.step4_group.setTitle(self._tr("4️⃣ 지형 복원", "4️⃣ Terrain Reconstruction"))
        self.step4_group.setToolTip(
            self._tr(
                "고도값을 가진 등고선에서 DEM과 hillshade를 생성합니다",
                "Build a DEM and hillshade from elevated contours",
            )
        )
        self.step4_desc.setText(
            self._tr(
                "💡 투영 좌표계(m)의 등고선과 고도 필드가 필요합니다",
                "💡 Requires contours with elevations in a projected CRS (metres)",
            )
        )
        self.dem_btn.setText(self._tr("⛰️ DEM 생성…", "⛰️ Build DEM…"))
        self.dem_btn.setToolTip(
            self._tr(
                "선형 TIN DEM과 GDAL hillshade 생성 대화상자 열기",
                "Open the linear-TIN DEM and GDAL hillshade builder",
            )
        )

        self.status_box.setTitle(self._tr("📋 상태", "📋 Status"))
        self.status_label.setToolTip(self._tr("현재 트레이싱 상태를 표시합니다", "Shows current tracing state"))
        self.controls_title_label.setText(self._tr("📖 사용법:", "📖 Controls:"))
        self.controls_label.setText(
            self._tr(
                "• 드래그: 선 그리기 / 클릭: 체크포인트\n"
                "• 그리는 중 Ctrl+Z: 마지막 체크포인트로 되돌리기\n"
                "• 저장 후 Ctrl+Z: QGIS 편집 작업 되돌리기\n"
                "• Esc: 현재 그리기 취소 / Del: 전체 취소\n"
                "• 시작점 클릭: 폴리곤 닫기 → 해발값\n"
                "• 우클릭/Enter: 저장",
                "• Drag: draw line / Click: checkpoint\n"
                "• While tracing Ctrl+Z: undo to last checkpoint\n"
                "• After save Ctrl+Z: undo the QGIS edit command\n"
                "• Esc: cancel current trace / Del: cancel all\n"
                "• Click start point: close polygon -> elevation\n"
                "• Right click / Enter: save",
            )
        )
        self.controls_label.setToolTip(
            self._tr(
                "클릭으로 체크포인트 저장\n그리는 중에는 체크포인트, 저장 후에는 QGIS 작업을 Ctrl+Z로 되돌립니다",
                "Click to place checkpoints\nCtrl+Z undoes a checkpoint while tracing and the QGIS edit after saving",
            )
        )

        self.debug_box.setTitle(self._tr("🔧 디버그 및 도움말", "🔧 Debug & Help"))
        self.debug_box.setToolTip(self._tr("문제 해결을 위한 도구들", "Tools for troubleshooting"))
        self.preview_edge_btn.setText(self._tr("👁️ 추적 미리보기", "👁️ Trace Preview"))
        self.preview_edge_btn.setToolTip(
            self._tr(
                "Ink/LSD/HED/Legacy Canny 모드에서는 감지된 중심선·엣지를\n임시 래스터 레이어로 표시합니다.\nSAM 모드에서는 지도에서 대화형 초록색 미리보기를 사용합니다.",
                "Ink/LSD/HED/Legacy Canny modes show detected line evidence\nas a temporary raster layer.\nSAM modes use the interactive green preview on the map.",
            )
        )
        self.help_btn.setText(self._tr("❓ 도움말", "❓ Help"))
        self.help_btn.setToolTip(self._tr("사용법과 문제해결 안내", "Usage guide and troubleshooting"))

        self.auto_path_check.setText("AI Proposal / Auto Path (Experimental)")
        self.auto_path_check.setToolTip(
            self._tr(
                "Ink/LSD/HED/Legacy Canny에서는 커서를 따라 표시되는 전체 Live-Wire 경로를 클릭 한 번으로 채택합니다. SAM은 커서를 잠시 멈춘 뒤 표시되며 같은 위치를 다시 클릭해 채택합니다.",
                "Ink/LSD/HED/Legacy Canny show the full Live-Wire route and accept it with one click. SAM appears after a short pause and is accepted by clicking the same target again.",
            )
        )
        self.sam_download_btn.setText(self._download_button_text())

        if not self.trace_btn.isEnabled():
            self._set_status_label(self._tr("SHP 파일을 먼저 생성하세요", "Create or select an SHP layer first"))
        elif self.trace_btn.isChecked():
            self._set_status_label(
                self._tr(
                    "🖊️ [{mode}] 등고선을 클릭하세요",
                    "🖊️ [{mode}] Click on contours",
                ).format(
                    mode=self._mode_name(self.model_combo.currentIndex()),
                )
            )
        else:
            self._set_ready_state()
        self._refresh_recovery_availability()

    def on_language_changed(self, _index):
        selected = self.lang_combo.currentData()
        if selected not in (LANG_KO, LANG_EN):
            return
        self.current_language = selected
        self._save_language()
        self.apply_language()
        self.on_model_changed(self.model_combo.currentIndex())
        if self.active_tool:
            self.active_tool.language = self.current_language
        if self.dem_dialog:
            self.dem_dialog.set_inputs(
                self.vector_combo.currentLayer(),
                self.layer_combo.currentLayer(),
                self.current_language,
            )

    def browse_shp(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            self._tr("SHP 파일 저장 위치", "Save SHP File"),
            "",
            "Shapefile (*.shp)",
        )
        if path:
            if not path.endswith(".shp"):
                path += ".shp"
            self.shp_path.setText(path)

    def create_shp_layer(self):
        path = self.shp_path.text().strip()
        if not path:
            QMessageBox.warning(self, self._tr("경고", "Warning"), self._tr("파일 경로를 지정해주세요.", "Please specify an output file path."))
            return

        loaded_layers = _project_layers_using_path(path)
        if loaded_layers:
            layer_names = "\n".join(
                f"• {layer.name()}" for layer in loaded_layers
            )
            QMessageBox.warning(
                self,
                self._tr("출력 레이어 사용 중", "Output Layer Is Loaded"),
                self._tr(
                    "로드된 Shapefile을 덮어쓰면 저장하지 않은 편집이 "
                    "손실될 수 있습니다. 먼저 다음 레이어를 QGIS에서 제거하세요:\n{layers}",
                    "Overwriting a loaded Shapefile can lose uncommitted edits. "
                    "Remove these layers from QGIS first:\n{layers}",
                ).format(layers=layer_names),
            )
            return

        if os.path.exists(path):
            answer = QMessageBox.question(
                self,
                self._tr("기존 파일 덮어쓰기", "Overwrite Existing File"),
                self._tr(
                    "기존 Shapefile을 덮어쓸까요?\n{path}",
                    "Overwrite the existing Shapefile?\n{path}",
                ).format(path=path),
                _message_box_button("Yes") | _message_box_button("No"),
                _message_box_button("No"),
            )
            if answer != _message_box_button("Yes"):
                return

        raster = self.layer_combo.currentLayer()
        crs = raster.crs() if raster else QgsCoordinateReferenceSystem(DEFAULT_CRS_AUTHID)
        fields = [
            QgsField(FIELD_ID, _field_type("Int")),
            QgsField(FIELD_ELEVATION, _field_type("Double")),
        ]

        layer = QgsVectorLayer(f"LineString?crs={crs.authid()}", DEFAULT_OUTPUT_LAYER_NAME, "memory")
        layer.dataProvider().addAttributes(fields)
        layer.updateFields()
        error = _write_vector_layer(layer, path, crs)

        if error[0] == _writer_no_error():
            name = os.path.basename(path).replace(".shp", "")
            self.output_layer = QgsVectorLayer(path, name, "ogr")
            symbol = QgsSymbol.defaultSymbol(self.output_layer.geometryType())
            symbol.setColor(QColor(255, 0, 0))
            symbol.setWidth(1.2)
            self.output_layer.setRenderer(QgsSingleSymbolRenderer(symbol))
            QgsProject.instance().addMapLayer(self.output_layer)
            self.vector_combo.setLayer(self.output_layer)
            self.enable_tracing()
            QMessageBox.information(
                self,
                self._tr("성공", "Success"),
                self._tr("SHP 생성 완료:\n{path}", "SHP created successfully:\n{path}").format(path=path),
            )
        else:
            QMessageBox.critical(
                self,
                self._tr("오류", "Error"),
                self._tr("생성 실패: {error}", "Creation failed: {error}").format(error=error[1]),
            )

    def on_layer_selected(self, layer):
        if self.active_tool is not None:
            active_layer = getattr(self.active_tool, "vector_layer", None)
            if self._same_layer(layer, active_layer):
                return
            self.iface.mapCanvas().unsetMapTool(self.active_tool)
        self.output_layer = layer
        self.trace_btn.setEnabled(bool(layer))
        self._update_dem_button_for_tracing(self.trace_btn.isChecked())
        if layer:
            self.enable_tracing()
        elif not self.trace_btn.isChecked():
            self._set_status_label(self._tr("SHP 파일을 먼저 생성하세요", "Create or select an SHP layer first"))

    def on_raster_layer_selected(self, layer):
        """Stop a session if its raster is removed or replaced externally."""

        if self.active_tool is None:
            return
        active_layer = getattr(self.active_tool, "raster_layer", None)
        if not self._same_layer(layer, active_layer):
            self.iface.mapCanvas().unsetMapTool(self.active_tool)

    def enable_tracing(self):
        self.trace_btn.setEnabled(True)
        self._update_dem_button_for_tracing(self.trace_btn.isChecked())
        self._set_ready_state(prompt=True)

    def _update_dem_button_for_tracing(self, tracing):
        self.dem_btn.setEnabled(
            self.vector_combo.currentLayer() is not None and not tracing
        )

    def open_dem_dialog(self):
        contour_layer = self.vector_combo.currentLayer()
        if contour_layer is None:
            QMessageBox.warning(
                self,
                self._tr("등고선 레이어 필요", "Contour Layer Required"),
                self._tr(
                    "고도값을 가진 라인 레이어를 먼저 선택하세요.",
                    "Select a line layer with elevation values first.",
                ),
            )
            return
        if self.active_tool or self.trace_btn.isChecked():
            QMessageBox.warning(
                self,
                self._tr("트레이싱 진행 중", "Tracing In Progress"),
                self._tr(
                    "현재 등고선을 먼저 저장하거나 취소한 뒤 DEM을 생성하세요.",
                    "Save or cancel the current contour before building a DEM.",
                ),
            )
            return

        if self.dem_dialog is None:
            from .dem_dialog import DemBuildDialog

            self.dem_dialog = DemBuildDialog(
                self.iface,
                contour_layer=contour_layer,
                raster_layer=self.layer_combo.currentLayer(),
                language=self.current_language,
                parent=self.iface.mainWindow(),
            )
        else:
            self.dem_dialog.set_inputs(
                contour_layer,
                self.layer_combo.currentLayer(),
                self.current_language,
            )
        self.dem_dialog.show()
        self.dem_dialog.raise_()
        self.dem_dialog.activateWindow()

    def toggle_trace_tool(self, checked):
        if checked:
            raster = self.layer_combo.currentLayer()
            if not raster:
                QMessageBox.warning(self, self._tr("경고", "Warning"), self._tr("래스터 지도를 선택하세요.", "Please select a raster map."))
                self.trace_btn.setChecked(False)
                return

            output_layer = self.vector_combo.currentLayer()
            if output_layer is None:
                QMessageBox.warning(
                    self,
                    self._tr("경고", "Warning"),
                    self._tr("출력 라인 레이어를 선택하세요.", "Select an output line layer."),
                )
                self.trace_btn.setChecked(False)
                return
            self.output_layer = output_layer

            from ..tools.smart_trace_tool import SmartTraceTool

            unsupported_reason = SmartTraceTool.unsupported_output_reason(output_layer)
            if unsupported_reason:
                QMessageBox.warning(
                    self,
                    self._tr("지원되지 않는 출력 레이어", "Unsupported Output Layer"),
                    self._tr(
                        "ArchaeoTrace는 현재 2D 라인 레이어만 안전하게 편집합니다. "
                        "Z/M 값은 고도 필드로 변환한 2D 레이어를 선택하세요.",
                        "ArchaeoTrace currently edits only 2D line layers safely. "
                        "Choose a 2D layer with elevation stored in a field instead of Z/M values.",
                    ),
                )
                self.trace_btn.setChecked(False)
                return
            if output_layer.readOnly():
                QMessageBox.warning(
                    self,
                    self._tr("읽기 전용 출력", "Read-only Output"),
                    self._tr(
                        "선택한 출력 레이어는 읽기 전용입니다.",
                        "The selected output layer is read-only.",
                    ),
                )
                self.trace_btn.setChecked(False)
                return

            edge_weight = self.freedom_slider.value() / 100.0
            freehand = self.freehand_check.isChecked()
            auto_path = self.auto_path_check.isChecked() and not freehand
            model_idx = self.model_combo.currentIndex()
            smart_recovery = (
                self.smart_recovery_check.isChecked()
                and model_idx == MODEL_IDX_INK
                and not freehand
                and edge_weight > 0.0
            )
            recovery_engine = None
            if smart_recovery:
                recovery_engine, recovery_error = (
                    self._load_recovery_engine_offline()
                )
                if recovery_engine is None:
                    # Missing dependencies, absent/corrupt files, and load
                    # errors are never blockers. Ink starts unchanged and no
                    # download is attempted from this path.
                    self._set_recovery_state(
                        RECOVERY_STATE_INK_FALLBACK,
                        self._tr(
                            f"복구를 시작할 수 없어 Ink 유지: {recovery_error}",
                            f"Recovery could not start; Ink kept: {recovery_error}",
                        ),
                    )
            else:
                # Freehand, legacy models, and literal 0% tracing must not
                # retain optional ONNX sessions from an earlier run.
                self._cancel_recovery_prepare()
                self._release_recovery_engine()
            if self._is_sam_model(model_idx) and auto_path and edge_weight > 0.0:
                self.init_sam_engine()
            else:
                # Do not just drop the reference: a previously loaded SAM
                # predictor can keep hundreds of MB alive after switching
                # back to the default human-led mode or Freehand.
                self._release_sam_engine()
            use_sam = (
                not freehand
                and auto_path
                and edge_weight > 0.0
                and self._is_sam_model(model_idx)
                and self.sam_engine is not None
                and self.sam_engine.is_ready
            )
            if edge_weight <= 0.0:
                edge_method = DEFAULT_EDGE_METHOD
            else:
                edge_method = (
                    SAM_ASSIST_EDGE_METHOD
                    if use_sam
                    else EDGE_METHOD_BY_MODEL.get(model_idx, DEFAULT_EDGE_METHOD)
                )

            needs_cv2 = (
                not freehand
                and edge_weight > 0.0
                and (
                    model_idx in (MODEL_IDX_LSD, MODEL_IDX_HED)
                    or (
                        auto_path
                        and edge_weight > 0.0
                        and self._is_sam_model(model_idx)
                    )
                )
            )
            if needs_cv2 and not is_cv2_available():
                self._show_opencv_warning(self._tr("AI 트레이싱", "AI tracing"))
                self.trace_btn.setChecked(False)
                return

            if not freehand and edge_weight > 0.0 and model_idx == MODEL_IDX_HED:
                from ..core.edge_detector import EdgeDetector
                hed_status = EdgeDetector.get_hed_runtime_status(force_refresh=True)
                if not hed_status.get("ok"):
                    QMessageBox.warning(
                        self,
                        self._tr("경고", "Warning"),
                        self._tr(
                            "HED 모델이 아직 준비되지 않았습니다.\n{detail}\n먼저 다운로드/복구한 뒤 다시 시도하세요.",
                            "HED is not ready yet.\n{detail}\nDownload or repair it, then try again.",
                        ).format(detail=hed_status.get("message", "")),
                    )
                    self.trace_btn.setChecked(False)
                    return

            if (
                auto_path
                and edge_weight > 0.0
                and self._is_sam_model(model_idx)
                and not use_sam
            ):
                QMessageBox.warning(
                    self,
                    self._tr("경고", "Warning"),
                    self._tr(
                        "{name}이 아직 준비되지 않았습니다. 설치/다운로드 후 다시 시도하세요.",
                        "{name} is not ready yet. Install/download it and try again.",
                    ).format(
                        name=self._sam_display_name(model_idx),
                    ),
                )
                self.trace_btn.setChecked(False)
                return

            tool = None
            try:
                tool = SmartTraceTool(
                    self.iface.mapCanvas(),
                    raster,
                    output_layer,
                    model_type=model_idx,
                    edge_weight=edge_weight,
                    freehand=freehand,
                    sam_engine=self.sam_engine if use_sam else None,
                    edge_method=edge_method,
                    iface=self.iface,
                    language=self.current_language,
                    auto_path=auto_path,
                    recovery_engine=recovery_engine,
                    smart_recovery=smart_recovery,
                    recovery_state_callback=self._on_recovery_state_changed,
                )
                tool.deactivated.connect(self.on_tool_deactivated)
                self.active_tool = tool
                self.iface.mapCanvas().setMapTool(tool)
            except Exception as exc:
                if tool is not None:
                    try:
                        self.iface.mapCanvas().unsetMapTool(tool)
                    # Retain the original tool setup failure.
                    except Exception:  # nosec B110
                        pass
                    # QgsMapTool is parented to the canvas. Dropping this
                    # Python reference alone would retain the failed tool,
                    # its rubber bands, and its timers until QGIS exits.
                    tool.dispose()
                    tool.deleteLater()
                self.active_tool = None
                QMessageBox.critical(
                    self,
                    self._tr("트레이싱 시작 실패", "Could Not Start Tracing"),
                    str(exc),
                )
                self.trace_btn.setChecked(False)
                return

            self._set_trace_configuration_enabled(False)

            if not freehand and edge_weight <= 0.0:
                mode_name = self._tr("정확한 커서 (AI 0%)", "Exact Cursor (AI 0%)")
            elif freehand:
                mode_name = self._tr("프리핸드", "Freehand")
            elif smart_recovery and recovery_engine is not None:
                mode_name = self._tr(
                    "Ink + Smart Recovery",
                    "Ink + Smart Recovery",
                )
            elif self._is_sam_model(model_idx) and not use_sam:
                mode_name = self._tr(
                    "사람 주도 보조 (Ink Centerline)",
                    "Human-led Assist (Ink Centerline)",
                )
            else:
                mode_name = self._sam_display_name(model_idx) if use_sam else self._mode_name(model_idx)
            self._set_tracing_state(mode_name)
        else:
            if self.active_tool:
                self.iface.mapCanvas().unsetMapTool(self.active_tool)
            self._set_idle_ui()

    def on_tool_deactivated(self):
        tool = self.sender()
        if (
            tool is not None
            and self.active_tool is not None
            and tool is not self.active_tool
        ):
            # Do not let a delayed signal from an older tool reset a newer
            # session, but still release the canvas-owned stale tool.
            tool.dispose()
            tool.deleteLater()
            return
        self.active_tool = None
        self._set_idle_ui()
        if tool is not None:
            # QgsMapTool's QObject parent is the canvas, so it otherwise
            # survives every start/stop cycle until the canvas is destroyed.
            tool.dispose()
            tool.deleteLater()

    def on_model_changed(self, index):
        self._set_model_aux_visibility()
        self._release_sam_engine()
        if (
            hasattr(self, "smart_recovery_check")
            and index != MODEL_IDX_INK
            and self.smart_recovery_check.isChecked()
        ):
            self.smart_recovery_check.setChecked(False)
        if index == MODEL_IDX_INK:
            from ..core.edge_detector import EdgeDetector

            ink_status = EdgeDetector.get_ink_runtime_status()
            background = ink_status.get("background_backend", "unknown")
            thinning = ink_status.get("thinning_backend", "unknown")
            backend_text = f"{background} / {thinning}"
            if not is_livewire_available():
                self._set_sam_status(
                    self._tr(
                        "✅ Ink 중심선 준비됨 ({backend}); Live-Wire 없이 국소 보조 사용",
                        "✅ Ink centerline ready ({backend}); local assist is used without Live-Wire",
                    ).format(backend=backend_text),
                    "warning",
                )
                return
            self._set_sam_status(
                self._tr(
                    "✅ Ink 중심선 ({backend}) + 방향 인식 Live-Wire 준비됨",
                    "✅ Ink centerline ({backend}) + direction-aware Live-Wire ready",
                ).format(backend=backend_text),
                "info",
            )
            self.sam_status.setToolTip("")
            self._refresh_recovery_availability()
            return
        if index == MODEL_IDX_LEGACY_CANNY:
            if not is_livewire_available():
                self._set_sam_status(
                    self._tr(
                        "⚠️ Legacy Canny는 사용 가능하며 Live-Wire 대신 국소 보조를 사용합니다",
                        "⚠️ Legacy Canny is available with local assist instead of Live-Wire",
                    ),
                    "warning",
                )
                return
            self._set_sam_status(
                self._tr(
                    "✅ Legacy Canny + 방향 인식 Live-Wire 준비됨",
                    "✅ Legacy Canny + direction-aware Live-Wire ready",
                ),
                "info",
            )
            self.sam_status.setToolTip("")
            return
        if index == MODEL_IDX_LSD:
            if is_cv2_available():
                self._set_sam_status(self._tr("✅ OpenCV 로드됨", "✅ OpenCV loaded"), "info")
                self.sam_status.setToolTip("")
            else:
                self._set_sam_status(self._tr("❌ OpenCV 미설치", "❌ OpenCV not installed"), "error")
                self.sam_status.setToolTip(get_cv2_error_text())
                self._set_model_aux_visibility(show_install=True)
                self._set_install_hint(
                    self._tr("📦 OpenCV 설치 (복사 가능):", "📦 Install OpenCV (copy this):"),
                    get_opencv_install_command(),
                )
        elif index == MODEL_IDX_HED:
            self.check_hed_status()
        elif self._is_sam_model(index):
            if self.auto_path_check.isChecked():
                self._set_model_aux_visibility(show_check=True, show_report=True, show_download=True)
                self.sam_download_btn.setText(self._download_button_text(index))
                self.install_cmd.setText(self._install_command_for_model(index))
                self._set_sam_status(
                    "SAM is enabled only for explicit Auto Path mode.",
                    "info",
                )
            else:
                self._set_model_aux_visibility()
                self._set_sam_status(
                    "Human-led Ink Centerline assist is active; SAM installation is not needed.",
                    "info",
                )

    def _on_auto_path_toggled(self, _checked):
        """Refresh model guidance when the expensive route mode is toggled."""
        if hasattr(self, "model_combo"):
            self.on_model_changed(self.model_combo.currentIndex())

    def check_hed_status(self):
        from ..core.edge_detector import EdgeDetector
        status = EdgeDetector.get_hed_runtime_status()
        self.sam_status.setToolTip(status.get("message", ""))
        if status.get("ok"):
            self._set_sam_status(self._tr("✅ HED 모델 로드됨", "✅ HED model loaded"), "info")
        else:
            if status.get("reason") == "missing_opencv":
                self._set_sam_status(
                    self._tr("❌ OpenCV 미설치", "❌ OpenCV not installed"),
                    "error",
                )
                self._set_model_aux_visibility(show_install=True)
                self._set_install_hint(
                    self._tr("📦 OpenCV 설치 (복사 가능):", "📦 Install OpenCV (copy this):"),
                    get_opencv_install_command(),
                )
            elif status.get("reason") in ("missing_prototxt", "missing_weights"):
                self._set_sam_status(
                    self._tr(
                        f"⚠️ HED 모델 필요 ({self._hed_size_hint_mb()}MB)",
                        f"⚠️ HED model required (~{self._hed_size_hint_mb()}MB)",
                    ),
                    "warning",
                )
                self._set_model_aux_visibility(show_download=True)
                self.sam_download_btn.setText(self._download_button_text(MODEL_IDX_HED))
            else:
                self._set_sam_status(
                    self._tr(
                        "❌ HED 모델 손상 또는 로드 실패 - 다시 다운로드하세요",
                        "❌ HED model is invalid or failed to load - re-download it",
                    ),
                    "error",
                )
                self._set_model_aux_visibility(show_download=True)
                self.sam_download_btn.setText(self._download_button_text(MODEL_IDX_HED))

    def init_sam_engine(self):
        model_idx = self.model_combo.currentIndex()
        self._get_or_create_sam_engine(model_idx)
        self._set_model_aux_visibility(show_check=True, show_report=True, show_download=True)
        self.sam_download_btn.setText(self._download_button_text(model_idx))
        self.install_cmd.setText(self._install_command_for_model(model_idx))

        if not self._sam_backend_available(model_idx):
            self._set_sam_status(
                self._tr(
                    "❌ {name} 미설치",
                    "❌ {name} not installed",
                ).format(name=self._sam_display_name(model_idx)),
                "error",
            )
            self._set_model_aux_visibility(show_check=True, show_report=True, show_download=True, show_install=True)
            return

        success, load_msg = self.sam_engine.load_model()
        if success:
            if is_cv2_available():
                self._set_sam_status(
                    self._tr(
                        "✅ {name} 로드됨 (고정 모델 검증 가능)",
                        "✅ {name} loaded (pinned-model verification available)",
                    ).format(name=self._sam_display_name(model_idx)),
                    "info",
                )
                self.sam_status.setToolTip("")
            else:
                self._set_sam_status(
                    self._tr(
                        "⚠️ {name} 로드됨, 하지만 OpenCV가 없어 트레이싱 불가",
                        "⚠️ {name} loaded, but tracing is blocked until OpenCV is installed",
                    ).format(name=self._sam_display_name(model_idx)),
                    "warning",
                )
                self.sam_status.setToolTip(get_cv2_error_text())
                self._set_model_aux_visibility(show_check=True, show_report=True, show_download=True, show_install=True)
                self._set_install_hint(
                    self._tr("📦 OpenCV 설치 (복사 가능):", "📦 Install OpenCV (copy this):"),
                    get_opencv_install_command(),
                )
        else:
            weights_path = getattr(self.sam_engine, "weights_path", "")
            if weights_path and os.path.exists(weights_path):
                self._set_sam_status(
                    self._tr(
                        "❌ {name} 로드 실패 - 재다운로드 권장",
                        "❌ Failed to load {name} - re-download recommended",
                    ).format(name=self._sam_display_name(model_idx)),
                    "error",
                )
            else:
                self._set_sam_status(
                    self._tr(
                        "⚠️ {name} 모델 파일 필요",
                        "⚠️ {name} model file required",
                    ).format(name=self._sam_display_name(model_idx)),
                    "warning",
                )
            self.sam_status.setToolTip(load_msg)

    def download_sam(self):
        model_idx = self.model_combo.currentIndex()
        if model_idx == MODEL_IDX_HED:
            self.download_hed()
            return
        self._get_or_create_sam_engine(model_idx)
        self.sam_download_btn.setEnabled(False)
        self._set_sam_status(self._tr("⏬ 다운로드 중...", "⏬ Downloading..."))
        self.iface.mainWindow().repaint()
        if self.sam_engine:
            success = self.sam_engine.download_weights()
            if success:
                QMessageBox.information(
                    self,
                    self._tr("완료", "Done"),
                    self._tr(
                        "{name} 다운로드 완료!",
                        "{name} download complete!",
                    ).format(name=self._sam_display_name(model_idx)),
                )
                self.init_sam_engine()
                self.check_sam_update(show_message=False)
            else:
                QMessageBox.critical(
                    self,
                    self._tr("오류", "Error"),
                    self._tr(
                        "다운로드 실패. 인터넷 연결을 확인하세요.",
                        "Download failed. Check your internet connection.",
                    ),
                )
                self._set_sam_status(self._tr("❌ 다운로드 실패", "❌ Download failed"), "error")
        self.sam_download_btn.setEnabled(True)

    @staticmethod
    def _format_size(size_bytes):
        if size_bytes is None:
            return "?"
        size = float(size_bytes)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024 or unit == "GB":
                return f"{size:.1f}{unit}"
            size /= 1024.0
        return f"{size_bytes}B"

    def check_sam_update(self, show_message=True):
        model_idx = self.model_combo.currentIndex()
        self._get_or_create_sam_engine(model_idx)

        self.sam_check_btn.setEnabled(False)
        self._set_sam_status(self._tr("🔎 고정 모델 검증 중...", "🔎 Verifying pinned model..."))
        self.iface.mainWindow().repaint()

        info = self.sam_engine.check_weights_update()
        self.sam_check_btn.setEnabled(True)

        if not info.get("ok"):
            self._set_sam_status(self._tr("❌ 모델 검증 실패", "❌ Model verification failed"), "error")
            if show_message:
                QMessageBox.warning(
                    self,
                    self._tr("경고", "Warning"),
                    self._tr(
                        "모델 파일이 없고 설정된 다운로드 소스도 확인할 수 없습니다.\n인터넷 연결을 확인하세요.",
                        "The model file is missing and its configured download source could not be reached.\nCheck your internet connection.",
                    ),
                )
            return

        status = info.get("status")
        local = info.get("local") or {}
        remote = info.get("remote") or {}
        local_size = self._format_size(local.get("size"))
        remote_size = self._format_size(remote.get("content_length"))

        if status == "not_installed":
            self._set_sam_status(
                self._tr(
                    f"⚠️ {self._sam_display_name(model_idx)} 없음 (다운로드 소스 {remote_size})",
                    f"⚠️ {self._sam_display_name(model_idx)} not installed (download source {remote_size})",
                ),
                "warning",
            )
            self.sam_download_btn.setText(self._download_button_text(model_idx))
            return

        if status in ("invalid", "update_available"):
            self._set_sam_status(
                self._tr(
                    f"❌ {self._sam_display_name(model_idx)} 고정 파일 검증 실패 (로컬 {local_size})",
                    f"❌ {self._sam_display_name(model_idx)} failed pinned-file verification (local {local_size})",
                ),
                "error",
            )
            self.sam_download_btn.setText(
                self._tr(
                    f"⬇️ {self._sam_display_name(model_idx)} 다시 다운로드",
                    f"⬇️ Re-download {self._sam_display_name(model_idx)}",
                )
            )
            if show_message:
                QMessageBox.information(
                    self,
                    self._tr("완료", "Done"),
                    self._tr(
                        "로컬 {name} 파일이 고정된 크기/SHA-256 정체성과 일치하지 않습니다.\n재다운로드하세요.",
                        "The local {name} file does not match its pinned size/SHA-256 identity.\nRe-download it.",
                    ).format(
                        name=self._sam_display_name(model_idx),
                    ),
                )
            return

        if status == "up_to_date":
            self._set_sam_status(
                self._tr(
                    f"✅ {self._sam_display_name(model_idx)} 고정 정체성 검증 완료 (로컬 {local_size})",
                    f"✅ {self._sam_display_name(model_idx)} matches its pinned identity (local {local_size})",
                ),
                "info",
            )
            self.sam_download_btn.setText(
                self._tr(
                    f"⬇️ {self._sam_display_name(model_idx)} 재다운로드",
                    f"⬇️ Re-download {self._sam_display_name(model_idx)}",
                )
            )
            return

        self._set_sam_status(
            self._tr(
                "ℹ️ 고정 모델 검증 상태 불명 (필요 시 재다운로드 가능)",
                "ℹ️ Pinned-model verification status is unavailable (re-download available)",
            ),
        )
        self.sam_download_btn.setText(
            self._tr(
                f"⬇️ {self._sam_display_name(model_idx)} 재다운로드",
                f"⬇️ Re-download {self._sam_display_name(model_idx)}",
            )
        )

    @staticmethod
    def _safe_module_version(package_name):
        try:
            import importlib.metadata as md
            return md.version(package_name)
        except Exception:
            return None

    def export_sam_report(self):
        self._set_sam_status(self._tr("📄 SAM 리포트 생성 중...", "📄 Building SAM report..."))
        self.iface.mainWindow().repaint()

        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "qgis_version": getattr(Qgis, "QGIS_VERSION", None),
            "python_version": os.sys.version,
            "cwd": os.getcwd(),
            "environment": {
                "QGIS_PREFIX_PATH": os.environ.get("QGIS_PREFIX_PATH"),
                "PYTHONPATH": os.environ.get("PYTHONPATH"),
            },
            "modules": {
                "requests": self._safe_module_version("requests"),
                "torch": self._safe_module_version("torch"),
                "mobile_sam": self._safe_module_version("mobile_sam"),
                "segment_anything": self._safe_module_version("segment_anything"),
                "PyYAML": self._safe_module_version("PyYAML"),
            },
        }

        try:
            model_idx = self.model_combo.currentIndex()
            self._get_or_create_sam_engine(model_idx)

            update_info = self.sam_engine.check_weights_update()
            report["sam_engine"] = {
                "display_name": getattr(self.sam_engine, "display_name", None),
                "backend": getattr(self.sam_engine, "backend", None),
                "model_type": getattr(self.sam_engine, "model_type", None),
                "weights_path": getattr(self.sam_engine, "weights_path", None),
                "weights_meta_path": getattr(self.sam_engine, "weights_meta_path", None),
                "weights_url": getattr(self.sam_engine, "model_spec", {}).get("weights_url"),
                "local_info": self.sam_engine.get_local_weights_info(),
                "update_check": update_info,
            }

            report_stem = os.path.splitext(SAM_REPORT_FILENAME)[0]
            descriptor, out_path = tempfile.mkstemp(
                prefix=f"{report_stem}-",
                suffix=".json",
            )
            try:
                with os.fdopen(
                    descriptor,
                    "w",
                    encoding="utf-8",
                    closefd=True,
                ) as f:
                    descriptor = None
                    json.dump(report, f, ensure_ascii=False, indent=2)
                    f.write("\n")
                    f.flush()
                    os.fsync(f.fileno())
            except Exception:
                if descriptor is not None:
                    os.close(descriptor)
                try:
                    os.remove(out_path)
                except OSError:
                    pass
                raise

            QApplication.clipboard().setText(json.dumps(report, ensure_ascii=False, indent=2))

            status = update_info.get("status", "unknown")
            self._set_sam_status(
                self._tr(
                    f"✅ SAM 리포트 생성 완료 ({status})",
                    f"✅ SAM report generated ({status})",
                ),
                "info",
            )
            QMessageBox.information(
                self,
                self._tr("완료", "Done"),
                self._tr(
                    "SAM 상태 리포트를 생성했습니다.\n- 클립보드에 복사됨\n- 저장 경로: {path}",
                    "SAM status report generated.\n- Copied to clipboard\n- Saved at: {path}",
                ).format(path=out_path),
            )
        except Exception as e:
            report["error"] = str(e)
            report["traceback"] = traceback.format_exc()
            self._set_sam_status(self._tr("❌ SAM 리포트 생성 실패", "❌ Failed to build SAM report"), "error")
            QMessageBox.critical(
                self,
                self._tr("오류", "Error"),
                self._tr(
                    "SAM 리포트 생성 실패:\n{err}",
                    "Failed to generate SAM report:\n{err}",
                ).format(err=str(e)),
            )

    def download_hed(self):
        if not is_cv2_available():
            self._show_opencv_warning("HED")
            return

        self.sam_download_btn.setEnabled(False)
        self._set_sam_status(
            self._tr(
                f"⏬ HED 다운로드 중 ({self._hed_size_hint_mb()}MB)...",
                f"⏬ Downloading HED (~{self._hed_size_hint_mb()}MB)...",
            )
        )
        self.iface.mainWindow().repaint()
        try:
            from ..core.edge_detector import EdgeDetector
            self._set_sam_status(self._tr("⏬ HED 다운로드 중...", "⏬ Downloading HED..."))
            success, error_message = EdgeDetector.download_hed_assets()
            if not success:
                raise RuntimeError(error_message or "Unknown HED download error")
            QMessageBox.information(
                self,
                self._tr("완료", "Done"),
                self._tr("HED 모델 다운로드 및 검증 완료!", "HED model download and validation complete!"),
            )
            self.check_hed_status()
        except Exception as e:
            QMessageBox.critical(
                self,
                self._tr("오류", "Error"),
                self._tr(
                    "HED 다운로드 실패:\n{err}",
                    "HED download failed:\n{err}",
                ).format(err=str(e)),
            )
            self._set_sam_status(self._tr("❌ 다운로드 실패", "❌ Download failed"), "error")
        self.sam_download_btn.setEnabled(True)

    def preview_edges(self):
        import numpy as np
        from osgeo import gdal

        raster = self.layer_combo.currentLayer()
        if not raster:
            QMessageBox.warning(self, self._tr("경고", "Warning"), self._tr("래스터 지도를 먼저 선택하세요.", "Select a raster map first."))
            return

        model_idx = self.model_combo.currentIndex()

        if self._is_sam_model(model_idx):
            QMessageBox.information(
                self,
                self._tr("안내", "Info"),
                self._tr(
                    "{name}은 클릭/호버 프롬프트에 반응하는 인터랙티브 모델입니다.\n트레이싱 시작 후 초록색 미리보기 선으로 결과를 확인하세요.",
                    "{name} is an interactive prompt-based model.\nStart tracing and use the green preview line to inspect its result.",
                ).format(
                    name=self._sam_display_name(model_idx),
                ),
            )
            return

        edge_method = EDGE_METHOD_BY_MODEL.get(model_idx, DEFAULT_EDGE_METHOD)

        if edge_method in ("lsd", "hed") and not is_cv2_available():
            self._show_opencv_warning(self._tr("엣지 미리보기", "edge preview"))
            return

        preview_directory = None
        dataset = None
        try:
            from ..core.edge_detector import EdgeDetector

            extent = self._canvas_extent_in_layer_crs(raster)
            provider = raster.dataProvider()
            raster_ext = raster.extent()
            read_ext = extent.intersect(raster_ext)
            if read_ext.isEmpty():
                QMessageBox.warning(self, self._tr("경고", "Warning"), self._tr("래스터 범위 밖입니다.", "Current view is outside raster extent."))
                return

            out_w, out_h = compute_resampled_dimensions(
                raster_ext.width(),
                raster_ext.height(),
                raster.width(),
                raster.height(),
                read_ext.width(),
                read_ext.height(),
                PREVIEW_EDGE_MAX_DIMENSION,
                min_dimension=1,
            )
            bands = read_raster_bands(
                provider,
                read_ext,
                out_w,
                out_h,
                max_bands=MAX_RASTER_BANDS_FOR_RGB,
            )
            if not bands:
                QMessageBox.warning(self, self._tr("경고", "Warning"), self._tr("래스터 데이터를 읽을 수 없습니다.", "Failed to read raster data."))
                return

            image = np.stack(bands[:3], axis=-1) if len(bands) >= 3 else bands[0]
            edges = EdgeDetector(method=edge_method).detect_edges(image)

            preview_directory = tempfile.TemporaryDirectory(
                prefix="archaeotrace-edge-preview-"
            )
            temp_path = os.path.join(preview_directory.name, "preview.tif")
            driver = gdal.GetDriverByName("GTiff")
            if driver is None:
                raise RuntimeError("GDAL GeoTIFF driver is unavailable.")
            dataset = driver.Create(
                temp_path,
                out_w,
                out_h,
                1,
                gdal.GDT_Byte,
            )
            if dataset is None:
                raise RuntimeError("GDAL could not create the preview raster.")
            dataset.SetGeoTransform([read_ext.xMinimum(), read_ext.width() / out_w, 0, read_ext.yMaximum(), 0, -read_ext.height() / out_h])
            dataset.SetProjection(raster.crs().toWkt())
            dataset.GetRasterBand(1).WriteArray(edges)
            dataset.FlushCache()
            dataset = None

            from qgis.core import QgsRasterLayer
            layer_name = self._tr("엣지 미리보기", "Edge Preview") + f" ({edge_method.upper()})"
            edge_layer = QgsRasterLayer(temp_path, layer_name)
            if edge_layer.isValid():
                QgsProject.instance().addMapLayer(edge_layer)
                self._preview_store.track(edge_layer, preview_directory)
                preview_directory = None
                QMessageBox.information(
                    self,
                    self._tr("완료", "Done"),
                    self._tr(
                        "'{name}' 레이어가 추가되었습니다.\n흰색=감지된 엣지",
                        "Layer '{name}' added.\nWhite=detected edges",
                    ).format(name=layer_name),
                )
            else:
                QMessageBox.critical(self, self._tr("오류", "Error"), self._tr("미리보기 레이어 생성 실패", "Failed to create preview layer"))
        except Exception as e:
            QMessageBox.critical(
                self,
                self._tr("오류", "Error"),
                self._tr(
                    "엣지 감지 실패:\n{err}",
                    "Edge detection failed:\n{err}",
                ).format(err=str(e)),
            )
        finally:
            dataset = None
            if preview_directory is not None:
                _TemporaryPreviewStore._cleanup_directory(preview_directory)

    def _help_text(self):
        if self.current_language == LANG_EN:
            return f"""
<h2>🏛️ {PLUGIN_NAME} Guide</h2>
<h3>📋 Basic Workflow</h3>
<ol>
<li><b>Select Raster Map</b> - choose a scanned map with contour lines.</li>
<li><b>Create SHP Output</b> - create a new line SHP or pick an existing line layer.</li>
<li><b>Choose Tracing Mode</b> - start with Ink Centerline; enable Smart Recovery only when wanted. LSD/HED/MobileSAM/SAM/Legacy Canny remain under Advanced.</li>
<li><b>Start Tracing</b> - click along contours and save the result.</li>
</ol>

<h3>🤖 Tracing Modes</h3>
<table border='1' cellpadding='5'>
<tr><th>Mode</th><th>Role</th><th>Requirements</th></tr>
<tr><td>🖋 Ink Centerline</td><td>Multi-scale dark/coloured line evidence with bounded Live-Wire</td><td>QGIS NumPy; SciPy/scikit-image optional; no model or OpenCV</td></tr>
<tr><td>🛟 Smart Recovery</td><td>EfficientSAM corridor challenger only on uncertain Ink segments; default OFF and Ink-preserving on failure</td><td>ONNX Runtime + explicitly installed, fixed-hash ~39.4 MiB split model</td></tr>
<tr><td>📐 LSD</td><td>Live-Wire over line-segment evidence</td><td>OpenCV + SciPy</td></tr>
<tr><td>🧠 HED</td><td>Learned edge-map assistance</td><td>OpenCV 4.8–4.11 + ~{self._hed_size_hint_mb()}MB model</td></tr>
<tr><td>🎯 MobileSAM</td><td>Prompt mask plus edge/A*</td><td>OpenCV + PyTorch + mobile_sam + ~{self._sam_size_hint_mb(MODEL_IDX_MOBILE_SAM)}MB weights</td></tr>
<tr><td>🧩 SAM</td><td>Prompt mask plus edge/A*</td><td>OpenCV + PyTorch + segment_anything + ~{self._sam_size_hint_mb(MODEL_IDX_SAM)}MB checkpoint</td></tr>
<tr><td>🔧 Legacy Canny</td><td>Compatibility gradient-edge mode</td><td>NumPy; OpenCV optional</td></tr>
</table>
<p>Accuracy rankings are intentionally omitted until the historical-map benchmark dataset is complete.</p>

<h3>🖱️ Controls</h3>
<ul>
<li><b>Left Click</b>: place/confirm points while tracing.</li>
<li><b>Right Click / Enter</b>: save current line.</li>
<li><b>Esc / Delete</b>: cancel current trace.</li>
<li><b>Ctrl+Z</b>: undo a checkpoint while tracing; after save, undo the complete QGIS edit command.</li>
<li><b>Click near start point</b>: close loop and enter elevation.</li>
</ul>

<h3>💡 Tips</h3>
<ul>
<li>Zoom in until contour lines are clearly visible for better snapping.</li>
<li>The assist slider is literal: 0% is the exact cursor, intermediate values blend geometry, and 100% uses the full Live-Wire route.</li>
<li>The green line is the exact path that one click will accept. Auto Path is required only for SAM proposals.</li>
<li>Smart Recovery reports Ink, Recovering, Enhanced, or Ink fallback. It never auto-downloads its model.</li>
<li>If SAM/HED is unavailable, start with Ink Centerline.</li>
<li>Use <b>Verify Selected SAM Model</b> to check the local file against its pinned size and SHA-256 identity.</li>
<li><b>SAM Status Report</b> performs the same integrity check before creating a shareable JSON report; internet is used only when the model is missing.</li>
</ul>

<h3>⚠️ Troubleshooting</h3>
<ul>
<li><b>No raster selected</b>: choose a raster layer in Step 1.</li>
<li><b>Model download failed</b>: check internet connection and retry.</li>
<li><b>No edges in preview</b>: zoom to map area and try another model.</li>
</ul>
"""
        return f"""
<h2>🏛️ {PLUGIN_NAME} 사용 가이드</h2>
<h3>📋 기본 워크플로우</h3>
<ol>
<li><b>래스터 지도 선택</b> - 등고선이 있는 스캔 지도를 선택합니다.</li>
<li><b>SHP 출력 설정</b> - 새 라인 SHP를 만들거나 기존 라인 레이어를 선택합니다.</li>
<li><b>추적 방식 선택</b> - 기본 Ink Centerline으로 시작하고 필요할 때만 Smart Recovery를 켭니다. LSD/HED/MobileSAM/SAM/Legacy Canny는 Advanced에 보존됩니다.</li>
<li><b>트레이싱 시작</b> - 등고선을 따라 클릭하며 추적한 뒤 저장합니다.</li>
</ol>

<h3>🤖 추적 방식</h3>
<table border='1' cellpadding='5'>
<tr><th>방식</th><th>역할</th><th>필요 항목</th></tr>
<tr><td>🖋 Ink Centerline</td><td>다중 스케일 검은색·유색 선 증거와 제한된 Live-Wire</td><td>QGIS NumPy, SciPy/scikit-image 선택, model·OpenCV 불필요</td></tr>
<tr><td>🛟 Smart Recovery</td><td>불확실한 Ink 구간에만 EfficientSAM corridor challenger 사용, 기본 OFF·실패 시 Ink 보존</td><td>ONNX Runtime과 명시적으로 설치한 고정-hash 약 39.4 MiB split model</td></tr>
<tr><td>📐 LSD</td><td>선분 지도를 이용한 Live-Wire</td><td>OpenCV + SciPy</td></tr>
<tr><td>🧠 HED</td><td>학습된 엣지 지도 보조</td><td>OpenCV 4.8–4.11 및 약 {self._hed_size_hint_mb()}MB 모델</td></tr>
<tr><td>🎯 MobileSAM</td><td>프롬프트 마스크와 edge/A*</td><td>OpenCV + PyTorch + mobile_sam 및 약 {self._sam_size_hint_mb(MODEL_IDX_MOBILE_SAM)}MB 가중치</td></tr>
<tr><td>🧩 SAM</td><td>프롬프트 마스크와 edge/A*</td><td>OpenCV + PyTorch + segment_anything 및 약 {self._sam_size_hint_mb(MODEL_IDX_SAM)}MB 체크포인트</td></tr>
<tr><td>🔧 Legacy Canny</td><td>기존 그래디언트 경계 검출 호환 모드</td><td>NumPy, OpenCV 선택</td></tr>
</table>
<p>실제 고지도 benchmark 데이터셋이 완성되기 전까지 정확도 순위는 표시하지 않습니다.</p>

<h3>🖱️ 조작법</h3>
<ul>
<li><b>좌클릭</b>: 점 배치/확정</li>
<li><b>우클릭 / Enter</b>: 현재 선 저장</li>
<li><b>Esc / Delete</b>: 현재 그리기 취소</li>
<li><b>Ctrl+Z</b>: 그리는 중에는 체크포인트, 저장 후에는 전체 QGIS 편집 명령 되돌리기</li>
<li><b>시작점 근처 클릭</b>: 닫힌 루프 생성 후 해발값 입력</li>
</ul>

<h3>💡 팁</h3>
<ul>
<li>등고선이 명확히 보일 정도로 확대하면 스냅 품질이 좋아집니다.</li>
<li>AI 개입 슬라이더는 실제 비율입니다. 0%는 정확한 커서, 중간값은 경로 혼합, 100%는 Live-Wire 전체 경로입니다.</li>
<li>초록색 선이 클릭 한 번으로 채택될 정확한 경로입니다. Auto Path는 SAM 제안에만 필요합니다.</li>
<li>Smart Recovery는 Ink, Recovering, Enhanced, Ink fallback 상태를 표시하며 model을 자동 download하지 않습니다.</li>
<li>SAM/HED가 준비되지 않았다면 Ink Centerline부터 시작하세요.</li>
<li><b>선택 SAM 모델 검증</b> 버튼으로 로컬 파일이 고정된 크기와 SHA-256 정체성에 맞는지 확인하세요.</li>
<li><b>SAM 상태 리포트</b>는 같은 무결성 검증 후 공유용 JSON을 생성하며, 모델이 없을 때만 인터넷을 사용합니다.</li>
</ul>

<h3>⚠️ 문제 해결</h3>
<ul>
<li><b>래스터 선택 안 됨</b>: 1단계에서 래스터 레이어를 선택하세요.</li>
<li><b>모델 다운로드 실패</b>: 인터넷 연결 확인 후 다시 시도하세요.</li>
<li><b>엣지 미리보기가 비어 있음</b>: 지도 범위로 이동/확대 후 다른 모델을 시도하세요.</li>
</ul>
"""

    def show_help(self):
        msg = QMessageBox(self)
        msg.setWindowTitle(self._tr(f"{PLUGIN_NAME} 도움말", f"{PLUGIN_NAME} Help"))
        msg.setTextFormat(_qt_value("RichText", "TextFormat"))
        msg.setText(self._help_text())
        msg.setStandardButtons(_message_box_button("Ok"))
        _exec_dialog(msg)


# Keep old name for compatibility
AIVectorizerDialog = AIVectorizerDock
