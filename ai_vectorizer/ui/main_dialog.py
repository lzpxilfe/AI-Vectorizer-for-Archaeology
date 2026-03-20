# -*- coding: utf-8 -*-
"""
ArchaeoTrace - AI-assisted contour tracing for historical maps
Dockable panel with guided workflow and tooltips
"""

import os
import json
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
    QgsSymbol,
    QgsSingleSymbolRenderer,
    Qgis,
)
from qgis.gui import QgsMapLayerComboBox
from qgis.PyQt.QtCore import Qt, QVariant, QSettings
from qgis.PyQt.QtGui import QColor

from ..core.dependencies import get_cv2_error_text, get_opencv_install_command, is_cv2_available
from ..core.raster_utils import compute_resampled_dimensions, read_raster_bands
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
    MODEL_IDX_CANNY,
    MODEL_IDX_HED,
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


class AIVectorizerDock(QDockWidget):
    """Dockable panel for ArchaeoTrace plugin."""

    def __init__(self, iface, parent=None):
        super().__init__(PLUGIN_NAME, parent)
        self.iface = iface
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self.active_tool = None
        self.output_layer = None
        self.sam_engines = {}
        self.sam_engine = None
        self.current_language = self._load_language()

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

    def _model_items(self):
        return [
            MODEL_MENU_LABELS[idx][self.current_language]
            for idx in (MODEL_IDX_CANNY, MODEL_IDX_LSD, MODEL_IDX_HED, MODEL_IDX_MOBILE_SAM, MODEL_IDX_SAM)
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

    def _set_idle_ui(self, prompt=False):
        self._set_trace_button_idle()
        self._set_ready_state(prompt=prompt)

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

    def _get_or_create_sam_engine(self, model_idx=None):
        spec = self._sam_engine_spec(model_idx)
        if spec is None:
            self.sam_engine = None
            return None

        SAMEngine = self._import_sam_engine()
        cache_key = (spec["backend"], spec["model_type"])
        if cache_key not in self.sam_engines:
            self.sam_engines[cache_key] = SAMEngine(
                backend=spec["backend"],
                model_type=spec["model_type"],
            )
        self.sam_engine = self.sam_engines[cache_key]
        return self.sam_engine

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

    def cleanup(self):
        if self.active_tool:
            try:
                self.iface.mapCanvas().unsetMapTool(self.active_tool)
            except Exception as exc:
                self._log_nonfatal_ui_error("Failed to unset active tool", exc)
        self.active_tool = None
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
        self.layer_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
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
        self.vector_combo.setFilters(QgsMapLayerProxyModel.LineLayer)
        self.vector_combo.layerChanged.connect(self.on_layer_selected)
        step2_layout.addWidget(self.vector_combo)
        self.step2_group.setLayout(step2_layout)
        self.layout.addWidget(self.step2_group)

        self.step3_group = QGroupBox()
        step3_layout = QVBoxLayout()
        self.model_desc_label = QLabel()
        self.model_desc_label.setStyleSheet("color: gray; font-size: 10px;")
        step3_layout.addWidget(self.model_desc_label)

        model_layout = QHBoxLayout()
        self.model_label = QLabel()
        model_layout.addWidget(self.model_label)
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_combo)
        step3_layout.addLayout(model_layout)

        self.sam_status = QLabel("")
        self.sam_status.setStyleSheet("font-size: 10px;")
        step3_layout.addWidget(self.sam_status)

        self.sam_check_btn = QPushButton()
        self.sam_check_btn.clicked.connect(self.check_sam_update)
        self.sam_check_btn.setVisible(False)
        step3_layout.addWidget(self.sam_check_btn)

        self.sam_report_btn = QPushButton()
        self.sam_report_btn.clicked.connect(self.export_sam_report)
        self.sam_report_btn.setVisible(False)
        step3_layout.addWidget(self.sam_report_btn)

        self.sam_download_btn = QPushButton()
        self.sam_download_btn.clicked.connect(self.download_sam)
        self.sam_download_btn.setVisible(False)
        step3_layout.addWidget(self.sam_download_btn)

        self.install_guide = QLabel()
        self.install_guide.setStyleSheet("color: #e67e22; font-size: 9px;")
        self.install_guide.setVisible(False)
        step3_layout.addWidget(self.install_guide)

        self.install_cmd = QLineEdit()
        self.install_cmd.setText(self._install_command_for_model())
        self.install_cmd.setReadOnly(True)
        self.install_cmd.setStyleSheet("background: #fff3e0; font-size: 9px; padding: 3px;")
        self.install_cmd.setVisible(False)
        step3_layout.addWidget(self.install_cmd)

        self.freehand_check = QCheckBox()
        step3_layout.addWidget(self.freehand_check)

        edge_layout = QHBoxLayout()
        self.edge_strength_label = QLabel()
        edge_layout.addWidget(self.edge_strength_label)
        self.freedom_slider = QSlider(Qt.Horizontal)
        self.freedom_slider.setMinimum(0)
        self.freedom_slider.setMaximum(100)
        self.freedom_slider.setValue(DEFAULT_FREEDOM_SLIDER_VALUE)
        edge_layout.addWidget(self.freedom_slider)
        self.freedom_label = QLabel(f"{DEFAULT_FREEDOM_SLIDER_VALUE}%")
        self.freedom_slider.valueChanged.connect(lambda v: self.freedom_label.setText(f"{v}%"))
        edge_layout.addWidget(self.freedom_label)
        step3_layout.addLayout(edge_layout)

        self.trace_btn = QPushButton()
        self.trace_btn.setCheckable(True)
        self.trace_btn.clicked.connect(self.toggle_trace_tool)
        self.trace_btn.setStyleSheet(TRACE_BUTTON_IDLE_STYLE)
        self.trace_btn.setEnabled(False)
        step3_layout.addWidget(self.trace_btn)

        self.step3_group.setLayout(step3_layout)
        self.layout.addWidget(self.step3_group)

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
        self.step3_group.setToolTip(self._tr("등고선을 따라 그리기 위한 AI 설정", "AI options for contour tracing"))
        self.model_desc_label.setText(self._tr("💡 AI 모델: 등고선 인식 방식 선택", "💡 AI model: choose contour detection behavior"))
        self.model_label.setText(self._tr("AI 모델:", "AI Model:"))
        self.model_label.setToolTip(
            self._tr(
                (
                    "각 모델의 장단점:\n"
                    "• Canny: 가장 빠름, 기본\n"
                    "• LSD: 선분 기반, 빠름\n"
                    f"• HED: 딥러닝, 매끄러움 (~{self._hed_size_hint_mb()}MB)\n"
                    f"• MobileSAM: 경량 세그멘테이션 (~{self._sam_size_hint_mb(MODEL_IDX_MOBILE_SAM)}MB)\n"
                    f"• SAM: 고정밀 세그멘테이션 (~{self._sam_size_hint_mb(MODEL_IDX_SAM)}MB)"
                ),
                (
                    "Model tradeoffs:\n"
                    "• Canny: fastest baseline\n"
                    "• LSD: line-based, fast\n"
                    f"• HED: deep-learning, smooth (~{self._hed_size_hint_mb()}MB)\n"
                    f"• MobileSAM: lightweight segmentation (~{self._sam_size_hint_mb(MODEL_IDX_MOBILE_SAM)}MB)\n"
                    f"• SAM: highest-precision segmentation (~{self._sam_size_hint_mb(MODEL_IDX_SAM)}MB)"
                ),
            )
        )
        self.model_combo.setToolTip(
            self._tr(
                "Canny: 기본\nLSD: 선분 기반\nHED: 딥러닝 엣지\nMobileSAM: 경량 세그멘테이션\nSAM: 정밀 세그멘테이션",
                "Canny: baseline\n"
                "LSD: line detector\n"
                "HED: deep edge detector\n"
                "MobileSAM: lightweight segmentation\n"
                "SAM: precise segmentation",
            )
        )
        self.sam_check_btn.setText(self._tr("🔎 선택 SAM 모델 최신 확인", "🔎 Check Selected SAM Model"))
        self.sam_check_btn.setToolTip(
            self._tr(
                "현재 선택된 SAM 계열 모델의 원격 메타데이터(ETag/크기)와 비교합니다",
                "Compare the selected SAM-family model against remote metadata (ETag/size)",
            )
        )
        self.sam_report_btn.setText(self._tr("📄 SAM 상태 리포트", "📄 SAM Status Report"))
        self.sam_report_btn.setToolTip(
            self._tr(
                "현재 SAM 환경/버전/모델 상태를 JSON으로 저장하고 클립보드에 복사합니다",
                "Export current SAM environment/version/model status as JSON and copy it to clipboard",
            )
        )
        self.sam_download_btn.setToolTip(self._tr("인터넷 연결 필요. 최초 1회만 다운로드", "Internet required. Download once on first use"))
        self.install_guide.setText(self._tr("📦 선택 모델 설치 (복사 가능):", "📦 Selected Model Install (copy this):"))
        self.install_cmd.setText(self._install_command_for_model())
        self.freehand_check.setText(self._tr("✏️ 프리핸드 (AI 비활성)", "✏️ Freehand (AI Off)"))
        self.freehand_check.setToolTip(self._tr("체크: AI 없이 순수 마우스 추적", "Checked: pure mouse tracing without AI"))
        self.edge_strength_label.setText(self._tr("AI 강도:", "AI Strength:"))
        self.freedom_slider.setToolTip(self._tr("0%: 자유롭게\n100%: 엣지 따라감", "0%: freer draw\n100%: stronger edge following"))
        if self.trace_btn.isChecked():
            self._set_trace_button_active()
        else:
            self._set_trace_button_idle()
        self.trace_btn.setToolTip(self._tr("클릭하여 트레이싱 시작", "Click to start tracing"))

        self.status_box.setTitle(self._tr("📋 상태", "📋 Status"))
        self.status_label.setToolTip(self._tr("현재 트레이싱 상태를 표시합니다", "Shows current tracing state"))
        self.controls_title_label.setText(self._tr("📖 사용법:", "📖 Controls:"))
        self.controls_label.setText(
            self._tr(
                "• 드래그: 선 그리기 / 클릭: 체크포인트\n"
                "• Ctrl+Z: 마지막 체크포인트로 되돌리기\n"
                "• Esc: 현재 그리기 취소 / Del: 전체 취소\n"
                "• 시작점 클릭: 폴리곤 닫기 → 해발값\n"
                "• 우클릭/Enter: 저장",
                "• Drag: draw line / Click: checkpoint\n"
                "• Ctrl+Z: undo to last checkpoint\n"
                "• Esc: cancel current trace / Del: cancel all\n"
                "• Click start point: close polygon -> elevation\n"
                "• Right click / Enter: save",
            )
        )
        self.controls_label.setToolTip(self._tr("클릭으로 체크포인트 저장\n실수하면 Ctrl+Z로 되돌림", "Click to place checkpoints\nUse Ctrl+Z to undo"))

        self.debug_box.setTitle(self._tr("🔧 디버그 및 도움말", "🔧 Debug & Help"))
        self.debug_box.setToolTip(self._tr("문제 해결을 위한 도구들", "Tools for troubleshooting"))
        self.preview_edge_btn.setText(self._tr("👁️ AI가 보는 엣지 미리보기", "👁️ Preview AI-Detected Edges"))
        self.preview_edge_btn.setToolTip(
            self._tr(
                "현재 선택된 AI 모델이 감지하는 엣지를\n임시 래스터 레이어로 표시합니다.\n\n흰색 = AI가 인식하는 등고선",
                "Shows detected edges from the selected AI model\nas a temporary raster layer.\n\nWhite = detected contour edges",
            )
        )
        self.help_btn.setText(self._tr("❓ 도움말", "❓ Help"))
        self.help_btn.setToolTip(self._tr("사용법과 문제해결 안내", "Usage guide and troubleshooting"))

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
        path = self.shp_path.text()
        if not path:
            QMessageBox.warning(self, self._tr("경고", "Warning"), self._tr("파일 경로를 지정해주세요.", "Please specify an output file path."))
            return

        raster = self.layer_combo.currentLayer()
        crs = raster.crs() if raster else QgsCoordinateReferenceSystem(DEFAULT_CRS_AUTHID)
        fields = [QgsField(FIELD_ID, QVariant.Int), QgsField(FIELD_ELEVATION, QVariant.Double)]

        layer = QgsVectorLayer(f"LineString?crs={crs.authid()}", DEFAULT_OUTPUT_LAYER_NAME, "memory")
        layer.dataProvider().addAttributes(fields)
        layer.updateFields()
        error = QgsVectorFileWriter.writeAsVectorFormat(
            layer,
            path,
            DEFAULT_VECTOR_FILE_ENCODING,
            crs,
            "ESRI Shapefile",
        )

        if error[0] == QgsVectorFileWriter.NoError:
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
        if layer:
            self.output_layer = layer
            self.enable_tracing()

    def enable_tracing(self):
        self.trace_btn.setEnabled(True)
        self._set_ready_state(prompt=True)

    def toggle_trace_tool(self, checked):
        if checked:
            raster = self.layer_combo.currentLayer()
            if not raster:
                QMessageBox.warning(self, self._tr("경고", "Warning"), self._tr("래스터 지도를 선택하세요.", "Please select a raster map."))
                self.trace_btn.setChecked(False)
                return

            edge_weight = self.freedom_slider.value() / 100.0
            freehand = self.freehand_check.isChecked()
            model_idx = self.model_combo.currentIndex()
            if self._is_sam_model(model_idx) and not freehand:
                self._get_or_create_sam_engine(model_idx)
            else:
                self.sam_engine = None
            use_sam = (
                not freehand
                and self._is_sam_model(model_idx)
                and self.sam_engine is not None
                and self.sam_engine.is_ready
            )
            edge_method = SAM_ASSIST_EDGE_METHOD if use_sam else EDGE_METHOD_BY_MODEL.get(model_idx, DEFAULT_EDGE_METHOD)

            if not freehand and not is_cv2_available():
                self._show_opencv_warning(self._tr("AI 트레이싱", "AI tracing"))
                self.trace_btn.setChecked(False)
                return

            if not freehand and model_idx == MODEL_IDX_HED:
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

            if not freehand and self._is_sam_model(model_idx) and not use_sam:
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

            from ..tools.smart_trace_tool import SmartTraceTool
            self.active_tool = SmartTraceTool(
                self.iface.mapCanvas(),
                raster,
                self.output_layer,
                model_type=model_idx,
                edge_weight=edge_weight,
                freehand=freehand,
                sam_engine=self.sam_engine if use_sam else None,
                edge_method=edge_method,
                iface=self.iface,
                language=self.current_language,
            )
            self.iface.mapCanvas().setMapTool(self.active_tool)
            self.active_tool.deactivated.connect(self.on_tool_deactivated)

            if freehand:
                mode_name = self._tr("프리핸드", "Freehand")
            else:
                mode_name = self._sam_display_name(model_idx) if use_sam else self._mode_name(model_idx)
            self._set_tracing_state(mode_name)
        else:
            if self.active_tool:
                self.iface.mapCanvas().unsetMapTool(self.active_tool)
            self._set_idle_ui()

    def on_tool_deactivated(self):
        self._set_idle_ui()
        self.active_tool = None

    def on_model_changed(self, index):
        self._set_model_aux_visibility()
        self.sam_engine = None
        if index in (MODEL_IDX_CANNY, MODEL_IDX_LSD):
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
            self._set_model_aux_visibility(show_check=True, show_report=True)
            self.init_sam_engine()

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
                        "✅ {name} 로드됨 (최신 확인 가능)",
                        "✅ {name} loaded (update check available)",
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
        self._set_sam_status(self._tr("🔎 최신 모델 확인 중...", "🔎 Checking latest model..."))
        self.iface.mainWindow().repaint()

        info = self.sam_engine.check_weights_update()
        self.sam_check_btn.setEnabled(True)

        if not info.get("ok"):
            self._set_sam_status(self._tr("❌ 최신 확인 실패", "❌ Latest check failed"), "error")
            if show_message:
                QMessageBox.warning(
                    self,
                    self._tr("경고", "Warning"),
                    self._tr(
                        "최신 모델 확인에 실패했습니다.\n인터넷 연결을 확인하세요.",
                        "Failed to check latest model.\nPlease check your internet connection.",
                    ),
                )
            return

        status = info.get("status")
        local = info.get("local", {})
        remote = info.get("remote", {})
        local_size = self._format_size(local.get("size"))
        remote_size = self._format_size(remote.get("content_length"))

        if status == "not_installed":
            self._set_sam_status(
                self._tr(
                    f"⚠️ {self._sam_display_name(model_idx)} 없음 (원격 {remote_size})",
                    f"⚠️ {self._sam_display_name(model_idx)} not installed (remote {remote_size})",
                ),
                "warning",
            )
            self.sam_download_btn.setText(self._download_button_text(model_idx))
            return

        if status == "update_available":
            self._set_sam_status(
                self._tr(
                    f"⬆️ {self._sam_display_name(model_idx)} 업데이트 가능 (로컬 {local_size} → 원격 {remote_size})",
                    f"⬆️ {self._sam_display_name(model_idx)} update available (local {local_size} -> remote {remote_size})",
                ),
                "warning",
            )
            self.sam_download_btn.setText(
                self._tr(
                    f"⬆️ {self._sam_display_name(model_idx)} 업데이트",
                    f"⬆️ Update {self._sam_display_name(model_idx)}",
                )
            )
            if show_message:
                QMessageBox.information(
                    self,
                    self._tr("완료", "Done"),
                    self._tr(
                        "새 {name} 모델이 있습니다.\n'업데이트' 버튼으로 바로 받을 수 있습니다.",
                        "A newer {name} model is available.\nUse the update button to download it.",
                    ).format(
                        name=self._sam_display_name(model_idx),
                    ),
                )
            return

        if status == "up_to_date":
            self._set_sam_status(
                self._tr(
                    f"✅ {self._sam_display_name(model_idx)} 최신 상태 (로컬 {local_size})",
                    f"✅ {self._sam_display_name(model_idx)} is up to date (local {local_size})",
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
                "ℹ️ 버전 비교 정보 부족 (필요 시 재다운로드 가능)",
                "ℹ️ Not enough metadata to compare versions (re-download available)",
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

            out_path = os.path.join(tempfile.gettempdir(), SAM_REPORT_FILENAME)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

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
        import tempfile
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

        if not is_cv2_available():
            self._show_opencv_warning(self._tr("엣지 미리보기", "edge preview"))
            return

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

            temp_path = os.path.join(tempfile.gettempdir(), f"edge_preview_{edge_method}.tif")
            ds = gdal.GetDriverByName("GTiff").Create(temp_path, out_w, out_h, 1, gdal.GDT_Byte)
            ds.SetGeoTransform([read_ext.xMinimum(), read_ext.width() / out_w, 0, read_ext.yMaximum(), 0, -read_ext.height() / out_h])
            ds.SetProjection(raster.crs().toWkt())
            ds.GetRasterBand(1).WriteArray(edges)
            ds = None

            from qgis.core import QgsRasterLayer
            layer_name = self._tr("엣지 미리보기", "Edge Preview") + f" ({edge_method.upper()})"
            edge_layer = QgsRasterLayer(temp_path, layer_name)
            if edge_layer.isValid():
                QgsProject.instance().addMapLayer(edge_layer)
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

    def _help_text(self):
        if self.current_language == LANG_EN:
            return f"""
<h2>🏛️ {PLUGIN_NAME} Guide</h2>
<h3>📋 Basic Workflow</h3>
<ol>
<li><b>Select Raster Map</b> - choose a scanned map with contour lines.</li>
<li><b>Create SHP Output</b> - create a new line SHP or pick an existing line layer.</li>
<li><b>Choose AI Model</b> - Canny/LSD/HED/MobileSAM/SAM depending on speed and quality.</li>
<li><b>Start Tracing</b> - click along contours and save the result.</li>
</ol>

<h3>🤖 AI Model Comparison</h3>
<table border='1' cellpadding='5'>
<tr><th>Model</th><th>Speed</th><th>Quality</th><th>Notes</th></tr>
<tr><td>🔧 Canny</td><td>Fastest</td><td>Basic</td><td>Built-in</td></tr>
<tr><td>📐 LSD</td><td>Fast</td><td>Good</td><td>Built-in</td></tr>
<tr><td>🧠 HED</td><td>Medium</td><td>High</td><td>~{self._hed_size_hint_mb()}MB model</td></tr>
<tr><td>🎯 MobileSAM</td><td>Slow</td><td>High</td><td>Requires PyTorch + ~{self._sam_size_hint_mb(MODEL_IDX_MOBILE_SAM)}MB model</td></tr>
<tr><td>🧩 SAM</td><td>Slowest</td><td>Highest</td><td>Requires PyTorch + ~{self._sam_size_hint_mb(MODEL_IDX_SAM)}MB checkpoint</td></tr>
</table>

<h3>🖱️ Controls</h3>
<ul>
<li><b>Left Click</b>: place/confirm points while tracing.</li>
<li><b>Right Click / Enter</b>: save current line.</li>
<li><b>Esc / Delete</b>: cancel current trace.</li>
<li><b>Ctrl+Z</b>: undo back to checkpoint.</li>
<li><b>Click near start point</b>: close loop and enter elevation.</li>
</ul>

<h3>💡 Tips</h3>
<ul>
<li>Zoom in until contour lines are clearly visible for better snapping.</li>
<li>If tracing is noisy, move the mouse more slowly and lower AI strength.</li>
<li>If SAM/HED is unavailable, start with Canny or LSD first.</li>
<li>Use <b>Check Selected SAM Model</b> before downloading to see if an update is needed.</li>
<li>Use <b>SAM Status Report</b> to create a shareable JSON report for support.</li>
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
<li><b>AI 모델 선택</b> - 속도/품질에 맞춰 Canny/LSD/HED/MobileSAM/SAM을 선택합니다.</li>
<li><b>트레이싱 시작</b> - 등고선을 따라 클릭하며 추적한 뒤 저장합니다.</li>
</ol>

<h3>🤖 AI 모델 비교</h3>
<table border='1' cellpadding='5'>
<tr><th>모델</th><th>속도</th><th>품질</th><th>비고</th></tr>
<tr><td>🔧 Canny</td><td>가장 빠름</td><td>기본</td><td>내장</td></tr>
<tr><td>📐 LSD</td><td>빠름</td><td>좋음</td><td>내장</td></tr>
<tr><td>🧠 HED</td><td>보통</td><td>우수</td><td>약 {self._hed_size_hint_mb()}MB 모델 필요</td></tr>
<tr><td>🎯 MobileSAM</td><td>느림</td><td>우수</td><td>PyTorch 및 약 {self._sam_size_hint_mb(MODEL_IDX_MOBILE_SAM)}MB 모델 필요</td></tr>
<tr><td>🧩 SAM</td><td>가장 느림</td><td>최고</td><td>PyTorch 및 약 {self._sam_size_hint_mb(MODEL_IDX_SAM)}MB 체크포인트 필요</td></tr>
</table>

<h3>🖱️ 조작법</h3>
<ul>
<li><b>좌클릭</b>: 점 배치/확정</li>
<li><b>우클릭 / Enter</b>: 현재 선 저장</li>
<li><b>Esc / Delete</b>: 현재 그리기 취소</li>
<li><b>Ctrl+Z</b>: 체크포인트로 되돌리기</li>
<li><b>시작점 근처 클릭</b>: 닫힌 루프 생성 후 해발값 입력</li>
</ul>

<h3>💡 팁</h3>
<ul>
<li>등고선이 명확히 보일 정도로 확대하면 스냅 품질이 좋아집니다.</li>
<li>선이 튀면 마우스를 천천히 움직이고 AI 강도를 낮춰보세요.</li>
<li>SAM/HED가 준비되지 않았다면 Canny/LSD부터 시작하세요.</li>
<li>다운로드 전에 <b>선택 SAM 모델 최신 확인</b> 버튼으로 업데이트 필요 여부를 확인하세요.</li>
<li>문제 공유가 필요하면 <b>SAM 상태 리포트</b> 버튼으로 JSON 리포트를 생성하세요.</li>
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
        msg.setTextFormat(Qt.RichText)
        msg.setText(self._help_text())
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()


# Keep old name for compatibility
AIVectorizerDialog = AIVectorizerDock
