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
    QgsSymbol,
    QgsSingleSymbolRenderer,
    Qgis,
)
from qgis.gui import QgsMapLayerComboBox
from qgis.PyQt.QtCore import Qt, QVariant, QSettings
from qgis.PyQt.QtGui import QColor

from ..config import (
    DEFAULT_CRS_AUTHID,
    DEFAULT_EDGE_METHOD,
    DEFAULT_FREEDOM_SLIDER_VALUE,
    DEFAULT_OUTPUT_LAYER_NAME,
    DEFAULT_SAM_MODEL_TYPE,
    EDGE_METHOD_BY_MODEL,
    MAX_RASTER_BANDS_FOR_RGB,
    MODEL_IDX_CANNY,
    MODEL_IDX_HED,
    MODEL_IDX_LSD,
    MODEL_IDX_SAM,
    PREVIEW_EDGE_MAX_DIMENSION,
    TRACE_BUTTON_ACTIVE_STYLE,
    TRACE_BUTTON_IDLE_STYLE,
)


LANG_KO = "ko"
LANG_EN = "en"


class AIVectorizerDock(QDockWidget):
    """Dockable panel for ArchaeoTrace plugin."""

    SETTINGS_LANG_KEY = "ArchaeoTrace/language"

    def __init__(self, iface, parent=None):
        super().__init__("ArchaeoTrace", parent)
        self.iface = iface
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self.active_tool = None
        self.output_layer = None
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
        value = settings.value(self.SETTINGS_LANG_KEY, None)
        if value is None:
            locale = str(settings.value("locale/userLocale", "ko"))
            return LANG_EN if locale.lower().startswith("en") else LANG_KO
        lang = str(value)
        return lang if lang in (LANG_KO, LANG_EN) else LANG_KO

    def _save_language(self):
        QSettings().setValue(self.SETTINGS_LANG_KEY, self.current_language)

    def _model_items(self):
        if self.current_language == LANG_EN:
            return [
                "🔧 OpenCV Canny (Default)",
                "📐 LSD Line Detector (Fast)",
                "🧠 HED Deep Learning (Smooth)",
                "🎯 MobileSAM (High Quality)",
            ]
        return [
            "🔧 OpenCV Canny (기본)",
            "📐 LSD 선분검출 (빠름)",
            "🧠 HED 딥러닝 (매끄러움)",
            "🎯 MobileSAM (고품질)",
        ]

    def _mode_name(self, idx):
        names = {
            MODEL_IDX_CANNY: "Canny",
            MODEL_IDX_LSD: "LSD",
            MODEL_IDX_HED: "HED",
            MODEL_IDX_SAM: "SAM",
        }
        return names.get(idx, "OpenCV")

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
        self.install_cmd.setText(
            "pip install torch torchvision git+https://github.com/ChaoningZhang/MobileSAM.git"
        )
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

        self.setWindowTitle("ArchaeoTrace")
        self.header_label.setText(
            self._tr(
                "🏛️ ArchaeoTrace - 고지도 등고선 벡터화",
                "🏛️ ArchaeoTrace - Historical Map Contour Vectorization",
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
                "각 모델의 장단점:\n• Canny: 가장 빠름, 기본\n• LSD: 선분 기반, 빠름\n• HED: 딥러닝, 매끄러움\n• SAM: 최고 품질 (56MB)",
                "Model tradeoffs:\n• Canny: fastest baseline\n• LSD: line-based, fast\n• HED: deep-learning, smooth\n• SAM: best quality (~56MB)",
            )
        )
        self.model_combo.setToolTip(
            self._tr(
                "Canny: 기본\nLSD: 선분 기반\nHED: 딥러닝 엣지\nSAM: 세그멘테이션",
                "Canny: baseline\nLSD: line detector\nHED: deep edge detector\nSAM: segmentation",
            )
        )
        self.sam_check_btn.setText(self._tr("🔎 MobileSAM 최신 확인", "🔎 Check MobileSAM Latest"))
        self.sam_check_btn.setToolTip(
            self._tr(
                "원격 모델 메타데이터(ETag/크기)와 비교해 최신 여부를 확인합니다",
                "Compare remote model metadata (ETag/size) with local file",
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
        self.install_guide.setText(self._tr("📦 SAM 설치 (복사 가능):", "📦 SAM Install (copy this):"))
        self.freehand_check.setText(self._tr("✏️ 프리핸드 (AI 비활성)", "✏️ Freehand (AI Off)"))
        self.freehand_check.setToolTip(self._tr("체크: AI 없이 순수 마우스 추적", "Checked: pure mouse tracing without AI"))
        self.edge_strength_label.setText(self._tr("AI 강도:", "AI Strength:"))
        self.freedom_slider.setToolTip(self._tr("0%: 자유롭게\n100%: 엣지 따라감", "0%: freer draw\n100%: stronger edge following"))
        self.trace_btn.setText(self._tr("⏹️ 중지", "⏹️ Stop") if self.trace_btn.isChecked() else self._tr("🖊️ 트레이싱 시작", "🖊️ Start Tracing"))
        self.trace_btn.setToolTip(self._tr("클릭하여 트레이싱 시작", "Click to start tracing"))

        self.status_box.setTitle(self._tr("📋 상태", "📋 Status"))
        self.status_label.setToolTip(self._tr("현재 트레이싱 상태를 표시합니다", "Shows current tracing state"))
        self.controls_title_label.setText(self._tr("📖 사용법:", "📖 Controls:"))
        self.controls_label.setText(
            self._tr(
                "• 드래그: 선 그리기 / 클릭: 체크포인트\n• Ctrl+Z: 마지막 체크포인트로 되돌리기\n• Esc: 현재 그리기 취소 / Del: 전체 취소\n• 시작점 클릭: 폴리곤 닫기 → 해발값\n• 우클릭/Enter: 저장",
                "• Drag: draw line / Click: checkpoint\n• Ctrl+Z: undo to last checkpoint\n• Esc: cancel current trace / Del: cancel all\n• Click start point: close polygon -> elevation\n• Right click / Enter: save",
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

        if self.model_combo.currentIndex() == MODEL_IDX_HED:
            self.sam_download_btn.setText(self._tr("📥 HED 다운로드", "📥 Download HED"))
        else:
            self.sam_download_btn.setText(self._tr("⬇️ MobileSAM 다운로드 (~40MB)", "⬇️ Download MobileSAM (~40MB)"))

        if not self.trace_btn.isEnabled():
            self.status_label.setText(self._tr("SHP 파일을 먼저 생성하세요", "Create or select an SHP layer first"))
        elif self.trace_btn.isChecked():
            self.status_label.setText(self._tr("🖊️ [{mode}] 등고선을 클릭하세요", "🖊️ [{mode}] Click on contours").format(mode=self._mode_name(self.model_combo.currentIndex())))
        else:
            self.status_label.setText(self._tr("✅ 준비 완료", "✅ Ready"))

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
        fields = [QgsField("id", QVariant.Int), QgsField("elevation", QVariant.Double)]

        layer = QgsVectorLayer(f"LineString?crs={crs.authid()}", DEFAULT_OUTPUT_LAYER_NAME, "memory")
        layer.dataProvider().addAttributes(fields)
        layer.updateFields()
        error = QgsVectorFileWriter.writeAsVectorFormat(layer, path, "UTF-8", crs, "ESRI Shapefile")

        if error[0] == QgsVectorFileWriter.NoError:
            name = os.path.basename(path).replace(".shp", "")
            self.output_layer = QgsVectorLayer(path, name, "ogr")
            symbol = QgsSymbol.defaultSymbol(self.output_layer.geometryType())
            symbol.setColor(QColor(255, 0, 0))
            symbol.setWidth(1.2)
            self.output_layer.setRenderer(QgsSingleSymbolRenderer(symbol))
            QgsProject.instance().addMapLayer(self.output_layer)
            self.vector_combo.setLayer(self.output_layer)
            if not self.output_layer.isEditable():
                self.output_layer.startEditing()
            self.enable_tracing()
            QMessageBox.information(
                self,
                self._tr("성공", "Success"),
                self._tr("SHP 생성 완료 (편집 모드):\n{path}", "SHP created successfully (edit mode):\n{path}").format(path=path),
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
            if not self.output_layer.isEditable():
                self.output_layer.startEditing()
            self.enable_tracing()

    def enable_tracing(self):
        self.trace_btn.setEnabled(True)
        self.status_label.setText(self._tr("✅ 준비 완료! 트레이싱을 시작하세요", "✅ Ready! Start tracing"))
        self.status_label.setStyleSheet("color: green; font-weight: bold;")

    def toggle_trace_tool(self, checked):
        if checked:
            raster = self.layer_combo.currentLayer()
            if not raster:
                QMessageBox.warning(self, self._tr("경고", "Warning"), self._tr("래스터 지도를 선택하세요.", "Please select a raster map."))
                self.trace_btn.setChecked(False)
                return

            from ..tools.smart_trace_tool import SmartTraceTool

            edge_weight = self.freedom_slider.value() / 100.0
            freehand = self.freehand_check.isChecked()
            model_idx = self.model_combo.currentIndex()
            use_sam = model_idx == MODEL_IDX_SAM and self.sam_engine is not None and self.sam_engine.is_ready
            edge_method = EDGE_METHOD_BY_MODEL.get(model_idx, DEFAULT_EDGE_METHOD)

            self.active_tool = SmartTraceTool(
                self.iface.mapCanvas(),
                raster,
                self.output_layer,
                edge_weight=edge_weight,
                freehand=freehand,
                sam_engine=self.sam_engine if use_sam else None,
                edge_method=edge_method,
                iface=self.iface,
                language=self.current_language,
            )
            self.iface.mapCanvas().setMapTool(self.active_tool)
            self.active_tool.deactivated.connect(self.on_tool_deactivated)

            mode_name = "SAM" if use_sam else self._mode_name(model_idx)
            self.status_label.setText(self._tr("🖊️ [{mode}] 등고선을 클릭하세요", "🖊️ [{mode}] Click on contours").format(mode=mode_name))
            self.trace_btn.setText(self._tr("⏹️ 중지", "⏹️ Stop"))
            self.trace_btn.setStyleSheet(TRACE_BUTTON_ACTIVE_STYLE)
        else:
            if self.active_tool:
                self.iface.mapCanvas().unsetMapTool(self.active_tool)
            self.status_label.setText(self._tr("✅ 준비 완료", "✅ Ready"))
            self.trace_btn.setText(self._tr("🖊️ 트레이싱 시작", "🖊️ Start Tracing"))
            self.trace_btn.setStyleSheet(TRACE_BUTTON_IDLE_STYLE)

    def on_tool_deactivated(self):
        self.trace_btn.setChecked(False)
        self.trace_btn.setText(self._tr("🖊️ 트레이싱 시작", "🖊️ Start Tracing"))
        self.trace_btn.setStyleSheet(TRACE_BUTTON_IDLE_STYLE)
        self.status_label.setText(self._tr("✅ 준비 완료", "✅ Ready"))
        self.active_tool = None

    def on_model_changed(self, index):
        self.sam_check_btn.setVisible(False)
        self.sam_report_btn.setVisible(False)
        self.sam_download_btn.setVisible(False)
        self.install_guide.setVisible(False)
        self.install_cmd.setVisible(False)
        if index in (MODEL_IDX_CANNY, MODEL_IDX_LSD):
            self.sam_status.setText(self._tr("OpenCV 내장", "Built-in OpenCV"))
            self.sam_status.setStyleSheet("color: green; font-size: 10px;")
        elif index == MODEL_IDX_HED:
            self.check_hed_status()
        elif index == MODEL_IDX_SAM:
            self.sam_check_btn.setVisible(True)
            self.sam_report_btn.setVisible(True)
            self.init_sam_engine()

    def check_hed_status(self):
        from ..core.edge_detector import EdgeDetector
        if EdgeDetector.is_hed_available():
            self.sam_status.setText(self._tr("✅ HED 모델 로드됨", "✅ HED model loaded"))
            self.sam_status.setStyleSheet("color: green; font-size: 10px;")
        else:
            self.sam_status.setText(self._tr("⚠️ HED 모델 필요 (56MB)", "⚠️ HED model required (~56MB)"))
            self.sam_status.setStyleSheet("color: orange; font-size: 10px;")
            self.sam_download_btn.setVisible(True)
            self.sam_download_btn.setText(self._tr("📥 HED 다운로드", "📥 Download HED"))

    def init_sam_engine(self):
        try:
            from .core.sam_engine import SAMEngine, MOBILE_SAM_AVAILABLE
        except ImportError:
            from ..core.sam_engine import SAMEngine, MOBILE_SAM_AVAILABLE

        if self.sam_engine is None:
            self.sam_engine = SAMEngine(model_type=DEFAULT_SAM_MODEL_TYPE)

        self.sam_check_btn.setVisible(True)
        self.sam_report_btn.setVisible(True)
        self.sam_download_btn.setVisible(True)
        self.sam_download_btn.setText(self._tr("⬇️ MobileSAM 다운로드 (~40MB)", "⬇️ Download MobileSAM (~40MB)"))

        if not MOBILE_SAM_AVAILABLE:
            self.sam_status.setText(self._tr("❌ PyTorch/MobileSAM 미설치", "❌ PyTorch/MobileSAM not installed"))
            self.sam_status.setStyleSheet("color: red; font-size: 10px;")
            self.install_guide.setVisible(True)
            self.install_cmd.setVisible(True)
            return

        success, _msg = self.sam_engine.load_model()
        if success:
            self.sam_status.setText(self._tr("✅ MobileSAM 로드됨 (최신 확인 가능)", "✅ MobileSAM loaded (update check available)"))
            self.sam_status.setStyleSheet("color: green; font-size: 10px;")
            self.install_guide.setVisible(False)
        else:
            self.sam_status.setText(self._tr("⚠️ 모델 파일 필요", "⚠️ Model file required"))
            self.sam_status.setStyleSheet("color: orange; font-size: 10px;")
            self.install_guide.setVisible(False)

    def download_sam(self):
        if self.sam_engine is None:
            from ..core.sam_engine import SAMEngine
            self.sam_engine = SAMEngine(model_type=DEFAULT_SAM_MODEL_TYPE)

        model_idx = self.model_combo.currentIndex()
        if model_idx == MODEL_IDX_HED:
            self.download_hed()
            return
        self.sam_download_btn.setEnabled(False)
        self.sam_status.setText(self._tr("⏬ 다운로드 중...", "⏬ Downloading..."))
        self.iface.mainWindow().repaint()
        if self.sam_engine:
            success = self.sam_engine.download_weights()
            if success:
                QMessageBox.information(self, self._tr("완료", "Done"), self._tr("MobileSAM 다운로드 완료!", "MobileSAM download complete!"))
                self.init_sam_engine()
                self.check_sam_update(show_message=False)
            else:
                QMessageBox.critical(self, self._tr("오류", "Error"), self._tr("다운로드 실패. 인터넷 연결을 확인하세요.", "Download failed. Check your internet connection."))
                self.sam_status.setText(self._tr("❌ 다운로드 실패", "❌ Download failed"))
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
        if self.sam_engine is None:
            from ..core.sam_engine import SAMEngine
            self.sam_engine = SAMEngine(model_type=DEFAULT_SAM_MODEL_TYPE)

        self.sam_check_btn.setEnabled(False)
        self.sam_status.setText(self._tr("🔎 최신 모델 확인 중...", "🔎 Checking latest model..."))
        self.iface.mainWindow().repaint()

        info = self.sam_engine.check_weights_update()
        self.sam_check_btn.setEnabled(True)

        if not info.get("ok"):
            self.sam_status.setText(self._tr("❌ 최신 확인 실패", "❌ Latest check failed"))
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
            self.sam_status.setText(
                self._tr(
                    f"⚠️ MobileSAM 없음 (원격 {remote_size})",
                    f"⚠️ MobileSAM not installed (remote {remote_size})",
                )
            )
            self.sam_download_btn.setText(
                self._tr("⬇️ MobileSAM 다운로드", "⬇️ Download MobileSAM")
            )
            return

        if status == "update_available":
            self.sam_status.setText(
                self._tr(
                    f"⬆️ MobileSAM 업데이트 가능 (로컬 {local_size} → 원격 {remote_size})",
                    f"⬆️ MobileSAM update available (local {local_size} -> remote {remote_size})",
                )
            )
            self.sam_download_btn.setText(
                self._tr("⬆️ MobileSAM 업데이트", "⬆️ Update MobileSAM")
            )
            if show_message:
                QMessageBox.information(
                    self,
                    self._tr("완료", "Done"),
                    self._tr(
                        "새 MobileSAM 모델이 있습니다.\n'업데이트' 버튼으로 바로 받을 수 있습니다.",
                        "A newer MobileSAM model is available.\nUse the update button to download it.",
                    ),
                )
            return

        if status == "up_to_date":
            self.sam_status.setText(
                self._tr(
                    f"✅ MobileSAM 최신 상태 (로컬 {local_size})",
                    f"✅ MobileSAM is up to date (local {local_size})",
                )
            )
            self.sam_download_btn.setText(
                self._tr("⬇️ MobileSAM 재다운로드", "⬇️ Re-download MobileSAM")
            )
            return

        self.sam_status.setText(
            self._tr(
                "ℹ️ 버전 비교 정보 부족 (필요 시 재다운로드 가능)",
                "ℹ️ Not enough metadata to compare versions (re-download available)",
            )
        )
        self.sam_download_btn.setText(
            self._tr("⬇️ MobileSAM 재다운로드", "⬇️ Re-download MobileSAM")
        )

    @staticmethod
    def _safe_module_version(package_name):
        try:
            import importlib.metadata as md
            return md.version(package_name)
        except Exception:
            return None

    def export_sam_report(self):
        self.sam_status.setText(self._tr("📄 SAM 리포트 생성 중...", "📄 Building SAM report..."))
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
                "PyYAML": self._safe_module_version("PyYAML"),
            },
        }

        try:
            if self.sam_engine is None:
                from ..core.sam_engine import SAMEngine
                self.sam_engine = SAMEngine(model_type=DEFAULT_SAM_MODEL_TYPE)

            update_info = self.sam_engine.check_weights_update()
            report["sam_engine"] = {
                "weights_path": getattr(self.sam_engine, "weights_path", None),
                "weights_meta_path": getattr(self.sam_engine, "weights_meta_path", None),
                "weights_url": getattr(self.sam_engine, "WEIGHTS_DOWNLOAD_URL", None),
                "local_info": self.sam_engine.get_local_weights_info(),
                "update_check": update_info,
            }

            out_path = os.path.join(tempfile.gettempdir(), "archaeotrace_sam_report.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            QApplication.clipboard().setText(json.dumps(report, ensure_ascii=False, indent=2))

            status = update_info.get("status", "unknown")
            self.sam_status.setText(
                self._tr(
                    f"✅ SAM 리포트 생성 완료 ({status})",
                    f"✅ SAM report generated ({status})",
                )
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
            self.sam_status.setText(self._tr("❌ SAM 리포트 생성 실패", "❌ Failed to build SAM report"))
            QMessageBox.critical(
                self,
                self._tr("오류", "Error"),
                self._tr(
                    "SAM 리포트 생성 실패:\n{err}",
                    "Failed to generate SAM report:\n{err}",
                ).format(err=str(e)),
            )

    def download_hed(self):
        import urllib.request
        self.sam_download_btn.setEnabled(False)
        self.sam_status.setText(self._tr("⏬ HED 다운로드 중 (56MB)...", "⏬ Downloading HED (~56MB)..."))
        self.iface.mainWindow().repaint()
        try:
            from ..core.edge_detector import EdgeDetector
            info = EdgeDetector.get_hed_download_info()
            os.makedirs(os.path.dirname(info["caffemodel_path"]), exist_ok=True)
            self.sam_status.setText(self._tr("⏬ HED 다운로드 중...", "⏬ Downloading HED..."))
            urllib.request.urlretrieve(info["caffemodel_url"], info["caffemodel_path"])
            QMessageBox.information(self, self._tr("완료", "Done"), self._tr("HED 모델 다운로드 완료!", "HED model download complete!"))
            self.check_hed_status()
        except Exception as e:
            QMessageBox.critical(self, self._tr("오류", "Error"), self._tr("HED 다운로드 실패:\n{err}", "HED download failed:\n{err}").format(err=str(e)))
            self.sam_status.setText(self._tr("❌ 다운로드 실패", "❌ Download failed"))
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
        edge_method = EDGE_METHOD_BY_MODEL.get(model_idx, DEFAULT_EDGE_METHOD)

        try:
            from ..core.edge_detector import EdgeDetector

            extent = self.iface.mapCanvas().extent()
            provider = raster.dataProvider()
            raster_ext = raster.extent()
            read_ext = extent.intersect(raster_ext)
            if read_ext.isEmpty():
                QMessageBox.warning(self, self._tr("경고", "Warning"), self._tr("래스터 범위 밖입니다.", "Current view is outside raster extent."))
                return

            raster_res = raster_ext.width() / raster.width()
            out_w = min(PREVIEW_EDGE_MAX_DIMENSION, int(read_ext.width() / raster_res))
            out_h = min(PREVIEW_EDGE_MAX_DIMENSION, int(read_ext.height() / raster_res))
            bands = []
            for b in range(1, min(MAX_RASTER_BANDS_FOR_RGB + 1, provider.bandCount() + 1)):
                block = provider.block(b, read_ext, out_w, out_h)
                if block.isValid() and block.data():
                    bands.append(np.frombuffer(block.data(), dtype=np.uint8).reshape((out_h, out_w)))
            if not bands:
                QMessageBox.warning(self, self._tr("경고", "Warning"), self._tr("래스터 데이터를 읽을 수 없습니다.", "Failed to read raster data."))
                return

            import cv2
            image = cv2.cvtColor(np.stack(bands[:3], axis=-1), cv2.COLOR_RGB2GRAY) if len(bands) >= 3 else bands[0]
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
                QMessageBox.information(self, self._tr("완료", "Done"), self._tr("'{name}' 레이어가 추가되었습니다.\n흰색=감지된 엣지", "Layer '{name}' added.\nWhite=detected edges").format(name=layer_name))
            else:
                QMessageBox.critical(self, self._tr("오류", "Error"), self._tr("미리보기 레이어 생성 실패", "Failed to create preview layer"))
        except Exception as e:
            QMessageBox.critical(self, self._tr("오류", "Error"), self._tr("엣지 감지 실패:\n{err}", "Edge detection failed:\n{err}").format(err=str(e)))

    def _help_text(self):
        if self.current_language == LANG_EN:
            return """
<h2>🏛️ ArchaeoTrace Guide</h2>
<h3>📋 Basic Workflow</h3>
<ol>
<li><b>Select Raster Map</b> - choose a scanned map with contour lines.</li>
<li><b>Create SHP Output</b> - create a new line SHP or pick an existing line layer.</li>
<li><b>Choose AI Model</b> - Canny/LSD/HED/MobileSAM depending on speed and quality.</li>
<li><b>Start Tracing</b> - click along contours and save the result.</li>
</ol>

<h3>🤖 AI Model Comparison</h3>
<table border='1' cellpadding='5'>
<tr><th>Model</th><th>Speed</th><th>Quality</th><th>Notes</th></tr>
<tr><td>🔧 Canny</td><td>Fastest</td><td>Basic</td><td>Built-in</td></tr>
<tr><td>📐 LSD</td><td>Fast</td><td>Good</td><td>Built-in</td></tr>
<tr><td>🧠 HED</td><td>Medium</td><td>High</td><td>~56MB model</td></tr>
<tr><td>🎯 MobileSAM</td><td>Slow</td><td>Best</td><td>Requires PyTorch + model file</td></tr>
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
<li>Use <b>Check MobileSAM Latest</b> before downloading to see if an update is needed.</li>
<li>Use <b>SAM Status Report</b> to create a shareable JSON report for support.</li>
</ul>

<h3>⚠️ Troubleshooting</h3>
<ul>
<li><b>No raster selected</b>: choose a raster layer in Step 1.</li>
<li><b>Model download failed</b>: check internet connection and retry.</li>
<li><b>No edges in preview</b>: zoom to map area and try another model.</li>
</ul>
"""
        return """
<h2>🏛️ ArchaeoTrace 사용 가이드</h2>
<h3>📋 기본 워크플로우</h3>
<ol>
<li><b>래스터 지도 선택</b> - 등고선이 있는 스캔 지도를 선택합니다.</li>
<li><b>SHP 출력 설정</b> - 새 라인 SHP를 만들거나 기존 라인 레이어를 선택합니다.</li>
<li><b>AI 모델 선택</b> - 속도/품질에 맞춰 Canny/LSD/HED/MobileSAM을 선택합니다.</li>
<li><b>트레이싱 시작</b> - 등고선을 따라 클릭하며 추적한 뒤 저장합니다.</li>
</ol>

<h3>🤖 AI 모델 비교</h3>
<table border='1' cellpadding='5'>
<tr><th>모델</th><th>속도</th><th>품질</th><th>비고</th></tr>
<tr><td>🔧 Canny</td><td>가장 빠름</td><td>기본</td><td>내장</td></tr>
<tr><td>📐 LSD</td><td>빠름</td><td>좋음</td><td>내장</td></tr>
<tr><td>🧠 HED</td><td>보통</td><td>우수</td><td>약 56MB 모델 필요</td></tr>
<tr><td>🎯 MobileSAM</td><td>느림</td><td>최고</td><td>PyTorch 및 모델 파일 필요</td></tr>
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
<li>다운로드 전에 <b>MobileSAM 최신 확인</b> 버튼으로 업데이트 필요 여부를 확인하세요.</li>
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
        msg.setWindowTitle(self._tr("ArchaeoTrace 도움말", "ArchaeoTrace Help"))
        msg.setTextFormat(Qt.RichText)
        msg.setText(self._help_text())
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()


# Keep old name for compatibility
AIVectorizerDialog = AIVectorizerDock
