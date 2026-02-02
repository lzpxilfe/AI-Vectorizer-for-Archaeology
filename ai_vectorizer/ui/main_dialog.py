# -*- coding: utf-8 -*-
"""
ArchaeoTrace - AI-assisted contour tracing for historical maps
Dockable panel with guided workflow and tooltips
"""
import os
from qgis.PyQt.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox, 
    QPushButton, QGroupBox, QFileDialog, QLineEdit, QSlider, QMessageBox
)
from qgis.core import (
    QgsProject, QgsMapLayerProxyModel, QgsVectorLayer,
    QgsField, QgsVectorFileWriter, QgsCoordinateReferenceSystem
)
from qgis.gui import QgsMapLayerComboBox
from qgis.PyQt.QtCore import Qt, QVariant

class AIVectorizerDock(QDockWidget):
    """Dockable panel for ArchaeoTrace plugin."""
    
    def __init__(self, iface, parent=None):
        super().__init__("ArchaeoTrace", parent)
        self.iface = iface
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        self.active_tool = None
        self.output_layer = None
        
        # Main widget
        main_widget = QWidget()
        self.layout = QVBoxLayout()
        main_widget.setLayout(self.layout)
        self.setWidget(main_widget)
        
        self.setup_ui()
        
    def setup_ui(self):
        # === Header ===
        header = QLabel("🏛️ ArchaeoTrace - 고지도 등고선 벡터화")
        header.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px; background: #2c3e50; color: white; border-radius: 3px;")
        self.layout.addWidget(header)
        
        # === Step 1: Input Map ===
        step1 = QGroupBox("1️⃣ 입력 지도")
        step1.setToolTip("벡터화할 래스터 지도를 선택하세요")
        step1_layout = QVBoxLayout()
        
        step1_desc = QLabel("💡 등고선이 있는 스캔 지도 선택")
        step1_desc.setStyleSheet("color: gray; font-size: 10px;")
        step1_layout.addWidget(step1_desc)
        
        self.layer_combo = QgsMapLayerComboBox()
        self.layer_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.layer_combo.setToolTip("QGIS에 로드된 래스터 레이어 중 선택")
        step1_layout.addWidget(self.layer_combo)
        step1.setLayout(step1_layout)
        self.layout.addWidget(step1)
        
        # === Step 2: Output SHP ===
        step2 = QGroupBox("2️⃣ 출력 파일")
        step2.setToolTip("등고선을 저장할 Shapefile 생성 또는 선택")
        step2_layout = QVBoxLayout()
        
        step2_desc = QLabel("💡 새 SHP 생성 또는 기존 레이어 선택")
        step2_desc.setStyleSheet("color: gray; font-size: 10px;")
        step2_layout.addWidget(step2_desc)
        
        # File path
        path_layout = QHBoxLayout()
        self.shp_path = QLineEdit()
        self.shp_path.setPlaceholderText("저장할 SHP 파일 경로...")
        browse_btn = QPushButton("📂")
        browse_btn.setFixedWidth(30)
        browse_btn.setToolTip("파일 위치 찾기")
        browse_btn.clicked.connect(self.browse_shp)
        path_layout.addWidget(self.shp_path)
        path_layout.addWidget(browse_btn)
        step2_layout.addLayout(path_layout)
        
        self.create_shp_btn = QPushButton("📁 새 SHP 생성")
        self.create_shp_btn.clicked.connect(self.create_shp_layer)
        self.create_shp_btn.setToolTip("지정한 경로에 새 Shapefile을 생성합니다")
        step2_layout.addWidget(self.create_shp_btn)
        
        step2_layout.addWidget(QLabel("또는 기존 라인 레이어:"))
        self.vector_combo = QgsMapLayerComboBox()
        self.vector_combo.setFilters(QgsMapLayerProxyModel.LineLayer)
        self.vector_combo.layerChanged.connect(self.on_layer_selected)
        self.vector_combo.setToolTip("이미 있는 라인 레이어에 추가")
        step2_layout.addWidget(self.vector_combo)
        
        step2.setLayout(step2_layout)
        self.layout.addWidget(step2)
        
        # === Step 3: Tracing Options ===
        step3 = QGroupBox("3️⃣ 트레이싱 설정")
        step3_layout = QVBoxLayout()
        
        # AI Model selector
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("AI 모델:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "🔧 OpenCV (기본, 빠름)",
            "🧠 MobileSAM (고품질, 느림)"
        ])
        self.model_combo.setToolTip("OpenCV: 설치 불필요\nMobileSAM: 딥러닝 기반, 더 정확")
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_combo)
        step3_layout.addLayout(model_layout)
        
        # SAM status & download
        self.sam_status = QLabel("")
        self.sam_status.setStyleSheet("font-size: 10px;")
        step3_layout.addWidget(self.sam_status)
        
        self.sam_download_btn = QPushButton("⬇️ MobileSAM 다운로드 (~40MB)")
        self.sam_download_btn.clicked.connect(self.download_sam)
        self.sam_download_btn.setVisible(False)
        self.sam_download_btn.setToolTip("인터넷 연결 필요. 최초 1회만 다운로드")
        step3_layout.addWidget(self.sam_download_btn)
        
        # Install guide (for SAM dependencies)
        self.install_guide = QLabel(
            "📦 SAM 사용을 위해 설치 필요:\n"
            "pip install torch torchvision mobile-sam"
        )
        self.install_guide.setStyleSheet("color: #e67e22; font-size: 9px; background: #fff3e0; padding: 5px; border-radius: 3px;")
        self.install_guide.setVisible(False)
        self.install_guide.setWordWrap(True)
        step3_layout.addWidget(self.install_guide)
        
        # Freehand checkbox
        self.freehand_check = QCheckBox("✏️ 프리핸드 (AI 비활성)")
        self.freehand_check.setToolTip("체크: AI 없이 순수 마우스 추적")
        step3_layout.addWidget(self.freehand_check)
        
        # Edge strength slider
        edge_layout = QHBoxLayout()
        edge_label = QLabel("AI 강도:")
        edge_layout.addWidget(edge_label)
        
        self.freedom_slider = QSlider(Qt.Horizontal)
        self.freedom_slider.setMinimum(0)
        self.freedom_slider.setMaximum(100)
        self.freedom_slider.setValue(30)
        self.freedom_slider.setToolTip("0%: 자유롭게\n100%: 엣지 따라감")
        edge_layout.addWidget(self.freedom_slider)
        
        self.freedom_label = QLabel("30%")
        self.freedom_slider.valueChanged.connect(lambda v: self.freedom_label.setText(f"{v}%"))
        edge_layout.addWidget(self.freedom_label)
        step3_layout.addLayout(edge_layout)
        
        # Start button
        self.trace_btn = QPushButton("🖊️ 트레이싱 시작")
        self.trace_btn.setCheckable(True)
        self.trace_btn.clicked.connect(self.toggle_trace_tool)
        self.trace_btn.setStyleSheet("font-weight: bold; padding: 8px; background: #27ae60; color: white;")
        self.trace_btn.setEnabled(False)
        self.trace_btn.setToolTip("클릭하여 트레이싱 시작")
        step3_layout.addWidget(self.trace_btn)
        
        step3.setLayout(step3_layout)
        self.layout.addWidget(step3)
        
        # === Status & Controls ===
        status_box = QGroupBox("📋 상태")
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("SHP 파일을 먼저 생성하세요")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        
        # Controls guide
        controls = QLabel(
            "🖱️ 좌클릭: 점 추가\n"
            "🖱️ 우클릭: 완료 저장\n"
            "⌨️ Esc: 마지막 취소\n"
            "⌨️ Del: 전체 취소\n"
            "🔵 청록원 근처 클릭: 폴리곤 닫기"
        )
        controls.setStyleSheet("color: #666; font-size: 9px; background: #f5f5f5; padding: 5px; border-radius: 3px;")
        status_layout.addWidget(controls)
        
        status_box.setLayout(status_layout)
        self.layout.addWidget(status_box)
        
        # Add stretch to push everything up
        self.layout.addStretch()

    def browse_shp(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "SHP 파일 저장 위치", "", "Shapefile (*.shp)"
        )
        if path:
            if not path.endswith('.shp'):
                path += '.shp'
            self.shp_path.setText(path)

    def create_shp_layer(self):
        path = self.shp_path.text()
        if not path:
            QMessageBox.warning(self, "경고", "파일 경로를 지정해주세요.")
            return
        
        raster = self.layer_combo.currentLayer()
        crs = raster.crs() if raster else QgsCoordinateReferenceSystem("EPSG:4326")
        
        fields = [
            QgsField("id", QVariant.Int),
            QgsField("elevation", QVariant.Double)
        ]
        
        layer = QgsVectorLayer(f"LineString?crs={crs.authid()}", "Contours", "memory")
        layer.dataProvider().addAttributes(fields)
        layer.updateFields()
        
        error = QgsVectorFileWriter.writeAsVectorFormat(
            layer, path, "UTF-8", crs, "ESRI Shapefile"
        )
        
        if error[0] == QgsVectorFileWriter.NoError:
            name = os.path.basename(path).replace('.shp', '')
            self.output_layer = QgsVectorLayer(path, name, "ogr")
            QgsProject.instance().addMapLayer(self.output_layer)
            self.vector_combo.setLayer(self.output_layer)
            self.enable_tracing()
            QMessageBox.information(self, "성공", f"SHP 생성 완료:\n{path}")
        else:
            QMessageBox.critical(self, "오류", f"생성 실패: {error[1]}")

    def on_layer_selected(self, layer):
        if layer:
            self.output_layer = layer
            self.enable_tracing()

    def enable_tracing(self):
        self.trace_btn.setEnabled(True)
        self.status_label.setText("✅ 준비 완료! 트레이싱을 시작하세요")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")

    def toggle_trace_tool(self, checked):
        if checked:
            raster = self.layer_combo.currentLayer()
            if not raster:
                QMessageBox.warning(self, "경고", "래스터 지도를 선택하세요.")
                self.trace_btn.setChecked(False)
                return
                
            from .tools.smart_trace_tool import SmartTraceTool
            
            edge_weight = self.freedom_slider.value() / 100.0
            freehand = self.freehand_check.isChecked()
            use_sam = self.model_combo.currentIndex() == 1 and hasattr(self, 'sam_engine') and self.sam_engine and self.sam_engine.is_ready
            
            self.active_tool = SmartTraceTool(
                self.iface.mapCanvas(),
                raster,
                self.output_layer,
                edge_weight=edge_weight,
                freehand=freehand,
                sam_engine=self.sam_engine if use_sam else None
            )
            self.iface.mapCanvas().setMapTool(self.active_tool)
            self.active_tool.deactivated.connect(self.on_tool_deactivated)
            
            mode_name = "SAM" if use_sam else "OpenCV"
            self.status_label.setText(f"🖊️ [{mode_name}] 등고선을 클릭하세요")
            self.trace_btn.setText("⏹️ 중지")
            self.trace_btn.setStyleSheet("font-weight: bold; padding: 8px; background: #e74c3c; color: white;")
        else:
            if self.active_tool:
                self.iface.mapCanvas().unsetMapTool(self.active_tool)
            self.status_label.setText("✅ 준비 완료")
            self.trace_btn.setText("🖊️ 트레이싱 시작")
            self.trace_btn.setStyleSheet("font-weight: bold; padding: 8px; background: #27ae60; color: white;")

    def on_tool_deactivated(self):
        self.trace_btn.setChecked(False)
        self.trace_btn.setText("🖊️ 트레이싱 시작")
        self.trace_btn.setStyleSheet("font-weight: bold; padding: 8px; background: #27ae60; color: white;")
        self.status_label.setText("✅ 준비 완료")
        self.active_tool = None

    def on_model_changed(self, index):
        """Handle AI model selection change."""
        if index == 1:  # MobileSAM
            self.init_sam_engine()
        else:
            self.sam_status.setText("")
            self.sam_download_btn.setVisible(False)
            self.install_guide.setVisible(False)

    def init_sam_engine(self):
        """Initialize SAM engine."""
        try:
            from .core.sam_engine import SAMEngine, MOBILE_SAM_AVAILABLE
        except ImportError:
            from ..core.sam_engine import SAMEngine, MOBILE_SAM_AVAILABLE
        
        if not MOBILE_SAM_AVAILABLE:
            self.sam_status.setText("❌ PyTorch/MobileSAM 미설치")
            self.sam_status.setStyleSheet("color: red; font-size: 10px;")
            self.install_guide.setVisible(True)
            self.sam_download_btn.setVisible(False)
            return
        
        if not hasattr(self, 'sam_engine') or self.sam_engine is None:
            self.sam_engine = SAMEngine(model_type="vit_t")
        
        success, msg = self.sam_engine.load_model()
        if success:
            self.sam_status.setText("✅ MobileSAM 로드됨")
            self.sam_status.setStyleSheet("color: green; font-size: 10px;")
            self.sam_download_btn.setVisible(False)
            self.install_guide.setVisible(False)
        else:
            self.sam_status.setText("⚠️ 모델 파일 필요")
            self.sam_status.setStyleSheet("color: orange; font-size: 10px;")
            self.sam_download_btn.setVisible(True)
            self.install_guide.setVisible(False)

    def download_sam(self):
        """Download MobileSAM weights."""
        self.sam_download_btn.setEnabled(False)
        self.sam_status.setText("⏬ 다운로드 중...")
        self.iface.mainWindow().repaint()
        
        if hasattr(self, 'sam_engine') and self.sam_engine:
            success = self.sam_engine.download_weights()
            if success:
                QMessageBox.information(self, "완료", "MobileSAM 다운로드 완료!")
                self.init_sam_engine()
            else:
                QMessageBox.critical(self, "오류", "다운로드 실패. 인터넷 연결을 확인하세요.")
                self.sam_status.setText("❌ 다운로드 실패")
        
        self.sam_download_btn.setEnabled(True)


# Keep old name for compatibility
AIVectorizerDialog = AIVectorizerDock
