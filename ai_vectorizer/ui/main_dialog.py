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
        step3.setToolTip("등고선을 따라 그리기 위한 AI 설정")
        step3_layout = QVBoxLayout()
        
        # AI Model selector with description
        model_desc = QLabel("💡 AI 모델: 등고선 인식 방식 선택")
        model_desc.setStyleSheet("color: gray; font-size: 10px;")
        step3_layout.addWidget(model_desc)
        
        model_layout = QHBoxLayout()
        model_label = QLabel("AI 모델:")
        model_label.setToolTip("각 모델의 장단점:\n• Canny: 가장 빠름, 기본\n• LSD: 선분 기반, 빠름\n• HED: 딥러닝, 매끄러움\n• SAM: 최고 품질 (56MB)")
        model_layout.addWidget(model_label)
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "🔧 OpenCV Canny (기본)",
            "📐 LSD 선분검출 (빠름)",
            "🧠 HED 딥러닝 (매끄러움)",
            "🎯 MobileSAM (고품질)"
        ])
        self.model_combo.setToolTip("Canny: 기본\nLSD: 선분 기반\nHED: 딥러닝 엣지\nSAM: 세그멘테이션")
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
        
        # Install guide (for SAM dependencies) - COPYABLE
        install_label = QLabel("📦 SAM 설치 (복사 가능):")
        install_label.setStyleSheet("color: #e67e22; font-size: 9px;")
        install_label.setVisible(False)
        step3_layout.addWidget(install_label)
        
        self.install_cmd = QLineEdit()
        self.install_cmd.setText("pip install torch torchvision git+https://github.com/ChaoningZhang/MobileSAM.git")
        self.install_cmd.setReadOnly(True)
        self.install_cmd.setStyleSheet("background: #fff3e0; font-size: 9px; padding: 3px;")
        self.install_cmd.setVisible(False)
        step3_layout.addWidget(self.install_cmd)
        
        self.install_guide = install_label  # Reference for visibility toggle
        
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
        self.status_label.setToolTip("현재 트레이싱 상태를 표시합니다")
        status_layout.addWidget(self.status_label)
        
        # Controls guide with better formatting
        controls_title = QLabel("📖 사용법:")
        controls_title.setStyleSheet("font-weight: bold; color: #333; margin-top: 5px;")
        status_layout.addWidget(controls_title)
        
        controls = QLabel(
            "• 좌클릭 + 드래그: 등고선 따라 그리기\n"
            "• 시작점 근처 클릭: 폴리곤 닫기 → 해발값 입력\n"
            "• 우클릭: 열린 선으로 저장\n"
            "• Esc: 마지막 점 취소 / Del: 전체 취소\n"
            "• Enter: 현재 선 저장"
        )
        controls.setStyleSheet("color: #555; font-size: 9px; background: #f8f9fa; padding: 8px; border-radius: 4px; line-height: 1.4;")
        controls.setToolTip("마우스와 키보드로 등고선을 따라 그립니다.\n폴리곤을 닫으면 해발값을 입력할 수 있습니다.")
        status_layout.addWidget(controls)
        
        status_box.setLayout(status_layout)
        self.layout.addWidget(status_box)
        
        # === Debug Tools ===
        debug_box = QGroupBox("🔧 디버그 및 도움말")
        debug_box.setToolTip("문제 해결을 위한 도구들")
        debug_layout = QVBoxLayout()
        
        self.preview_edge_btn = QPushButton("👁️ AI가 보는 엣지 미리보기")
        self.preview_edge_btn.clicked.connect(self.preview_edges)
        self.preview_edge_btn.setToolTip("현재 선택된 AI 모델이 감지하는 엣지를\n임시 래스터 레이어로 표시합니다.\n\n흰색 = AI가 인식하는 등고선")
        debug_layout.addWidget(self.preview_edge_btn)
        
        # Help button
        help_btn = QPushButton("❓ 도움말")
        help_btn.clicked.connect(self.show_help)
        help_btn.setToolTip("사용법과 문제해결 안내")
        debug_layout.addWidget(help_btn)
        
        debug_box.setLayout(debug_layout)
        self.layout.addWidget(debug_box)
        
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
                
            from ..tools.smart_trace_tool import SmartTraceTool
            
            edge_weight = self.freedom_slider.value() / 100.0
            freehand = self.freehand_check.isChecked()
            
            # Model selection: 0=Canny, 1=LSD, 2=HED, 3=SAM
            model_idx = self.model_combo.currentIndex()
            use_sam = model_idx == 3 and hasattr(self, 'sam_engine') and self.sam_engine and self.sam_engine.is_ready
            
            # Determine edge method
            edge_methods = {0: 'canny', 1: 'lsd', 2: 'hed', 3: 'canny'}
            edge_method = edge_methods.get(model_idx, 'canny')
            
            self.active_tool = SmartTraceTool(
                self.iface.mapCanvas(),
                raster,
                self.output_layer,
                edge_weight=edge_weight,
                freehand=freehand,
                sam_engine=self.sam_engine if use_sam else None,
                edge_method=edge_method
            )
            self.iface.mapCanvas().setMapTool(self.active_tool)
            self.active_tool.deactivated.connect(self.on_tool_deactivated)
            
            mode_names = {0: "Canny", 1: "LSD", 2: "HED", 3: "SAM"}
            mode_name = "SAM" if use_sam else mode_names.get(model_idx, "OpenCV")
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
        # Hide all extra controls first
        self.sam_download_btn.setVisible(False)
        self.install_guide.setVisible(False)
        self.install_cmd.setVisible(False)
        
        if index == 0:  # Canny
            self.sam_status.setText("OpenCV 내장")
            self.sam_status.setStyleSheet("color: green; font-size: 10px;")
        elif index == 1:  # LSD
            self.sam_status.setText("OpenCV 내장")
            self.sam_status.setStyleSheet("color: green; font-size: 10px;")
        elif index == 2:  # HED
            self.check_hed_status()
        elif index == 3:  # MobileSAM
            self.init_sam_engine()

    def check_hed_status(self):
        """Check if HED model is available."""
        from ..core.edge_detector import EdgeDetector
        
        if EdgeDetector.is_hed_available():
            self.sam_status.setText("✅ HED 모델 로드됨")
            self.sam_status.setStyleSheet("color: green; font-size: 10px;")
        else:
            self.sam_status.setText("⚠️ HED 모델 필요 (56MB)")
            self.sam_status.setStyleSheet("color: orange; font-size: 10px;")
            self.sam_download_btn.setVisible(True)
            self.sam_download_btn.setText("📥 HED 다운로드")

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
            self.install_cmd.setVisible(True)
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
        """Download model weights (SAM or HED based on selection)."""
        model_idx = self.model_combo.currentIndex()
        
        if model_idx == 2:  # HED
            self.download_hed()
            return
        
        # SAM download
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

    def download_hed(self):
        """Download HED model weights."""
        import os
        import urllib.request
        
        self.sam_download_btn.setEnabled(False)
        self.sam_status.setText("⏬ HED 다운로드 중 (56MB)...")
        self.iface.mainWindow().repaint()
        
        try:
            from ..core.edge_detector import EdgeDetector
            info = EdgeDetector.get_hed_download_info()
            
            # Create models directory
            models_dir = os.path.dirname(info['caffemodel_path'])
            os.makedirs(models_dir, exist_ok=True)
            
            # Download caffemodel
            self.sam_status.setText("⏬ HED 다운로드 중...")
            urllib.request.urlretrieve(
                info['caffemodel_url'],
                info['caffemodel_path']
            )
            
            QMessageBox.information(self, "완료", "HED 모델 다운로드 완료!")
            self.check_hed_status()
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"HED 다운로드 실패:\n{str(e)}")
            self.sam_status.setText("❌ 다운로드 실패")
        
        self.sam_download_btn.setEnabled(True)

    def preview_edges(self):
        """Show what the AI edge detector sees as a preview layer."""
        import os
        import tempfile
        import numpy as np
        from osgeo import gdal
        
        raster = self.layer_combo.currentLayer()
        if not raster:
            QMessageBox.warning(self, "경고", "래스터 지도를 먼저 선택하세요.")
            return
        
        # Get model method
        model_idx = self.model_combo.currentIndex()
        edge_methods = {0: 'canny', 1: 'lsd', 2: 'hed', 3: 'canny'}
        edge_method = edge_methods.get(model_idx, 'canny')
        
        try:
            from ..core.edge_detector import EdgeDetector
            
            # Read current view extent
            extent = self.iface.mapCanvas().extent()
            provider = raster.dataProvider()
            raster_ext = raster.extent()
            read_ext = extent.intersect(raster_ext)
            
            if read_ext.isEmpty():
                QMessageBox.warning(self, "경고", "래스터 범위 밖입니다.")
                return
            
            # Read raster
            raster_res = raster_ext.width() / raster.width()
            out_w = min(800, int(read_ext.width() / raster_res))
            out_h = min(800, int(read_ext.height() / raster_res))
            
            bands = []
            for b in range(1, min(4, provider.bandCount() + 1)):
                block = provider.block(b, read_ext, out_w, out_h)
                if block.isValid() and block.data():
                    arr = np.frombuffer(block.data(), dtype=np.uint8).reshape((out_h, out_w))
                    bands.append(arr)
            
            if not bands:
                QMessageBox.warning(self, "경고", "래스터 데이터를 읽을 수 없습니다.")
                return
            
            # Convert to grayscale
            import cv2
            if len(bands) >= 3:
                image = cv2.cvtColor(np.stack(bands[:3], axis=-1), cv2.COLOR_RGB2GRAY)
            else:
                image = bands[0]
            
            # Detect edges
            detector = EdgeDetector(method=edge_method)
            edges = detector.detect_edges(image)
            
            # Save as temporary GeoTiff
            temp_path = os.path.join(tempfile.gettempdir(), f"edge_preview_{edge_method}.tif")
            
            driver = gdal.GetDriverByName('GTiff')
            ds = driver.Create(temp_path, out_w, out_h, 1, gdal.GDT_Byte)
            ds.SetGeoTransform([
                read_ext.xMinimum(), 
                read_ext.width() / out_w, 0,
                read_ext.yMaximum(), 
                0, -read_ext.height() / out_h
            ])
            ds.SetProjection(raster.crs().toWkt())
            ds.GetRasterBand(1).WriteArray(edges)
            ds = None  # Close
            
            # Load as layer
            from qgis.core import QgsRasterLayer
            layer_name = f"Edge Preview ({edge_method.upper()})"
            edge_layer = QgsRasterLayer(temp_path, layer_name)
            if edge_layer.isValid():
                QgsProject.instance().addMapLayer(edge_layer)
                QMessageBox.information(self, "완료", f"'{layer_name}' 레이어가 추가되었습니다.\n흰색=감지된 엣지")
            else:
                QMessageBox.critical(self, "오류", "미리보기 레이어 생성 실패")
                
        except Exception as e:
            QMessageBox.critical(self, "오류", f"엣지 감지 실패:\n{str(e)}")

    def show_help(self):
        """Show comprehensive help dialog."""
        help_text = """
<h2>🏛️ ArchaeoTrace 사용 가이드</h2>

<h3>📋 기본 워크플로우</h3>
<ol>
<li><b>래스터 지도 선택</b> - 등고선이 있는 스캔 지도</li>
<li><b>SHP 파일 생성</b> - 결과를 저장할 파일</li>
<li><b>AI 모델 선택</b> - 아래 설명 참고</li>
<li><b>트레이싱 시작</b> - 등고선 따라 그리기</li>
</ol>

<h3>🤖 AI 모델 비교</h3>
<table border='1' cellpadding='5'>
<tr><th>모델</th><th>속도</th><th>품질</th><th>크기</th></tr>
<tr><td>🔧 Canny</td><td>⚡최고</td><td>기본</td><td>내장</td></tr>
<tr><td>📐 LSD</td><td>⚡빠름</td><td>좋음</td><td>내장</td></tr>
<tr><td>🧠 HED</td><td>보통</td><td>우수</td><td>56MB</td></tr>
<tr><td>🎯 SAM</td><td>느림</td><td>최고</td><td>설치필요</td></tr>
</table>

<h3>🖱️ 조작법</h3>
<ul>
<li><b>좌클릭 + 드래그</b>: 등고선 따라 그리기</li>
<li><b>시작점 근처 클릭</b>: 폴리곤 닫기 → 해발값 입력</li>
<li><b>우클릭</b>: 열린 선으로 저장</li>
<li><b>Esc</b>: 마지막 점 취소</li>
<li><b>Del</b>: 전체 취소</li>
<li><b>Enter</b>: 현재 선 저장</li>
</ul>

<h3>💡 팁</h3>
<ul>
<li>줌 레벨을 적절히 조절하세요 (등고선이 3~5픽셀 두께가 최적)</li>
<li>AI 강도를 높이면 엣지를 더 강하게 따라갑니다</li>
<li>프리핸드 모드는 AI 없이 순수 마우스 추적입니다</li>
<li>엣지 미리보기로 AI가 인식하는 선을 확인하세요</li>
</ul>

<h3>⚠️ 문제 해결</h3>
<ul>
<li><b>선이 자글자글</b>: AI 강도를 낮추거나, 천천히 그리세요</li>
<li><b>AI가 선을 못 인식</b>: 엣지 미리보기로 확인, 다른 모델 시도</li>
<li><b>폴리곤 안 닫힘</b>: 노란 시작점 근처에서 클릭하세요</li>
</ul>
"""
        msg = QMessageBox(self)
        msg.setWindowTitle("ArchaeoTrace 도움말")
        msg.setTextFormat(Qt.RichText)
        msg.setText(help_text)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()


# Keep old name for compatibility
AIVectorizerDialog = AIVectorizerDock
