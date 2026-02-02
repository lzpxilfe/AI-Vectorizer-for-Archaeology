# -*- coding: utf-8 -*-
"""
ArchaeoTrace - AI-assisted contour tracing for historical maps
Main dialog with guided workflow
"""
import os
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox, 
    QPushButton, QFormLayout, QMessageBox, QGroupBox, QFileDialog,
    QLineEdit, QSpinBox
)
from qgis.core import (
    QgsProject, QgsMapLayerProxyModel, QgsVectorLayer,
    QgsField, QgsVectorFileWriter, QgsCoordinateReferenceSystem
)
from qgis.gui import QgsMapLayerComboBox
from qgis.PyQt.QtCore import Qt, QVariant

class AIVectorizerDialog(QDialog):
    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.setWindowTitle("ArchaeoTrace - 등고선 벡터화")
        self.resize(420, 500)
        
        self.active_tool = None
        self.output_layer = None
        
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.setup_ui()
        
    def setup_ui(self):
        # === Step 1: Input Map ===
        step1 = QGroupBox("1️⃣ 입력 지도 선택")
        step1_layout = QVBoxLayout()
        self.layer_combo = QgsMapLayerComboBox()
        self.layer_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        step1_layout.addWidget(self.layer_combo)
        step1.setLayout(step1_layout)
        self.layout.addWidget(step1)
        
        # === Step 2: Create Output SHP ===
        step2 = QGroupBox("2️⃣ 출력 SHP 파일 생성")
        step2_layout = QVBoxLayout()
        
        # File path
        path_layout = QHBoxLayout()
        self.shp_path = QLineEdit()
        self.shp_path.setPlaceholderText("저장할 SHP 파일 경로...")
        browse_btn = QPushButton("찾아보기")
        browse_btn.clicked.connect(self.browse_shp)
        path_layout.addWidget(self.shp_path)
        path_layout.addWidget(browse_btn)
        step2_layout.addLayout(path_layout)
        
        self.create_shp_btn = QPushButton("📁 SHP 파일 생성")
        self.create_shp_btn.clicked.connect(self.create_shp_layer)
        self.create_shp_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        step2_layout.addWidget(self.create_shp_btn)
        
        # Or select existing
        step2_layout.addWidget(QLabel("또는 기존 레이어 선택:"))
        self.vector_combo = QgsMapLayerComboBox()
        self.vector_combo.setFilters(QgsMapLayerProxyModel.LineLayer)
        self.vector_combo.layerChanged.connect(self.on_layer_selected)
        step2_layout.addWidget(self.vector_combo)
        
        step2.setLayout(step2_layout)
        self.layout.addWidget(step2)
        
        # === Step 3: Trace Lines ===
        step3 = QGroupBox("3️⃣ 등고선 트레이싱")
        step3_layout = QVBoxLayout()
        
        # Freedom slider
        from qgis.PyQt.QtWidgets import QSlider
        freedom_layout = QHBoxLayout()
        freedom_layout.addWidget(QLabel("자유도:"))
        self.freedom_slider = QSlider(Qt.Horizontal)
        self.freedom_slider.setMinimum(0)
        self.freedom_slider.setMaximum(100)
        self.freedom_slider.setValue(50)  # Default: balanced
        self.freedom_slider.setToolTip("낮음=엣지 따라감 | 높음=자유롭게 그리기")
        freedom_layout.addWidget(self.freedom_slider)
        self.freedom_label = QLabel("50%")
        self.freedom_slider.valueChanged.connect(lambda v: self.freedom_label.setText(f"{v}%"))
        freedom_layout.addWidget(self.freedom_label)
        step3_layout.addLayout(freedom_layout)
        
        self.trace_btn = QPushButton("🖊️ 선 그리기 시작")
        self.trace_btn.setCheckable(True)
        self.trace_btn.clicked.connect(self.toggle_trace_tool)
        self.trace_btn.setStyleSheet("font-weight: bold; padding: 10px; font-size: 14px;")
        self.trace_btn.setEnabled(False)
        step3_layout.addWidget(self.trace_btn)
        
        # Status
        self.status_label = QLabel("SHP 파일을 먼저 생성하거나 선택하세요")
        self.status_label.setStyleSheet("color: #666;")
        step3_layout.addWidget(self.status_label)
        
        step3.setLayout(step3_layout)
        self.layout.addWidget(step3)
        
        # === Step 4: Point Mode (Optional) ===  
        step4 = QGroupBox("4️⃣ 포인트 모드 (선택사항)")
        step4_layout = QVBoxLayout()
        
        self.point_btn = QPushButton("📍 포인트 찍기 모드")
        self.point_btn.setCheckable(True)
        self.point_btn.clicked.connect(self.toggle_point_mode)
        self.point_btn.setEnabled(False)
        step4_layout.addWidget(self.point_btn)
        
        step4_layout.addWidget(QLabel("표고점 등을 추가할 수 있습니다"))
        step4.setLayout(step4_layout)
        self.layout.addWidget(step4)
        
        # === Controls Guide ===
        guide = QLabel(
            "조작법: 좌클릭=점추가 | 우클릭=완료 | Esc=취소 | Del=전체삭제\n"
            "🟢 초록색=미리보기 | 🔴 빨간색=확정된 선"
        )
        guide.setStyleSheet("color: gray; font-size: 10px; padding: 5px;")
        self.layout.addWidget(guide)

    def browse_shp(self):
        """Open file dialog to select SHP save location."""
        path, _ = QFileDialog.getSaveFileName(
            self, "SHP 파일 저장 위치", "", "Shapefile (*.shp)"
        )
        if path:
            if not path.endswith('.shp'):
                path += '.shp'
            self.shp_path.setText(path)

    def create_shp_layer(self):
        """Create a new shapefile for output."""
        path = self.shp_path.text()
        if not path:
            QMessageBox.warning(self, "경고", "파일 경로를 지정해주세요.")
            return
        
        # Get CRS from input raster
        raster = self.layer_combo.currentLayer()
        if raster:
            crs = raster.crs()
        else:
            crs = QgsCoordinateReferenceSystem("EPSG:4326")
        
        # Create shapefile
        fields = [
            QgsField("id", QVariant.Int),
            QgsField("elevation", QVariant.Double)
        ]
        
        layer = QgsVectorLayer(f"LineString?crs={crs.authid()}", "Contours", "memory")
        layer.dataProvider().addAttributes(fields)
        layer.updateFields()
        
        # Save to file
        error = QgsVectorFileWriter.writeAsVectorFormat(
            layer, path, "UTF-8", crs, "ESRI Shapefile"
        )
        
        if error[0] == QgsVectorFileWriter.NoError:
            # Load the saved file
            name = os.path.basename(path).replace('.shp', '')
            self.output_layer = QgsVectorLayer(path, name, "ogr")
            QgsProject.instance().addMapLayer(self.output_layer)
            self.vector_combo.setLayer(self.output_layer)
            self.enable_tracing()
            QMessageBox.information(self, "성공", f"SHP 파일이 생성되었습니다:\n{path}")
        else:
            QMessageBox.critical(self, "오류", f"파일 생성 실패: {error[1]}")

    def on_layer_selected(self, layer):
        """When user selects existing layer."""
        if layer:
            self.output_layer = layer
            self.enable_tracing()

    def enable_tracing(self):
        """Enable tracing buttons after output is set."""
        self.trace_btn.setEnabled(True)
        self.point_btn.setEnabled(True)
        self.status_label.setText("준비 완료! '선 그리기 시작' 버튼을 누르세요")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")

    def toggle_trace_tool(self, checked):
        """Toggle line tracing tool."""
        if checked:
            raster = self.layer_combo.currentLayer()
            if not raster:
                QMessageBox.warning(self, "경고", "래스터 지도를 선택하세요.")
                self.trace_btn.setChecked(False)
                return
                
            from ..tools.smart_trace_tool import SmartTraceTool
            
            # Convert slider (0-100) to edge_weight (1.0 to 0.0)
            # 0% freedom = 1.0 edge weight (strict follow)
            # 100% freedom = 0.0 edge weight (free draw)
            freedom = 1.0 - (self.freedom_slider.value() / 100.0)
            
            self.active_tool = SmartTraceTool(
                self.iface.mapCanvas(),
                raster,
                self.output_layer,
                edge_weight=freedom
            )
            self.iface.mapCanvas().setMapTool(self.active_tool)
            self.active_tool.deactivated.connect(self.on_tool_deactivated)
            
            self.status_label.setText("🖊️ 트레이싱 모드 - 등고선 위를 클릭하세요")
            self.trace_btn.setText("⏹️ 그리기 중지")
            self.point_btn.setChecked(False)
        else:
            if self.active_tool:
                self.iface.mapCanvas().unsetMapTool(self.active_tool)
            self.status_label.setText("준비 완료")
            self.trace_btn.setText("🖊️ 선 그리기 시작")

    def toggle_point_mode(self, checked):
        """Toggle point digitizing mode (for elevation points)."""
        if checked:
            # Use QGIS default point digitizing
            from qgis.gui import QgsMapToolDigitizeFeature
            
            # Create point layer if needed
            raster = self.layer_combo.currentLayer()
            crs = raster.crs().authid() if raster else "EPSG:4326"
            
            point_layer = QgsVectorLayer(f"Point?crs={crs}", "Elevation Points", "memory")
            point_layer.dataProvider().addAttributes([
                QgsField("id", QVariant.Int),
                QgsField("elevation", QVariant.Double)
            ])
            point_layer.updateFields()
            QgsProject.instance().addMapLayer(point_layer)
            
            self.iface.setActiveLayer(point_layer)
            point_layer.startEditing()
            self.iface.actionAddFeature().trigger()
            
            self.status_label.setText("📍 포인트 모드 - 지도를 클릭하여 점 추가")
            self.point_btn.setText("⏹️ 포인트 모드 중지")
            self.trace_btn.setChecked(False)
        else:
            self.status_label.setText("준비 완료")
            self.point_btn.setText("📍 포인트 찍기 모드")

    def on_tool_deactivated(self):
        self.trace_btn.setChecked(False)
        self.trace_btn.setText("🖊️ 선 그리기 시작")
        self.status_label.setText("준비 완료")
        self.active_tool = None
