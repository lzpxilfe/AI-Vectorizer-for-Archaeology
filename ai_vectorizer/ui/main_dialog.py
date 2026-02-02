# -*- coding: utf-8 -*-
import os
from qgis.PyQt import uic
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QComboBox, QCheckBox, 
    QPushButton, QFormLayout, QMessageBox, QProgressBar
)
from qgis.core import QgsProject, QgsMapLayerProxyModel
from qgis.gui import QgsMapLayerComboBox
from qgis.PyQt.QtCore import Qt

# Import engines
from ..core.sam_engine import SAMEngine

class AIVectorizerDialog(QDialog):
    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.setWindowTitle("AI Vectorizer")
        self.resize(400, 350)
        
        self.sam_engine = None
        self.active_tool = None
        
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # UI Elements
        self.setup_ui()
        
    def setup_ui(self):
        # 1. Raster Layer Selection
        self.layout.addWidget(QLabel("Target Raster Layer (Old Map):"))
        self.layer_combo = QgsMapLayerComboBox(self)
        self.layer_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.layout.addWidget(self.layer_combo)
        
        # 2. Vector Layer Selection (Output)
        self.layout.addWidget(QLabel("Target Vector Layer (Contours):"))
        self.vector_combo = QgsMapLayerComboBox(self)
        self.vector_combo.setFilters(QgsMapLayerProxyModel.LineLayer)
        self.layout.addWidget(self.vector_combo)
        
        # 3. AI Model Selection
        self.layout.addWidget(QLabel("AI Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Lite (OpenCV Edge Detection)",
            "Standard (MobileSAM)",
            "Pro (Full SAM) - Coming Soon"
        ])
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        self.layout.addWidget(self.model_combo)
        
        # Model Status & Download
        self.model_status_label = QLabel("")
        self.model_status_label.setStyleSheet("color: gray;")
        self.layout.addWidget(self.model_status_label)
        
        self.download_btn = QPushButton("Download MobileSAM Model")
        self.download_btn.clicked.connect(self.download_model)
        self.download_btn.setVisible(False)
        self.layout.addWidget(self.download_btn)

        # 4. Settings
        settings_layout = QFormLayout()
        self.check_snap = QCheckBox("Enable Edge Snapping")
        self.check_snap.setChecked(True)
        self.check_smooth = QCheckBox("Smooth Lines")
        self.check_smooth.setChecked(True)
        settings_layout.addRow(self.check_snap)
        settings_layout.addRow(self.check_smooth)
        self.layout.addLayout(settings_layout)
        
        # 5. Tools Actions
        self.tool_btn = QPushButton("Activate Smart Trace Tool")
        self.tool_btn.setCheckable(True)
        self.tool_btn.clicked.connect(self.toggle_tool)
        self.layout.addWidget(self.tool_btn)
        
        # Status
        self.status_label = QLabel("Ready")
        self.layout.addWidget(self.status_label)

    def on_model_changed(self, index):
        if index == 1: # Standard (MobileSAM)
            self.init_sam_engine()
        else:
            self.download_btn.setVisible(False)
            self.model_status_label.setText("")

    def init_sam_engine(self):
        if not self.sam_engine:
            self.sam_engine = SAMEngine(model_type="vit_t")
            
        success, msg = self.sam_engine.load_model()
        if success:
            self.model_status_label.setText("MobileSAM Loaded ✅")
            self.model_status_label.setStyleSheet("color: green;")
            self.download_btn.setVisible(False)
        else:
            self.model_status_label.setText(f"Model Missing: {msg}")
            self.model_status_label.setStyleSheet("color: red;")
            if "weights not found" in msg or "not found" in msg:
                self.download_btn.setVisible(True)

    def download_model(self):
        self.status_label.setText("Downloading model... Please wait.")
        self.download_btn.setEnabled(False)
        self.iface.mainWindow().repaint()
        
        if self.sam_engine:
            success = self.sam_engine.download_weights()
            if success:
                QMessageBox.information(self, "Success", "Model downloaded successfully!")
                self.init_sam_engine() # Reload
                self.status_label.setText("Ready")
            else:
                QMessageBox.critical(self, "Error", "Download failed. Check internet connection.")
                self.status_label.setText("Download failed")
        
        self.download_btn.setEnabled(True)

    def toggle_tool(self, checked):
        if checked:
            raster_layer = self.layer_combo.currentLayer()
            vector_layer = self.vector_combo.currentLayer()
            model_idx = self.model_combo.currentIndex()
            
            if not raster_layer:
                QMessageBox.warning(self, "Warning", "Please select a raster layer first.")
                self.tool_btn.setChecked(False)
                return
            
            if model_idx == 1 and not (self.sam_engine and self.sam_engine.is_ready):
                 QMessageBox.warning(self, "Warning", "MobileSAM model is not loaded. Please download it first.")
                 self.tool_btn.setChecked(False)
                 return

            # Import tool here to avoid circular imports
            from ..tools.smart_trace_tool import SmartTraceTool
            
            # Initialize tool
            self.active_tool = SmartTraceTool(
                self.iface.mapCanvas(),
                raster_layer,
                vector_layer,
                model_type=model_idx,
                sam_engine=self.sam_engine
            )
            
            self.iface.mapCanvas().setMapTool(self.active_tool)
            self.status_label.setText("Tools: Click start point -> Click end point")
            
            # Connect tool signals
            self.active_tool.deactivated.connect(self.on_tool_deactivated)
            
        else:
            if self.active_tool:
                self.iface.mapCanvas().unsetMapTool(self.active_tool)
                self.active_tool = None
            self.status_label.setText("Tool Deactivated")

    def on_tool_deactivated(self):
        self.tool_btn.setChecked(False)
        self.status_label.setText("Ready")
        self.active_tool = None
