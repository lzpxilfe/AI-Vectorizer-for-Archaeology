# -*- coding: utf-8 -*-
"""
Magnetic Trace Tool - Edge-following contour tracing with undo support
"""
import numpy as np
import cv2
from qgis.gui import QgsMapToolEmitPoint, QgsRubberBand
from qgis.core import (
    QgsWkbTypes, QgsProject, QgsPointXY, QgsGeometry, 
    QgsFeature, QgsCoordinateTransform, QgsRectangle,
    QgsVectorLayer, QgsField
)
from qgis.PyQt.QtCore import Qt, QVariant
from qgis.PyQt.QtGui import QColor

from ..core.edge_detector import EdgeDetector
from ..core.path_finder import PathFinder

class SmartTraceTool(QgsMapToolEmitPoint):
    def __init__(self, canvas, raster_layer, vector_layer, model_type=0, sam_engine=None, edge_weight=0.5):
        self.canvas = canvas
        super().__init__(self.canvas)
        
        self.raster_layer = raster_layer
        self.vector_layer = vector_layer
        self.model_type = model_type  
        self.sam_engine = sam_engine
        self.edge_weight = edge_weight  # 0.0 = free, 1.0 = strict edge follow
        
        # Path tracking
        self.path_points = []  # Confirmed points (map coords)
        self.is_tracing = False
        
        # RubberBands
        self.preview_band = QgsRubberBand(self.canvas, QgsWkbTypes.LineGeometry)
        self.preview_band.setColor(QColor(0, 200, 0, 200))
        self.preview_band.setWidth(3)
        
        self.confirm_band = QgsRubberBand(self.canvas, QgsWkbTypes.LineGeometry)
        self.confirm_band.setColor(QColor(255, 50, 50, 255))
        self.confirm_band.setWidth(3)
        
        # Start point marker
        self.start_marker = QgsRubberBand(self.canvas, QgsWkbTypes.PointGeometry)
        self.start_marker.setColor(QColor(255, 255, 0, 255))
        self.start_marker.setWidth(8)
        
        # AI engines
        self.edge_detector = EdgeDetector()
        self.path_finder = PathFinder()
        
        # Edge cache
        self.cached_edges = None
        self.cached_cost = None
        self.cache_extent = None
        self.pixel_width = 1
        self.pixel_height = 1
        
        # CRS transform
        self.transform = QgsCoordinateTransform(
            self.canvas.mapSettings().destinationCrs(),
            self.raster_layer.crs(),
            QgsProject.instance()
        )
        
        # Auto-create output layer
        if not self.vector_layer:
            self.vector_layer = self.create_output_layer()

    def create_output_layer(self):
        """Create output vector layer."""
        crs = self.canvas.mapSettings().destinationCrs().authid()
        layer = QgsVectorLayer(f"LineString?crs={crs}", "Contours", "memory")
        pr = layer.dataProvider()
        pr.addAttributes([QgsField("id", QVariant.Int)])
        layer.updateFields()
        QgsProject.instance().addMapLayer(layer)
        return layer

    def keyPressEvent(self, event):
        """Handle keyboard input for undo and cancel."""
        if event.key() == Qt.Key_Escape:
            # Undo last point
            if self.path_points:
                self.path_points.pop()
                self.redraw_confirmed_path()
            else:
                self.reset_tracing()
        elif event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            # Cancel entire trace
            self.reset_tracing()
        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            # Finish and save
            if len(self.path_points) >= 2:
                self.save_to_layer()
            self.reset_tracing()

    def canvasPressEvent(self, event):
        """Handle mouse clicks."""
        if event.button() == Qt.LeftButton:
            point = self.toMapCoordinates(event.pos())
            
            if not self.is_tracing:
                # Start new trace
                self.is_tracing = True
                self.path_points = [point]
                self.start_marker.reset(QgsWkbTypes.PointGeometry)
                self.start_marker.addPoint(point)
                self.update_edge_cache()
            else:
                # Get the snapped path from preview and add all points
                if hasattr(self, 'preview_path') and self.preview_path:
                    self.path_points.extend(self.preview_path[1:])  # Skip first (duplicate)
                else:
                    self.path_points.append(point)
                    
            self.redraw_confirmed_path()
                
        elif event.button() == Qt.RightButton:
            # Finish tracing
            if len(self.path_points) >= 2:
                self.save_to_layer()
            self.reset_tracing()

    def canvasMoveEvent(self, event):
        """Live preview on mouse move."""
        if not self.is_tracing or not self.path_points:
            return
            
        current = self.toMapCoordinates(event.pos())
        last = self.path_points[-1]
        
        # Find magnetic path
        path = self.find_path_along_edges(last, current)
        self.preview_path = path
        
        # Draw preview
        self.preview_band.reset(QgsWkbTypes.LineGeometry)
        for pt in path:
            self.preview_band.addPoint(pt)

    def update_edge_cache(self):
        """Cache edge detection for current view."""
        extent = self.canvas.extent()
        image = self.read_raster(extent)
        
        if image is None:
            self.cached_edges = None
            return
            
        self.cached_edges = self.edge_detector.detect_edges(image)
        self.cached_cost = self.edge_detector.get_edge_cost_map(self.cached_edges, self.edge_weight)
        self.cache_extent = extent
        self.pixel_width = extent.width() / image.shape[1]
        self.pixel_height = extent.height() / image.shape[0]

    def read_raster(self, extent):
        """Read raster as grayscale image."""
        provider = self.raster_layer.dataProvider()
        raster_ext = self.raster_layer.extent()
        read_ext = extent.intersect(raster_ext)
        
        if read_ext.isEmpty():
            return None
        
        # Limit size for performance
        raster_res = raster_ext.width() / self.raster_layer.width()
        out_w = min(800, int(read_ext.width() / raster_res))
        out_h = min(800, int(read_ext.height() / raster_res))
        
        if out_w <= 0 or out_h <= 0:
            return None
        
        # Read bands
        bands = []
        for b in range(1, min(4, provider.bandCount() + 1)):
            block = provider.block(b, read_ext, out_w, out_h)
            if block.isValid() and block.data():
                try:
                    arr = np.frombuffer(block.data(), dtype=np.uint8).reshape((out_h, out_w))
                    bands.append(arr)
                except:
                    pass
        
        if not bands:
            return None
        if len(bands) >= 3:
            return cv2.cvtColor(np.stack(bands[:3], axis=-1), cv2.COLOR_RGB2GRAY)
        return bands[0]

    def find_path_along_edges(self, start, end):
        """Find path following detected edges."""
        if self.cached_cost is None:
            return [start, end]
        
        ext = self.cache_extent
        h, w = self.cached_cost.shape
        
        def to_pixel(pt):
            px = int((pt.x() - ext.xMinimum()) / self.pixel_width)
            py = int((ext.yMaximum() - pt.y()) / self.pixel_height)
            return (max(0, min(w-1, px)), max(0, min(h-1, py)))
        
        def to_map(px, py):
            x = ext.xMinimum() + px * self.pixel_width
            y = ext.yMaximum() - py * self.pixel_height
            return QgsPointXY(x, y)
        
        p1 = to_pixel(start)
        p2 = to_pixel(end)
        
        path = self.path_finder.find_path(p1, p2, self.cached_cost)
        
        if not path:
            return [start, end]
        
        # Simplify for display
        step = max(1, len(path) // 30)
        simplified = path[::step]
        if path[-1] not in simplified:
            simplified.append(path[-1])
        
        return [to_map(px, py) for px, py in simplified]

    def redraw_confirmed_path(self):
        """Redraw the confirmed path."""
        self.confirm_band.reset(QgsWkbTypes.LineGeometry)
        for pt in self.path_points:
            self.confirm_band.addPoint(pt)

    def save_to_layer(self):
        """Save path to vector layer."""
        if len(self.path_points) < 2:
            return
            
        geom = QgsGeometry.fromPolylineXY(self.path_points)
        feat = QgsFeature()
        feat.setGeometry(geom)
        feat.setAttributes([self.vector_layer.featureCount() + 1])
        
        self.vector_layer.startEditing()
        self.vector_layer.addFeature(feat)
        self.vector_layer.commitChanges()
        self.vector_layer.triggerRepaint()

    def reset_tracing(self):
        """Reset all tracing state."""
        self.is_tracing = False
        self.path_points = []
        self.preview_path = []
        self.preview_band.reset(QgsWkbTypes.LineGeometry)
        self.confirm_band.reset(QgsWkbTypes.LineGeometry)
        self.start_marker.reset(QgsWkbTypes.PointGeometry)

    def deactivate(self):
        self.reset_tracing()
        super().deactivate()
        self.deactivated.emit()
