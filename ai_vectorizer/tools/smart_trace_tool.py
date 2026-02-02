# -*- coding: utf-8 -*-
"""
Magnetic Trace Tool - Photoshop-like edge following for contour tracing
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
from qgis.PyQt.QtGui import QColor, QCursor

from ..core.edge_detector import EdgeDetector
from ..core.path_finder import PathFinder

class SmartTraceTool(QgsMapToolEmitPoint):
    def __init__(self, canvas, raster_layer, vector_layer, model_type=0, sam_engine=None):
        self.canvas = canvas
        super().__init__(self.canvas)
        
        self.raster_layer = raster_layer
        self.vector_layer = vector_layer
        self.model_type = model_type  
        self.sam_engine = sam_engine
        
        # Collected path points (in map coordinates)
        self.path_points = []
        self.is_tracing = False
        
        # RubberBand for live preview (Green = preview)
        self.preview_band = QgsRubberBand(self.canvas, QgsWkbTypes.LineGeometry)
        self.preview_band.setColor(QColor(0, 255, 0, 200))
        self.preview_band.setWidth(2)
        
        # RubberBand for confirmed path (Red = confirmed)
        self.confirm_band = QgsRubberBand(self.canvas, QgsWkbTypes.LineGeometry)
        self.confirm_band.setColor(QColor(255, 0, 0, 255))
        self.confirm_band.setWidth(2)
        
        # Core engines
        self.edge_detector = EdgeDetector()
        self.path_finder = PathFinder()
        
        # Cached edge map for current view
        self.cached_edges = None
        self.cached_extent = None
        self.cached_transform = None
        
        # CRS Transformer
        self.transform_to_raster = QgsCoordinateTransform(
            self.canvas.mapSettings().destinationCrs(),
            self.raster_layer.crs(),
            QgsProject.instance()
        )
        
        # Create output layer if not provided
        if not self.vector_layer:
            self.vector_layer = self.create_temp_layer()

    def create_temp_layer(self):
        """Create a temporary memory layer for output."""
        crs = self.canvas.mapSettings().destinationCrs().authid()
        layer = QgsVectorLayer(f"LineString?crs={crs}", "ArchaeoTrace Output", "memory")
        pr = layer.dataProvider()
        pr.addAttributes([QgsField("id", QVariant.Int)])
        layer.updateFields()
        QgsProject.instance().addMapLayer(layer)
        return layer

    def canvasPressEvent(self, event):
        """Start or continue tracing on click."""
        if event.button() == Qt.LeftButton:
            point_canvas = self.toMapCoordinates(event.pos())
            
            if not self.is_tracing:
                # Start new trace
                self.is_tracing = True
                self.path_points = [point_canvas]
                self.confirm_band.reset(QgsWkbTypes.LineGeometry)
                self.confirm_band.addPoint(point_canvas)
                self.update_cache_for_view()
            else:
                # Add point to path
                self.path_points.append(point_canvas)
                self.confirm_band.addPoint(point_canvas)
                
        elif event.button() == Qt.RightButton:
            # Finish tracing
            if self.is_tracing and len(self.path_points) >= 2:
                self.save_to_layer()
            self.reset_tracing()

    def canvasMoveEvent(self, event):
        """Show live preview path as mouse moves."""
        if not self.is_tracing or not self.path_points:
            return
            
        current_pos = self.toMapCoordinates(event.pos())
        last_point = self.path_points[-1]
        
        # Find path from last_point to current_pos
        path = self.find_magnetic_path(last_point, current_pos)
        
        # Update preview band
        self.preview_band.reset(QgsWkbTypes.LineGeometry)
        for pt in path:
            self.preview_band.addPoint(pt)

    def update_cache_for_view(self):
        """Cache edge detection for current canvas extent."""
        extent = self.canvas.extent()
        
        # Read visible portion of raster
        image = self.read_raster_as_image(extent)
        if image is None:
            return
            
        # Detect edges
        self.cached_edges = self.edge_detector.detect_edges(image)
        self.cached_extent = extent
        
        # Store pixel resolution
        self.pixel_width = extent.width() / image.shape[1]
        self.pixel_height = extent.height() / image.shape[0]

    def read_raster_as_image(self, extent):
        """Read raster data as grayscale numpy array."""
        provider = self.raster_layer.dataProvider()
        
        # Get intersection with raster extent
        raster_ext = self.raster_layer.extent()
        read_ext = extent.intersect(raster_ext)
        if read_ext.isEmpty():
            return None
        
        # Calculate output size (limit for performance)
        raster_res_x = raster_ext.width() / self.raster_layer.width()
        raster_res_y = raster_ext.height() / self.raster_layer.height()
        
        out_width = min(1000, int(read_ext.width() / raster_res_x))
        out_height = min(1000, int(read_ext.height() / raster_res_y))
        
        if out_width <= 0 or out_height <= 0:
            return None
        
        # Read all bands and convert to grayscale
        bands = []
        for band_num in range(1, min(4, provider.bandCount() + 1)):  # Max 3 bands (RGB)
            block = provider.block(band_num, read_ext, out_width, out_height)
            if block.isValid():
                data = block.data()
                if data:
                    try:
                        band_data = np.frombuffer(data, dtype=np.uint8).reshape((out_height, out_width))
                        bands.append(band_data)
                    except:
                        pass
        
        if not bands:
            return None
        
        if len(bands) == 1:
            return bands[0]
        elif len(bands) >= 3:
            # Convert RGB to grayscale
            rgb = np.stack(bands[:3], axis=-1)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            return gray
        else:
            return bands[0]

    def find_magnetic_path(self, start_map, end_map):
        """Find path along edges from start to end (in map coords)."""
        if self.cached_edges is None:
            return [start_map, end_map]
        
        ext = self.cached_extent
        h, w = self.cached_edges.shape
        
        # Convert map coords to pixel coords in cached image
        def map_to_pixel(pt):
            px = int((pt.x() - ext.xMinimum()) / self.pixel_width)
            py = int((ext.yMaximum() - pt.y()) / self.pixel_height)
            return (max(0, min(w-1, px)), max(0, min(h-1, py)))
        
        def pixel_to_map(px, py):
            x = ext.xMinimum() + px * self.pixel_width
            y = ext.yMaximum() - py * self.pixel_height
            return QgsPointXY(x, y)
        
        p1 = map_to_pixel(start_map)
        p2 = map_to_pixel(end_map)
        
        # Get cost map and find path
        cost_map = self.edge_detector.get_edge_cost_map(self.cached_edges)
        path_pixels = self.path_finder.find_path(p1, p2, cost_map)
        
        if not path_pixels:
            return [start_map, end_map]
        
        # Simplify path (take every Nth point)
        step = max(1, len(path_pixels) // 50)
        simplified = path_pixels[::step]
        if path_pixels[-1] not in simplified:
            simplified.append(path_pixels[-1])
        
        # Convert back to map coords
        return [pixel_to_map(px, py) for px, py in simplified]

    def save_to_layer(self):
        """Save current path to vector layer."""
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
        """Reset tracing state."""
        self.is_tracing = False
        self.path_points = []
        self.preview_band.reset(QgsWkbTypes.LineGeometry)
        self.confirm_band.reset(QgsWkbTypes.LineGeometry)

    def deactivate(self):
        self.reset_tracing()
        super().deactivate()
        self.deactivated.emit()
