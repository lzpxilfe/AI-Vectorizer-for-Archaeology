# -*- coding: utf-8 -*-
import numpy as np
import os
import cv2
from qgis.gui import QgsMapToolEmitPoint, QgsRubberBand
from qgis.core import (
    QgsWkbTypes, QgsProject, QgsPointXY, QgsGeometry, 
    QgsFeature, QgsCoordinateTransform, QgsRectangle
)
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor

from ..core.edge_detector import EdgeDetector
from ..core.path_finder import PathFinder
from ..core.vectorizer import Vectorizer

class SmartTraceTool(QgsMapToolEmitPoint):
    def __init__(self, canvas, raster_layer, vector_layer, model_type=0, sam_engine=None):
        self.canvas = canvas
        super().__init__(self.canvas)
        
        self.raster_layer = raster_layer
        self.vector_layer = vector_layer
        self.model_type = model_type  # 0: Lite, 1: Standard, 2: Pro
        self.sam_engine = sam_engine
        
        self.start_point = None
        self.rubber_band = QgsRubberBand(self.canvas, QgsWkbTypes.LineGeometry)
        self.rubber_band.setColor(QColor(255, 0, 0, 255))
        self.rubber_band.setWidth(2)
        
        # Core engines
        self.edge_detector = EdgeDetector()
        self.path_finder = PathFinder()
        self.vectorizer = Vectorizer()
        
        # CRS Transformer
        self.transform_to_raster = QgsCoordinateTransform(
            self.canvas.mapSettings().destinationCrs(),
            self.raster_layer.crs(),
            QgsProject.instance()
        )

    def canvasReleaseEvent(self, event):
        # Get click point in Map Canvas CRS
        point_canvas = self.toMapCoordinates(event.pos())
        
        # Transform to Raster Layer CRS
        point_layer = self.transform_to_raster.transform(point_canvas)
        
        # Convert to Pixel Coordinates
        pixel_x, pixel_y = self.layer_to_pixel(point_layer)
        
        if self.start_point is None:
            self.start_point = (pixel_x, pixel_y)
            self.rubber_band.reset(QgsWkbTypes.PointGeometry)
            self.rubber_band.addPoint(point_canvas, True)
            self.rubber_band.show()
        else:
            end_point = (pixel_x, pixel_y)
            self.process_segment(self.start_point, end_point)
            self.start_point = end_point # Continuous drawing

    def layer_to_pixel(self, point_layer):
        """Convert layer coordinates to pixel coordinates."""
        ext = self.raster_layer.extent()
        width = self.raster_layer.width()
        height = self.raster_layer.height()
        
        res_x = (ext.xMaximum() - ext.xMinimum()) / width
        res_y = (ext.yMaximum() - ext.yMinimum()) / height
        
        col = int((point_layer.x() - ext.xMinimum()) / res_x)
        row = int((ext.yMaximum() - point_layer.y()) / res_y)
        
        return col, row

    def pixel_to_layer(self, col, row):
        """Convert pixel coordinates to layer coordinates."""
        ext = self.raster_layer.extent()
        width = self.raster_layer.width()
        height = self.raster_layer.height()
        
        res_x = (ext.xMaximum() - ext.xMinimum()) / width
        res_y = (ext.yMaximum() - ext.yMinimum()) / height
        
        x = ext.xMinimum() + (col + 0.5) * res_x
        y = ext.yMaximum() - (row + 0.5) * res_y
        
        return QgsPointXY(x, y)

    def process_segment(self, p1, p2):
        """Process ROI and find path between p1 (col, row) and p2 (col, row)."""
        
        # 1. Define ROI
        padding = 50
        min_col = max(0, min(p1[0], p2[0]) - padding)
        max_col = min(self.raster_layer.width(), max(p1[0], p2[0]) + padding)
        min_row = max(0, min(p1[1], p2[1]) - padding)
        max_row = min(self.raster_layer.height(), max(p1[1], p2[1]) + padding)
        
        width = max_col - min_col
        height = max_row - min_row
        
        if width <= 0 or height <= 0:
            return

        # 2. Read ROI raster data
        provider = self.raster_layer.dataProvider()
        ext = self.raster_layer.extent()
        res_x = (ext.xMaximum() - ext.xMinimum()) / self.raster_layer.width()
        res_y = (ext.yMaximum() - ext.yMinimum()) / self.raster_layer.height()
        
        roi_rect = QgsRectangle(
            ext.xMinimum() + min_col * res_x,
            ext.yMaximum() - max_row * res_y,
            ext.xMinimum() + max_col * res_x,
            ext.yMaximum() - min_row * res_y
        )
        
        block = provider.block(1, roi_rect, width, height)
        if not block.isValid():
            return

        data = block.data()
        if not data:
            return
            
        try:
            image = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
        except Exception as e:
            print(f"Error converting block: {e}")
            return
            
        roi_p1 = (p1[0] - min_col, p1[1] - min_row)
        roi_p2 = (p2[0] - min_col, p2[1] - min_row)
        
        path_pixels = []

        # 3. AI Inference
        if self.model_type == 1 and self.sam_engine: # Standard (MobileSAM)
            # Convert to RGB for SAM
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = image # Only if provider gave 3 channels, but here likely 1

            self.sam_engine.set_image(image_rgb)
            
            # Predict
            mask = self.sam_engine.predict_point([roi_p1, roi_p2], [1, 1])
            
            if mask is not None:
                # Vectorize Mask -> Line
                geom = self.vectorizer.mask_to_line(mask)
                if geom:
                    points = geom.asPolyline() # List of QgsPointXY
                    # Convert simple point list (x, y)
                    path_pixels = [ (p.x(), p.y()) for p in points ]
            
            # Fallback if SAM fails to find path suitable for vectorization
            if not path_pixels:
                print("SAM result vectorization failed, falling back to straight line")
                path_pixels = [roi_p1, roi_p2]

        else: # Lite Mode (OpenCV)
            edges = self.edge_detector.detect_edges(image)
            cost_map = self.edge_detector.get_edge_cost_map(edges)
            path_pixels = self.path_finder.find_path(roi_p1, roi_p2, cost_map)
        
        if not path_pixels:
            path_pixels = [roi_p1, roi_p2]
        
        # 4. Convert to Map Coords
        line_points = []
        for px, py in path_pixels:
            global_col = min_col + px
            global_row = min_row + py
            pt_layer = self.pixel_to_layer(global_col, global_row)
            pt_canvas = self.transform_to_raster.transform(pt_layer, direction=QgsCoordinateTransform.ReverseTransform)
            line_points.append(pt_canvas)
        
        # 5. Add to Layer & RubberBand
        self.add_feature_to_layer(line_points)
        for pt in line_points:
            self.rubber_band.addPoint(pt)

    def add_feature_to_layer(self, points):
        """Add a polyline feature to the vector layer."""
        if not self.vector_layer:
            return
            
        tr = QgsCoordinateTransform(
            self.canvas.mapSettings().destinationCrs(),
            self.vector_layer.crs(),
            QgsProject.instance()
        )
        
        transformed_points = [tr.transform(p) for p in points]
        geom = QgsGeometry.fromPolylineXY(transformed_points)
        
        feat = QgsFeature()
        feat.setGeometry(geom)
        
        self.vector_layer.startEditing()
        self.vector_layer.addFeature(feat)
        self.vector_layer.commitChanges()
        self.vector_layer.triggerRepaint()

    def deactivate(self):
        self.rubber_band.reset(QgsWkbTypes.LineGeometry)
        super().deactivate()
        self.deactivated.emit()
