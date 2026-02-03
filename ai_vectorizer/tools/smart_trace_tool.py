# -*- coding: utf-8 -*-
"""
Smart Trace Tool v2 - Magnetic Edge Snapping (No more glass walls!)

Key concept:
- User controls direction (mouse movement)
- AI just snaps to nearest edge within snap radius
- If no edge nearby, follows mouse exactly
- Result is smoothed with Bézier curves
"""
import numpy as np
import cv2
from qgis.gui import QgsMapToolEmitPoint, QgsRubberBand
from qgis.core import (
    QgsWkbTypes, QgsProject, QgsPointXY, QgsGeometry, 
    QgsFeature, QgsCoordinateTransform, QgsRectangle,
    QgsVectorLayer, QgsField
)
from qgis.PyQt.QtCore import Qt, QVariant, pyqtSignal
from qgis.PyQt.QtGui import QColor

from ..core.edge_detector import EdgeDetector


class SmartTraceTool(QgsMapToolEmitPoint):
    deactivated = pyqtSignal()
    
    def __init__(self, canvas, raster_layer, vector_layer, model_type=0, 
                 sam_engine=None, edge_weight=0.5, freehand=False, edge_method='canny'):
        self.canvas = canvas
        super().__init__(self.canvas)
        
        self.raster_layer = raster_layer
        self.vector_layer = vector_layer
        self.sam_engine = sam_engine
        self.freehand = freehand
        self.edge_method = edge_method
        
        # Snap radius (pixels) - higher = more magnetic
        self.snap_radius = int(15 * (1.0 - edge_weight * 0.7))  # 5~15 pixels
        
        # Path tracking
        self.path_points = []
        self.is_tracing = False
        self.start_point = None
        self.last_map_point = None
        
        # Sampling interval (map units per sample point)
        self.sample_interval = 0
        
        # RubberBands for visualization
        self.preview_band = QgsRubberBand(self.canvas, QgsWkbTypes.LineGeometry)
        self.preview_band.setColor(QColor(0, 200, 0, 180))
        self.preview_band.setWidth(3)
        
        self.confirm_band = QgsRubberBand(self.canvas, QgsWkbTypes.LineGeometry)
        self.confirm_band.setColor(QColor(255, 50, 50, 255))
        self.confirm_band.setWidth(3)
        
        self.start_marker = QgsRubberBand(self.canvas, QgsWkbTypes.PointGeometry)
        self.start_marker.setColor(QColor(255, 255, 0, 255))
        self.start_marker.setWidth(12)
        self.start_marker.setIcon(QgsRubberBand.ICON_CIRCLE)
        
        self.close_indicator = QgsRubberBand(self.canvas, QgsWkbTypes.PointGeometry)
        self.close_indicator.setColor(QColor(0, 255, 255, 200))
        self.close_indicator.setWidth(16)
        self.close_indicator.setIcon(QgsRubberBand.ICON_CIRCLE)
        
        # Edge detector
        self.edge_detector = EdgeDetector(method=self.edge_method)
        
        # Edge cache
        self.cached_edges = None
        self.cache_extent = None
        self.cache_transform = None  # Pixel <-> Map transform
        
        # CRS transform
        self.transform = QgsCoordinateTransform(
            self.canvas.mapSettings().destinationCrs(),
            self.raster_layer.crs(),
            QgsProject.instance()
        )
        
        # Auto-create output layer if needed
        if not self.vector_layer:
            self.vector_layer = self.create_output_layer()

    def create_output_layer(self):
        crs = self.canvas.mapSettings().destinationCrs().authid()
        layer = QgsVectorLayer(f"LineString?crs={crs}", "Contours", "memory")
        pr = layer.dataProvider()
        pr.addAttributes([QgsField("id", QVariant.Int)])
        layer.updateFields()
        QgsProject.instance().addMapLayer(layer)
        return layer

    def canvasPressEvent(self, event):
        if event.button() == Qt.RightButton:
            # Right click = save and finish
            if len(self.path_points) >= 2:
                self.save_to_layer(closed=False)
            self.reset_tracing()
            return
        
        if event.button() != Qt.LeftButton:
            return
        
        point = self.toMapCoordinates(event.pos())
        
        if not self.is_tracing:
            # Start tracing
            self.is_tracing = True
            self.start_point = point
            self.last_map_point = point
            self.path_points = [point]
            
            # Show start marker
            self.start_marker.reset(QgsWkbTypes.PointGeometry)
            self.start_marker.addPoint(point)
            
            # Set sample interval based on scale (larger = smoother, less jitter)
            self.sample_interval = self.canvas.mapUnitsPerPixel() * 12
            
            # Update edge cache
            if not self.freehand:
                self.update_edge_cache()
        else:
            # Check if closing polygon (near start)
            if self.is_near_start(point):
                self.path_points.append(self.start_point)
                # Ask for elevation value
                elevation = self.ask_elevation()
                if elevation is not None:
                    self.save_to_layer(closed=True, elevation=elevation)
                self.reset_tracing()
                return
            
            # Confirm current preview path
            self.redraw_confirmed_path()

    def canvasMoveEvent(self, event):
        if not self.is_tracing:
            return
        
        current_point = self.toMapCoordinates(event.pos())
        
        # Check close indicator
        if self.is_near_start(current_point):
            self.close_indicator.reset(QgsWkbTypes.PointGeometry)
            self.close_indicator.addPoint(self.start_point)
        else:
            self.close_indicator.reset(QgsWkbTypes.PointGeometry)
        
        if self.last_map_point is None:
            self.last_map_point = current_point
            return
        
        # Calculate distance moved - use LARGER interval for smoother result
        dx = current_point.x() - self.last_map_point.x()
        dy = current_point.y() - self.last_map_point.y()
        dist = np.sqrt(dx*dx + dy*dy)
        
        # Much larger sample interval = smoother lines
        if dist < self.sample_interval:
            return
        
        # Get AI-assisted point
        if self.freehand or self.cached_edges is None:
            snapped = current_point
        else:
            # TRUE EDGE FOLLOWING: trace along the edge, not just snap
            snapped = self.follow_edge(current_point)
        
        # Add to path
        self.path_points.append(snapped)
        self.last_map_point = current_point
        
        # Update preview
        self.redraw_confirmed_path()

    def follow_edge(self, map_point):
        """
        TRUE EDGE FOLLOWING: 
        Instead of just snapping to nearest edge, this walks along the edge
        in the direction the user is moving. This provides real AI assistance.
        """
        if self.cached_edges is None or self.cache_transform is None:
            return map_point
        
        try:
            px, py = self.map_to_pixel(map_point)
            h, w = self.cached_edges.shape
            
            if px < 0 or py < 0 or px >= w or py >= h:
                return map_point
            
            # Get user's movement direction
            direction = (0, 0)
            if len(self.path_points) >= 1:
                last_point = self.path_points[-1]
                last_px, last_py = self.map_to_pixel(last_point)
                dx = px - last_px
                dy = py - last_py
                mag = np.sqrt(dx*dx + dy*dy)
                if mag > 0:
                    direction = (dx/mag, dy/mag)
            
            # STEP ALONG EDGE: Start from last point, walk toward current mouse
            # following the edge line
            if len(self.path_points) >= 1 and direction != (0, 0):
                last_point = self.path_points[-1]
                start_px, start_py = self.map_to_pixel(last_point)
                
                # Walk in small steps toward mouse, sticking to edge
                step_count = 15  # Number of sub-steps
                best_x, best_y = start_px, start_py
                
                for step in range(1, step_count + 1):
                    t = step / step_count
                    # Interpolate position toward mouse
                    target_x = start_px + (px - start_px) * t
                    target_y = start_py + (py - start_py) * t
                    
                    # Find nearest edge within small radius of this target
                    search_radius = 12
                    found_edge = False
                    edge_x, edge_y = int(target_x), int(target_y)
                    min_dist = search_radius + 1
                    
                    for ddy in range(-search_radius, search_radius + 1, 2):
                        for ddx in range(-search_radius, search_radius + 1, 2):
                            nx = int(target_x + ddx)
                            ny = int(target_y + ddy)
                            if 0 <= nx < w and 0 <= ny < h:
                                if self.cached_edges[ny, nx] > 128:
                                    d = np.sqrt(ddx*ddx + ddy*ddy)
                                    if d < min_dist:
                                        min_dist = d
                                        edge_x, edge_y = nx, ny
                                        found_edge = True
                    
                    if found_edge:
                        best_x, best_y = edge_x, edge_y
                
                # Blend result: 80% edge following, 20% mouse for responsiveness
                edge_result = self.pixel_to_map(best_x, best_y)
                blend = 0.8
                final_x = edge_result.x() * blend + map_point.x() * (1 - blend)
                final_y = edge_result.y() * blend + map_point.y() * (1 - blend)
                return QgsPointXY(final_x, final_y)
            
            # Fallback: simple snap
            return self.snap_to_edge(map_point)
            
        except Exception:
            return map_point

    def snap_to_edge(self, map_point):
        """
        Simple edge snapping (fallback when no path history).
        """
        if self.cached_edges is None or self.cache_transform is None:
            return map_point
        
        try:
            px, py = self.map_to_pixel(map_point)
            h, w = self.cached_edges.shape
            
            if px < 0 or py < 0 or px >= w or py >= h:
                return map_point
            
            snap_radius = 10
            best_dist = snap_radius + 1
            best_px, best_py = px, py
            found_edge = False
            
            for dy in range(-snap_radius, snap_radius + 1, 2):
                for dx in range(-snap_radius, snap_radius + 1, 2):
                    nx, ny = int(px + dx), int(py + dy)
                    if 0 <= nx < w and 0 <= ny < h:
                        if self.cached_edges[ny, nx] > 128:
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist < best_dist:
                                best_dist = dist
                                best_px, best_py = nx, ny
                                found_edge = True
            
            if found_edge:
                edge_point = self.pixel_to_map(best_px, best_py)
                blend = 0.7
                blended_x = edge_point.x() * blend + map_point.x() * (1 - blend)
                blended_y = edge_point.y() * blend + map_point.y() * (1 - blend)
                return QgsPointXY(blended_x, blended_y)
            else:
                return map_point
                
        except Exception:
            return map_point

    def update_edge_cache(self):
        """Cache edge detection for current view."""
        try:
            extent = self.canvas.extent()
            provider = self.raster_layer.dataProvider()
            raster_ext = self.raster_layer.extent()
            read_ext = extent.intersect(raster_ext)
            
            if read_ext.isEmpty():
                return
            
            # Determine output size (limit for performance)
            raster_res = raster_ext.width() / self.raster_layer.width()
            out_w = min(1000, int(read_ext.width() / raster_res))
            out_h = min(1000, int(read_ext.height() / raster_res))
            
            if out_w < 10 or out_h < 10:
                return
            
            # Read bands
            bands = []
            for b in range(1, min(4, provider.bandCount() + 1)):
                block = provider.block(b, read_ext, out_w, out_h)
                if block.isValid() and block.data():
                    try:
                        arr = np.frombuffer(block.data(), dtype=np.uint8).reshape((out_h, out_w))
                        bands.append(arr.copy())
                    except:
                        pass
            
            if not bands:
                return
            
            # Convert to grayscale
            if len(bands) >= 3:
                image = cv2.cvtColor(np.stack(bands[:3], axis=-1), cv2.COLOR_RGB2GRAY)
            else:
                image = bands[0]
            
            # Detect edges
            self.cached_edges = self.edge_detector.detect_edges(image)
            self.cache_extent = read_ext
            
            # Store transform info
            self.cache_transform = {
                'x_min': read_ext.xMinimum(),
                'y_max': read_ext.yMaximum(),
                'px_w': read_ext.width() / out_w,
                'px_h': read_ext.height() / out_h,
                'width': out_w,
                'height': out_h
            }
            
        except Exception as e:
            print(f"Edge cache error: {e}")

    def map_to_pixel(self, map_point):
        """Convert map coordinates to pixel coordinates."""
        t = self.cache_transform
        px = (map_point.x() - t['x_min']) / t['px_w']
        py = (t['y_max'] - map_point.y()) / t['px_h']
        return int(px), int(py)

    def pixel_to_map(self, px, py):
        """Convert pixel coordinates to map coordinates."""
        t = self.cache_transform
        x = t['x_min'] + px * t['px_w']
        y = t['y_max'] - py * t['px_h']
        return QgsPointXY(x, y)

    def is_near_start(self, point):
        """Check if point is near start point for polygon close."""
        if not self.start_point or len(self.path_points) < 3:
            return False
        
        dx = point.x() - self.start_point.x()
        dy = point.y() - self.start_point.y()
        dist = np.sqrt(dx*dx + dy*dy)
        
        close_threshold = self.canvas.mapUnitsPerPixel() * 20
        return dist < close_threshold

    def redraw_confirmed_path(self):
        """Redraw the confirmed path."""
        self.confirm_band.reset(QgsWkbTypes.LineGeometry)
        for pt in self.path_points:
            self.confirm_band.addPoint(pt)

    def save_to_layer(self, closed=False, elevation=None):
        """Save path to vector layer with Bézier smoothing."""
        if len(self.path_points) < 2:
            return
        
        # Apply Bézier smoothing
        smoothed = self.smooth_bezier(self.path_points)
        
        # Create geometry
        if closed and len(smoothed) >= 3:
            ring = smoothed + [smoothed[0]]
            geom = QgsGeometry.fromPolygonXY([ring])
        else:
            geom = QgsGeometry.fromPolylineXY(smoothed)
        
        # Additional QGIS simplification (stronger for smoother result)
        tolerance = self.canvas.mapUnitsPerPixel() * 2.5
        geom = geom.simplify(tolerance)
        
        feat = QgsFeature()
        feat.setGeometry(geom)
        
        # Set attributes - include elevation if provided
        attrs = [self.vector_layer.featureCount() + 1]
        
        # Find or create elevation field
        fields = self.vector_layer.fields()
        elev_idx = fields.indexOf('elevation')
        if elev_idx == -1 and elevation is not None:
            # Add elevation field if not exists
            self.vector_layer.startEditing()
            self.vector_layer.dataProvider().addAttributes([QgsField('elevation', QVariant.Double)])
            self.vector_layer.updateFields()
            fields = self.vector_layer.fields()
            elev_idx = fields.indexOf('elevation')
        
        # Prepare attributes
        if elev_idx >= 0 and elevation is not None:
            while len(attrs) <= elev_idx:
                attrs.append(None)
            attrs[elev_idx] = float(elevation)
        
        feat.setAttributes(attrs)
        
        self.vector_layer.startEditing()
        self.vector_layer.addFeature(feat)
        self.vector_layer.commitChanges()
        self.vector_layer.triggerRepaint()

    def ask_elevation(self):
        """Show dialog to input elevation value."""
        from qgis.PyQt.QtWidgets import QInputDialog
        
        value, ok = QInputDialog.getDouble(
            None,
            "등고선 해발값",
            "해발고도 (m):",
            0.0,  # default
            -1000.0,  # min
            10000.0,  # max
            1  # decimals
        )
        
        if ok:
            return value
        return None

    def smooth_bezier(self, points):
        """
        Smooth points using Bézier-like curve fitting.
        Uses Chaikin's corner cutting algorithm for smooth curves.
        """
        if len(points) < 3:
            return points
        
        # Convert to numpy for easier math
        pts = np.array([[p.x(), p.y()] for p in points])
        
        # Apply Chaikin's algorithm 4 times for ultra-smooth curves
        for _ in range(4):
            if len(pts) < 3:
                break
            new_pts = [pts[0]]  # Keep first point
            
            for i in range(len(pts) - 1):
                p0, p1 = pts[i], pts[i + 1]
                # 1/4 and 3/4 points
                q = p0 * 0.75 + p1 * 0.25
                r = p0 * 0.25 + p1 * 0.75
                new_pts.extend([q, r])
            
            new_pts.append(pts[-1])  # Keep last point
            pts = np.array(new_pts)
        
        # Subsample to reduce point count
        if len(pts) > 100:
            indices = np.linspace(0, len(pts) - 1, 100, dtype=int)
            pts = pts[indices]
        
        return [QgsPointXY(p[0], p[1]) for p in pts]

    def reset_tracing(self):
        """Reset all tracing state."""
        self.is_tracing = False
        self.path_points = []
        self.start_point = None
        self.last_map_point = None
        self.preview_band.reset(QgsWkbTypes.LineGeometry)
        self.confirm_band.reset(QgsWkbTypes.LineGeometry)
        self.start_marker.reset(QgsWkbTypes.PointGeometry)
        self.close_indicator.reset(QgsWkbTypes.PointGeometry)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            if self.path_points:
                self.path_points.pop()
                self.redraw_confirmed_path()
            else:
                self.reset_tracing()
        elif event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            self.reset_tracing()
        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if len(self.path_points) >= 2:
                self.save_to_layer(closed=False)
            self.reset_tracing()

    def deactivate(self):
        self.reset_tracing()
        super().deactivate()
        self.deactivated.emit()
