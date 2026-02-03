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
        
        # Checkpoint markers (blue diamonds)
        self.checkpoint_markers = QgsRubberBand(self.canvas, QgsWkbTypes.PointGeometry)
        self.checkpoint_markers.setColor(QColor(50, 150, 255, 255))
        self.checkpoint_markers.setWidth(10)
        self.checkpoint_markers.setIcon(QgsRubberBand.ICON_BOX)
        
        # Checkpoints: list of point indices where user clicked
        self.checkpoints = []
        
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
            self.checkpoints = [0]  # Start point is first checkpoint
            
            # Show start marker
            self.start_marker.reset(QgsWkbTypes.PointGeometry)
            self.start_marker.addPoint(point)
            
            # Reset checkpoint markers
            self.checkpoint_markers.reset(QgsWkbTypes.PointGeometry)
            
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
            
            # ADD CHECKPOINT: Save current position as checkpoint
            if len(self.path_points) > 0:
                self.checkpoints.append(len(self.path_points) - 1)
                # Show checkpoint marker
                self.checkpoint_markers.addPoint(self.path_points[-1])
            
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
        
        # Large sample interval for smoother lines
        if dist < self.sample_interval:
            return
        
        # PURE MOUSE FOLLOWING - No edge snapping at all
        # All smoothing is done in post-processing (Chaikin + simplify)
        # This eliminates ALL jumping/jittering from edge detection
        final_point = current_point
        
        # Add to path
        self.path_points.append(final_point)
        self.last_map_point = current_point
        
        # Update preview
        self.redraw_confirmed_path()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for undo and save."""
        if not self.is_tracing:
            return
        
        # Ctrl+Z: Undo to last checkpoint
        if event.key() == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            self.undo_to_checkpoint()
            return
        
        # Esc: Remove last 10 points (quick undo)
        if event.key() == Qt.Key_Escape:
            self.undo_points(10)
            return
        
        # Delete: Cancel entire line
        if event.key() == Qt.Key_Delete:
            self.reset_tracing()
            return
        
        # Enter: Save current line
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if len(self.path_points) >= 2:
                self.save_to_layer(closed=False)
            self.reset_tracing()
            return

    def undo_to_checkpoint(self):
        """Undo back to the last checkpoint, but KEEP the checkpoint to continue from."""
        if len(self.checkpoints) <= 1:
            # Only start checkpoint - can't undo further, just notify
            return
        
        # Get last checkpoint index (the one we want to KEEP)
        last_cp_idx = self.checkpoints[-1]
        
        # Check if we're already AT the checkpoint (no new points after it)
        if len(self.path_points) <= last_cp_idx + 1:
            # Already at checkpoint, go back to PREVIOUS checkpoint
            if len(self.checkpoints) > 1:
                self.checkpoints.pop()  # Remove current checkpoint
                if self.checkpoints:
                    last_cp_idx = self.checkpoints[-1]
                else:
                    self.reset_tracing()
                    return
        
        # Trim path to checkpoint (keep points UP TO AND INCLUDING checkpoint)
        self.path_points = self.path_points[:last_cp_idx + 1]
        
        # Update last_map_point so user can continue from checkpoint
        if self.path_points:
            self.last_map_point = self.path_points[-1]
        
        # Rebuild checkpoint markers
        self.checkpoint_markers.reset(QgsWkbTypes.PointGeometry)
        for cp_idx in self.checkpoints[1:]:  # Skip start point
            if cp_idx < len(self.path_points):
                self.checkpoint_markers.addPoint(self.path_points[cp_idx])
        
        # Redraw
        self.redraw_confirmed_path()

    def undo_points(self, count):
        """Remove last N points."""
        if len(self.path_points) <= 1:
            return
        
        # Remove points but keep at least the start
        remove_count = min(count, len(self.path_points) - 1)
        self.path_points = self.path_points[:-remove_count]
        
        # Update last_map_point
        if self.path_points:
            self.last_map_point = self.path_points[-1]
        
        # Remove checkpoints that are now beyond the path
        while self.checkpoints and self.checkpoints[-1] >= len(self.path_points):
            self.checkpoints.pop()
        
        # Rebuild checkpoint markers
        self.checkpoint_markers.reset(QgsWkbTypes.PointGeometry)
        for cp_idx in self.checkpoints[1:]:
            if cp_idx < len(self.path_points):
                self.checkpoint_markers.addPoint(self.path_points[cp_idx])
        
        # Redraw
        self.redraw_confirmed_path()

    def gentle_snap(self, map_point):
        """
        VERY gentle edge snapping:
        - Only nudge slightly toward edge if very close
        - Never jump or cause jittery motion
        - Prioritize smooth drawing over edge accuracy
        """
        if self.cached_edges is None or self.cache_transform is None:
            return map_point
        
        try:
            px, py = self.map_to_pixel(map_point)
            h, w = self.cached_edges.shape
            
            if px < 0 or py < 0 or px >= w or py >= h:
                return map_point
            
            # Check only immediate vicinity (5 pixels)
            snap_radius = 5
            
            # Check if directly on edge first
            ipx, ipy = int(px), int(py)
            if 0 <= ipx < w and 0 <= ipy < h:
                if self.cached_edges[ipy, ipx] > 128:
                    # Already on edge, no change needed
                    return map_point
            
            # Look for nearby edge
            best_dist = snap_radius + 1
            best_px, best_py = px, py
            found = False
            
            for dy in range(-snap_radius, snap_radius + 1):
                for dx in range(-snap_radius, snap_radius + 1):
                    nx, ny = int(px + dx), int(py + dy)
                    if 0 <= nx < w and 0 <= ny < h:
                        if self.cached_edges[ny, nx] > 128:
                            dist = abs(dx) + abs(dy)  # Manhattan distance for stability
                            if dist < best_dist:
                                best_dist = dist
                                best_px, best_py = nx, ny
                                found = True
            
            if found:
                edge_point = self.pixel_to_map(best_px, best_py)
                # VERY gentle nudge - only 30% toward edge
                blend = 0.3
                result_x = map_point.x() * (1 - blend) + edge_point.x() * blend
                result_y = map_point.y() * (1 - blend) + edge_point.y() * blend
                return QgsPointXY(result_x, result_y)
            
            # No edge nearby - just follow mouse exactly
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
        
        # Heavy QGIS simplification for ultra-smooth result
        tolerance = self.canvas.mapUnitsPerPixel() * 4
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
        
        # Apply Chaikin's algorithm 5 times for maximum smoothness
        for _ in range(5):
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
        self.checkpoints = []
        self.start_point = None
        self.last_map_point = None
        self.preview_band.reset(QgsWkbTypes.LineGeometry)
        self.confirm_band.reset(QgsWkbTypes.LineGeometry)
        self.start_marker.reset(QgsWkbTypes.PointGeometry)
        self.close_indicator.reset(QgsWkbTypes.PointGeometry)
        self.checkpoint_markers.reset(QgsWkbTypes.PointGeometry)

    def deactivate(self):
        self.reset_tracing()
        super().deactivate()
        self.deactivated.emit()
