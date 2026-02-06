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
import heapq
import math
from qgis.gui import QgsMapToolEmitPoint, QgsRubberBand
from qgis.core import (
    QgsWkbTypes, QgsProject, QgsPointXY, QgsGeometry, 
    QgsFeature, QgsCoordinateTransform, QgsRectangle,
    QgsVectorLayer, QgsField, Qgis
)
from qgis.PyQt.QtCore import Qt, QVariant, pyqtSignal
from qgis.PyQt.QtGui import QColor, QCursor
from qgis.PyQt.QtWidgets import QMenu

from ..core.edge_detector import EdgeDetector


class SmartTraceTool(QgsMapToolEmitPoint):
    deactivated = pyqtSignal()
    
    def __init__(self, canvas, raster_layer, vector_layer, model_type=0, 
                 sam_engine=None, edge_weight=0.5, freehand=False, edge_method='canny', iface=None):
        self.canvas = canvas
        super().__init__(self.canvas)
        self.iface = iface
        
        self.raster_layer = raster_layer
        self.vector_layer = vector_layer
        self.sam_engine = sam_engine
        self.freehand = freehand
        self.edge_method = edge_method
        
        # Snap radius (pixels) - higher = more magnetic
        self.snap_radius = int(15 * (1.0 - edge_weight * 0.7))  # 5~15 pixels
        
        # Path tracking
        self.path_points = []
        self.preview_path = []  # For hovering preview
        self.is_tracing = False
        self.start_point = None
        self.last_map_point = None
        
        # Sampling interval (map units per sample point)
        self.sample_interval = 0
        
        # RubberBands for visualization
        self.preview_band = QgsRubberBand(self.canvas, QgsWkbTypes.LineGeometry)
        self.preview_band.setColor(QColor(0, 255, 0, 180))  # Brighter Green, slightly transparent
        self.preview_band.setWidth(6)  # Even thicker
        self.preview_band.setLineStyle(Qt.DashLine)  # Dash Line (longer dashes)
        
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
            # Right click = Context Menu (Safety mechanism)
            if not self.is_tracing:
                return

            menu = QMenu()
            save_action = menu.addAction("Finish Line (Enter)")
            undo_action = menu.addAction("Undo Last Segment (Ctrl+Z)")
            cancel_menu_action = menu.addAction("Cancel Menu")
            
            # Execute menu at mouse position
            action = menu.exec_(QCursor.pos())
            
            if action == save_action:
                if len(self.path_points) >= 2:
                    self.save_to_layer(closed=False)
                self.reset_tracing()
            elif action == undo_action:
                self.undo_to_checkpoint()
            
            # If Cancel or clicked outside, do nothing (prevent reset)
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
            self.sample_interval = self.canvas.mapUnitsPerPixel() * 18
            
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
            if self.preview_path:
                # Commit AI path (Click to confirm preview)
                self.path_points.extend(self.preview_path)
                self.preview_path = []
            else:
                # Manual click point
                if len(self.path_points) > 0:
                    # If points exist, add straight line to click
                    self.path_points.append(point)
            
            # Add checkpoint
            self.checkpoints.append(len(self.path_points) - 1)
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
        
        dx = current_point.x() - self.last_map_point.x()
        dy = current_point.y() - self.last_map_point.y()
        dist = np.sqrt(dx*dx + dy*dy)
        
        # Minimum movement check (optimization)
        if dist < self.sample_interval:
            return

        # MODE CHECK: Dragging vs Hovering
        if event.buttons() & Qt.LeftButton:
            # DRAGGING: Manual Draw (Mouse Following + Gentle Snap)
            self.preview_path = []
            
            if self.freehand or self.cached_edges is None:
                final_point = current_point
            else:
                final_point = self.angle_constrained_snap(current_point)
            
            self.path_points.append(final_point)
            self.last_map_point = current_point
            self.redraw_confirmed_path()
        else:
            # HOVERING: AI Auto-Path Preview (Bunting Style)
            if not self.path_points: return
            
            # Calculate A* path from last point to mouse
            ai_path = self.find_optimal_path(current_point)
            self.preview_path = ai_path
            
            # Draw preview (Green line)
            self.preview_band.reset(QgsWkbTypes.LineGeometry)
            if self.path_points:
                self.preview_band.addPoint(self.path_points[-1])
            for pt in ai_path:
                self.preview_band.addPoint(pt)

    def angle_constrained_snap(self, map_point):
        """
        Smart gently snap that checks ANGLE continuity.
        Only snaps to edge if it continues the current line naturally.
        Prevents jumping to perpendicular noise (broken glass effect).
        """
        if self.cached_edges is None:
            return map_point
        
        try:
            px, py = self.map_to_pixel(map_point)
            h, w = self.cached_edges.shape
            
            if px < 0 or py < 0 or px >= w or py >= h:
                return map_point
            
            # 1. Check if we have history to determine direction
            has_history = len(self.path_points) >= 2
            last_angle = 0
            if has_history:
                p1 = self.path_points[-2]
                p2 = self.path_points[-1]
                last_angle = math.atan2(p2.y() - p1.y(), p2.x() - p1.x())
            
            # 2. Search for nearby edge pixels
            snap_radius = 6  # Small radius
            best_dist = snap_radius + 1
            best_px, best_py = px, py
            found = False
            
            for dy in range(-snap_radius, snap_radius + 1):
                for dx in range(-snap_radius, snap_radius + 1):
                    nx, ny = int(px + dx), int(py + dy)
                    if 0 <= nx < w and 0 <= ny < h:
                        if self.cached_edges[ny, nx] > 128:
                            # 3. Angle Filter: Check if this point causes sharp turn
                            if has_history:
                                edge_pt = self.pixel_to_map(nx, ny)
                                last_pt = self.path_points[-1]
                                new_angle = math.atan2(edge_pt.y() - last_pt.y(), edge_pt.x() - last_pt.x())
                                angle_diff = abs(new_angle - last_angle)
                                while angle_diff > math.pi: angle_diff -= 2*math.pi
                                while angle_diff < -math.pi: angle_diff += 2*math.pi
                                
                                # If turn is sharper than 60 degrees, ignore this edge (it's noise/hairline)
                                if abs(angle_diff) > math.radians(60):
                                    continue
                            
                            dist = abs(dx) + abs(dy)
                            if dist < best_dist:
                                best_dist = dist
                                best_px, best_py = nx, ny
                                found = True
            
            if found:
                edge_point = self.pixel_to_map(best_px, best_py)
                # Gentle blend: 30% edge, 70% mouse
                blend = 0.3
                result_x = map_point.x() * (1 - blend) + edge_point.x() * blend
                result_y = map_point.y() * (1 - blend) + edge_point.y() * blend
                return QgsPointXY(result_x, result_y)
            
            return map_point
            
        except Exception:
            return map_point

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for undo and save."""
        if not self.is_tracing:
            return
        
        # Ctrl+Z: Undo to last checkpoint
        if (event.key() == Qt.Key_Z and event.modifiers() & Qt.ControlModifier) or event.key() == Qt.Key_Backspace:
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

    def find_optimal_path(self, target_point):
        """
        A* Path Finding from last point to target point.
        Uses cached_cost map to prefer edges.
        """
        if self.cached_cost is None or not self.path_points:
            return [target_point]
            
        try:
            start_point = self.path_points[-1]
            start_px, start_py = self.map_to_pixel(start_point)
            end_px, end_py = self.map_to_pixel(target_point)
            
            h, w = self.cached_cost.shape
            
            # Fix "Glass Wall": Clamp coordinates to image bounds
            # This allows tracing even if start point is off-screen (panned)
            # or cursor is slightly outside the canvas
            start_px = max(0, min(w-1, start_px))
            start_py = max(0, min(h-1, start_py))
            end_px = max(0, min(w-1, end_px))
            end_py = max(0, min(h-1, end_py))
            
            # Dijkstra / A*
            # Priority Queue: (cost, x, y)
            pq = [(0, start_px, start_py)]
            came_from = {}
            cost_so_far = {(start_px, start_py): 0}
            
            # Optimization: Limit iterations (don't search forever)
            # Distance based limit - Maximized for 1:50,000 scale map contours
            manhattan_dist = abs(end_px - start_px) + abs(end_py - start_py)
            max_iter = max(100000, manhattan_dist * 500) # Covers almost entire screen search
            iter_count = 0
            
            # Track closest approach in case of timeout
            best_node = None
            min_dist_to_target = float('inf')
            
            found = False
            
            while pq:
                iter_count += 1
                if iter_count > max_iter:
                    break # Too far / complex
                    
                current_cost, cx, cy = heapq.heappop(pq)
                
                # Track best progress
                dist_to_target = abs(end_px - cx) + abs(end_py - cy)
                if dist_to_target < min_dist_to_target:
                    min_dist_to_target = dist_to_target
                    best_node = (cx, cy)
                
                if (cx, cy) == (end_px, end_py):
                    found = True
                    break
                
                # Check 8 neighbors
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                    nx, ny = cx + dx, cy + dy
                    
                    if 0 <= nx < w and 0 <= ny < h:
                        # Cost calculation:
                        # Edge cost (from map) + Movement cost (1 for straight, 1.4 for diag)
                        move_cost = 1.414 if dx!=0 and dy!=0 else 1.0
                        edge_cost = self.cached_cost[ny, nx] # Lower is better (1.0 = on edge)
                        
                        # Weight edge cost heavily
                        new_cost = cost_so_far[(cx, cy)] + (edge_cost * move_cost)
                        
                        if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                            cost_so_far[(nx, ny)] = new_cost
                            # Heuristic: Euclidean distance to end
                            heuristic = math.sqrt((end_px-nx)**2 + (end_py-ny)**2)
                            priority = new_cost + heuristic
                            heapq.heappush(pq, (priority, nx, ny))
                            came_from[(nx, ny)] = (cx, cy)
            
            if not found and best_node is not None:
                # Timeout: Use partial path to closest point
                # This prevents "AI giving up" feeling
                end_px, end_py = best_node
                found = True
                
                # Feedback to user
                if self.iface:
                    self.iface.messageBar().pushMessage(
                        "ArchaeoTrace",
                        "Pathfinding timeout - simplified path used (Try zooming in)",
                        Qgis.Warning,
                        3
                    )
                else:
                    # Fallback if no iface
                    print("Pathfinding timeout")
            
            if found:
                # Reconstruct path
                path = []
                curr = (end_px, end_py)
                while curr != (start_px, start_py):
                    path.append(curr)
                    curr = came_from.get(curr)
                    if curr is None: break
                
                path.reverse()
                
                # Apply 5-point Moving Average Smoothing (Anti-Aliasing)
                # This converts integer grid steps into smooth float coordinates
                smoothed_path = []
                window_size = 5
                
                if len(path) > window_size:
                    path_arr = np.array(path)
                    for i in range(len(path)):
                        # Simple moving average window
                        start_idx = max(0, i - window_size // 2)
                        end_idx = min(len(path), i + window_size // 2 + 1)
                        # Mean of x and y coordinates
                        avg_pt = np.mean(path_arr[start_idx:end_idx], axis=0)
                        smoothed_path.append(avg_pt)
                else:
                    smoothed_path = path

                # Convert pixels to map points (subsample for performance)
                path_map = []
                for i, pt in enumerate(smoothed_path):
                    # Take every 2nd point (High Quality, resolved "Straight Line" issue)
                    if i % 2 == 0 or i == len(smoothed_path)-1: 
                        # Pass float coordinates for sub-pixel precision
                        path_map.append(self.pixel_to_map(pt[0], pt[1]))
                        
                return path_map
            else:
                # If path not found (timeout), return straight line
                return [target_point]
                
        except Exception as e:
            # print(f"A* Error: {e}")
            return [target_point]

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
            
            # Generate Cost Map for Path Finding
            self.cached_cost = self.edge_detector.get_edge_cost_map(self.cached_edges)
            
        except Exception as e:
            print(f"Edge cache error: {e}")
            self.cached_edges = None
            self.cached_cost = None

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
        smoothed = self.smooth_bezier(self.path_points, closed=closed)
        
        # Create geometry
        # ALWAYS use LineString. For closed loops, just make start==end.
        if closed and len(smoothed) >= 3:
            # Add first point to end to close the loop
            smoothed.append(smoothed[0])
            
        geom = QgsGeometry.fromPolylineXY(smoothed)
        
        # Heavy QGIS simplification for ultra-smooth result
        tolerance = self.canvas.mapUnitsPerPixel() * 6
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

    def smooth_bezier(self, points, closed=False):
        """
        Smooth points using Bézier-like curve fitting (Chaikin).
        Handles closed polygons correctly to prevent flattened ends.
        """
        if len(points) < 3:
            return points
        
        # Convert to numpy for easier math
        pts = np.array([[p.x(), p.y()] for p in points])
        
        # Apply Chaikin's algorithm 2 times for "smooth but straight" look (Polygonal style)
        for _ in range(2):
            if len(pts) < 3:
                break
            
            new_pts = []
            
            # If NOT closed, keep first point
            if not closed:
                new_pts.append(pts[0])
            
            # Loop segments
            count = len(pts) if closed else len(pts) - 1
            
            for i in range(count):
                p0 = pts[i]
                p1 = pts[(i + 1) % len(pts)]
                
                # 1/4 and 3/4 points (Standard Chaikin)
                q = p0 * 0.75 + p1 * 0.25
                r = p0 * 0.25 + p1 * 0.75
                new_pts.extend([q, r])
            
            # If NOT closed, keep last point
            if not closed:
                new_pts.append(pts[-1])
                
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
        self.preview_path = []
        self.checkpoints = []
        self.start_point = None
        self.last_map_point = None
        self.preview_band.reset(QgsWkbTypes.LineGeometry)
        self.confirm_band.reset(QgsWkbTypes.LineGeometry)
        self.start_marker.reset(QgsWkbTypes.PointGeometry)
        self.close_indicator.reset(QgsWkbTypes.PointGeometry)
        self.checkpoint_markers.reset(QgsWkbTypes.PointGeometry)

    def activate(self):
        """Called when tool is activated."""
        self.update_edge_cache()
        try:
            self.canvas.extentsChanged.connect(self.update_edge_cache)
        except:
            pass
        super().activate()

    def deactivate(self):
        """Called when tool is deactivated."""
        try:
            self.canvas.extentsChanged.disconnect(self.update_edge_cache)
        except:
            pass
        self.reset_tracing()
        super().deactivate()
        self.deactivated.emit()
