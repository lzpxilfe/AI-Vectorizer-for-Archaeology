# -*- coding: utf-8 -*-
"""
Vectorizer Module for AI Vectorizer
Converts binary masks (from SAM or Edge Detection) into vector polylines.
"""

import numpy as np
import cv2
from skimage.morphology import skeletonize
from qgis.core import QgsGeometry, QgsPointXY

class Vectorizer:
    def __init__(self):
        pass

    def mask_to_line(self, mask: np.ndarray) -> QgsGeometry:
        """
        Convert a binary mask to a QgsGeometry line.
        Uses skeletonization to thin the mask to a single pixel line.
        """
        # Ensure mask is binary (0, 1) or boolean
        if mask.dtype != bool:
            mask = mask > 0
            
        # Skeletonize (Topological thinning)
        skeleton = skeletonize(mask)
        
        # Convert skeleton to points
        # skeleton is boolean matrix. True = line pixel.
        # Need to order pixels into a line string.
        
        # Find all points
        points = np.argwhere(skeleton) # (row, col) -> (y, x)
        
        if len(points) < 2:
            return None
            
        # Convert to list of (col, row) = (x, y)
        # Note: argwhere returns (row, col)
        pts = [ (p[1], p[0]) for p in points ]
        
        # Sort points to form a continuous line
        # Simple Nearest Neighbor sort
        sorted_pts = [pts[0]]
        pts.pop(0)
        
        while pts:
            last_pt = sorted_pts[-1]
            # Find closest remaining point
            # Optimization: check only limited subset if large
            dists = [ (p[0]-last_pt[0])**2 + (p[1]-last_pt[1])**2 for p in pts ]
            min_idx = np.argmin(dists)
            
            # If distance is too large, it might be a disjoint segment.
            # For now, just connect them.
            sorted_pts.append(pts[min_idx])
            pts.pop(min_idx)
            
        # Convert to QgsGeometry
        qgs_pts = [QgsPointXY(float(p[0]), float(p[1])) for p in sorted_pts]
        return QgsGeometry.fromPolylineXY(qgs_pts)

    def simplify_line(self, geometry: QgsGeometry, tolerance: float = 1.0) -> QgsGeometry:
        """Simplify the geometry using Douglas-Peucker."""
        return geometry.simplify(tolerance)
