# -*- coding: utf-8 -*-
"""
Edge Detection Module for AI Vectorizer
Optimized for detecting dark contour lines in historical maps.
"""

import cv2
import numpy as np

class EdgeDetector:
    def __init__(self):
        """Initialize Edge Detector."""
        pass

    def detect_edges(self, image: np.ndarray, low_threshold=30, high_threshold=100) -> np.ndarray:
        """
        Detect edges using adaptive method for historical maps.
        Combines Canny with dark line extraction.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 1. Extract dark pixels (contour lines are usually dark)
        # Adaptive threshold to handle uneven lighting
        dark_mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 10
        )
        
        # 2. Also run Canny for fine edges
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        canny = cv2.Canny(blurred, low_threshold, high_threshold)
        
        # 3. Combine both methods
        combined = cv2.bitwise_or(dark_mask, canny)
        
        # 4. Clean up noise
        kernel = np.ones((2, 2), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        return combined

    def detect_dark_lines(self, image: np.ndarray, threshold=128) -> np.ndarray:
        """
        Simple dark line detection - extracts pixels darker than threshold.
        Good for maps where contour lines are clearly darker than background.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Pixels darker than threshold are considered "lines"
        _, dark_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        
        return dark_mask

    def get_edge_cost_map(self, edges: np.ndarray, edge_weight: float = 1.0) -> np.ndarray:
        """
        Create a cost map for pathfinding. 
        Edges (white pixels) have LOW cost, background HIGH cost.
        
        Args:
            edges: Binary edge map
            edge_weight: 0.0 = free draw, 1.0 = strict edge follow
        """
        # Distance to nearest edge pixel
        inverted = cv2.bitwise_not(edges)
        dist = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)
        
        # Lower edge_weight = more freedom to go off-edge
        # Higher edge_weight = strict edge following
        multiplier = 0.5 + edge_weight * 3.0  # Range: 0.5 ~ 3.5
        cost_map = 1.0 + dist * multiplier
        
        return cost_map.astype(np.float32)
