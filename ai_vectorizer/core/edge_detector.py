# -*- coding: utf-8 -*-
"""
Edge Detection Module - Optimized for historical maps with adjustable strictness
"""

import cv2
import numpy as np

class EdgeDetector:
    def __init__(self):
        pass

    def detect_edges(self, image: np.ndarray, low_threshold=30, high_threshold=100) -> np.ndarray:
        """
        Detect edges using adaptive method for historical maps.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Adaptive threshold for dark lines
        dark_mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 10
        )
        
        # Canny for fine edges
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        canny = cv2.Canny(blurred, low_threshold, high_threshold)
        
        # Combine
        combined = cv2.bitwise_or(dark_mask, canny)
        
        # Clean up
        kernel = np.ones((2, 2), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        return combined

    def get_edge_cost_map(self, edges: np.ndarray, edge_weight: float = 0.5) -> np.ndarray:
        """
        Create cost map with MUCH lower penalties for non-edge areas.
        This prevents the "glass wall" effect.
        
        edge_weight: 0.0 = almost no edge influence, 1.0 = strong edge influence
        """
        # Distance to nearest edge
        inverted = cv2.bitwise_not(edges)
        dist = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)
        
        # SIGNIFICANTLY reduced cost difference
        # Old: 1.0 + dist * 3.0 (big difference = wall effect)
        # New: 1.0 + dist * (0.1 to 1.0) (small difference = more freedom)
        multiplier = 0.1 + edge_weight * 0.9  # Range: 0.1 ~ 1.0
        cost_map = 1.0 + dist * multiplier
        
        return cost_map.astype(np.float32)
