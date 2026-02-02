# -*- coding: utf-8 -*-
"""
Edge Detection Module - Multiple detection methods for historical maps
Supports: Canny+Adaptive, LSD Line Detector
"""

import cv2
import numpy as np

class EdgeDetector:
    def __init__(self, method='canny'):
        """
        Args:
            method: 'canny' or 'lsd'
        """
        self.method = method
        
        # LSD detector instance
        if method == 'lsd':
            self.lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)

    def detect_edges(self, image: np.ndarray, low_threshold=30, high_threshold=100) -> np.ndarray:
        """
        Detect edges using selected method.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if self.method == 'lsd':
            return self._detect_lsd(gray)
        else:
            return self._detect_canny(gray, low_threshold, high_threshold)

    def _detect_canny(self, gray: np.ndarray, low_threshold=30, high_threshold=100) -> np.ndarray:
        """Canny + Adaptive threshold method."""
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

    def _detect_lsd(self, gray: np.ndarray) -> np.ndarray:
        """
        LSD Line Segment Detector - detects line segments directly.
        Much smoother and more accurate for contour lines.
        """
        # Detect line segments
        lines, widths, precs, nfas = self.lsd.detect(gray)
        
        # Create edge mask from detected lines
        edge_mask = np.zeros(gray.shape, dtype=np.uint8)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0].astype(int)
                # Draw thicker lines for better pathfinding
                cv2.line(edge_mask, (x1, y1), (x2, y2), 255, 2)
        
        # Also add dark line detection for completeness
        dark_mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 8
        )
        
        # Combine LSD with dark detection
        combined = cv2.bitwise_or(edge_mask, dark_mask)
        
        # Morphological closing for continuity
        kernel = np.ones((3, 3), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        return combined

    def get_edge_cost_map(self, edges: np.ndarray, edge_weight: float = 0.5) -> np.ndarray:
        """
        Create cost map for pathfinding.
        edge_weight: 0.0 = free draw, 1.0 = strict edge follow
        """
        # Distance to nearest edge
        inverted = cv2.bitwise_not(edges)
        dist = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)
        
        # Lower multiplier = more freedom
        multiplier = 0.1 + edge_weight * 0.9  # Range: 0.1 ~ 1.0
        cost_map = 1.0 + dist * multiplier
        
        return cost_map.astype(np.float32)
