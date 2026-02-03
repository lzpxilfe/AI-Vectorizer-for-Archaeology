# -*- coding: utf-8 -*-
"""
Edge Detection Module - Multiple detection methods for historical maps
Supports: Canny+Adaptive, LSD Line Detector, HED (Holistically-Nested Edge Detection)
"""

import cv2
import numpy as np
import os

# HED model paths
HED_PROTOTXT = os.path.join(os.path.dirname(__file__), 'models', 'hed_deploy.prototxt')
HED_CAFFEMODEL = os.path.join(os.path.dirname(__file__), 'models', 'hed_pretrained_bsds.caffemodel')

class EdgeDetector:
    def __init__(self, method='canny'):
        """
        Args:
            method: 'canny', 'lsd', or 'hed'
        """
        self.method = method
        self.hed_net = None
        
        # LSD detector instance
        if method == 'lsd':
            self.lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        
        # HED network
        if method == 'hed':
            self._init_hed()

    def _init_hed(self):
        """Initialize HED network if available."""
        try:
            if os.path.exists(HED_PROTOTXT) and os.path.exists(HED_CAFFEMODEL):
                self.hed_net = cv2.dnn.readNetFromCaffe(HED_PROTOTXT, HED_CAFFEMODEL)
                print("HED model loaded successfully")
            else:
                print(f"HED model files not found. Will fallback to Canny.")
                print(f"Expected: {HED_PROTOTXT}")
                print(f"Expected: {HED_CAFFEMODEL}")
        except Exception as e:
            print(f"HED init error: {e}")
            self.hed_net = None

    def detect_edges(self, image: np.ndarray, low_threshold=30, high_threshold=100) -> np.ndarray:
        """
        Detect edges using selected method.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            color = image
        else:
            gray = image
            color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if self.method == 'lsd':
            return self._detect_lsd(gray)
        elif self.method == 'hed':
            return self._detect_hed(color, gray)
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

    def _detect_hed(self, color: np.ndarray, gray: np.ndarray) -> np.ndarray:
        """
        HED (Holistically-Nested Edge Detection) - Deep learning based.
        Produces smoother, more natural edges than traditional methods.
        """
        if self.hed_net is None:
            # Fallback to Canny if HED not available
            print("HED not available, falling back to Canny")
            return self._detect_canny(gray)
        
        try:
            h, w = color.shape[:2]
            
            # Prepare input blob
            # HED expects specific preprocessing
            blob = cv2.dnn.blobFromImage(
                color, 
                scalefactor=1.0, 
                size=(w, h),
                mean=(104.00698793, 116.66876762, 122.67891434),
                swapRB=False, 
                crop=False
            )
            
            self.hed_net.setInput(blob)
            hed_output = self.hed_net.forward()
            
            # Post-process: convert to 0-255 range
            hed_edges = hed_output[0, 0]
            hed_edges = cv2.resize(hed_edges, (w, h))
            hed_edges = (255 * hed_edges).astype(np.uint8)
            
            # Threshold to binary
            _, binary = cv2.threshold(hed_edges, 50, 255, cv2.THRESH_BINARY)
            
            # Optional: thin the edges
            kernel = np.ones((2, 2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            return binary
            
        except Exception as e:
            print(f"HED detection error: {e}")
            return self._detect_canny(gray)

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

    @staticmethod
    def is_hed_available():
        """Check if HED model files are available."""
        return os.path.exists(HED_PROTOTXT) and os.path.exists(HED_CAFFEMODEL)

    @staticmethod
    def get_hed_download_info():
        """Get info about downloading HED model."""
        return {
            'prototxt_url': 'https://raw.githubusercontent.com/s9xie/hed/master/examples/hed/deploy.prototxt',
            'caffemodel_url': 'https://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel',
            'prototxt_path': HED_PROTOTXT,
            'caffemodel_path': HED_CAFFEMODEL,
            'size_mb': 56
        }
