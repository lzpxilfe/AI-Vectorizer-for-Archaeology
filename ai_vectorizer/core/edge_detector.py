# -*- coding: utf-8 -*-
"""
Edge Detection Module - Multiple detection methods for historical maps
Supports: Canny+Adaptive, LSD Line Detector, HED (Holistically-Nested Edge Detection)
"""

import cv2
import numpy as np
import os
from skimage.morphology import skeletonize

class EdgeDetector:
    METHOD_CANNY = 'canny'
    METHOD_LSD = 'lsd'
    METHOD_HED = 'hed'

    EDGE_MAX_VALUE = 255
    EDGE_PRESENCE_THRESHOLD = 50

    DEFAULT_CANNY_LOW_THRESHOLD = 30
    DEFAULT_CANNY_HIGH_THRESHOLD = 100
    CANNY_ADAPTIVE_BLOCK_SIZE = 21
    CANNY_ADAPTIVE_C = 10
    CANNY_BLUR_KERNEL = (3, 3)
    CANNY_CLOSE_KERNEL = (2, 2)

    LSD_LINE_WIDTH = 2
    LSD_ADAPTIVE_BLOCK_SIZE = 21
    LSD_ADAPTIVE_C = 8
    LSD_CLOSE_KERNEL = (3, 3)

    HED_MEAN = (104.00698793, 116.66876762, 122.67891434)
    HED_BINARY_THRESHOLD = 50
    HED_CLOSE_KERNEL = (2, 2)

    EDGE_COST_BASE_MULTIPLIER = 0.1
    EDGE_COST_WEIGHT_SCALE = 0.9
    DIST_TRANSFORM_MASK_SIZE = 5

    HED_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
    HED_PROTOTXT = os.path.join(HED_MODEL_DIR, 'hed_deploy.prototxt')
    HED_CAFFEMODEL = os.path.join(HED_MODEL_DIR, 'hed_pretrained_bsds.caffemodel')
    HED_PROTOTXT_URL = 'https://raw.githubusercontent.com/s9xie/hed/master/examples/hed/deploy.prototxt'
    HED_CAFFEMODEL_URL = 'https://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel'
    HED_MODEL_SIZE_MB = 56

    def __init__(self, method=METHOD_CANNY):
        """
        Args:
            method: 'canny', 'lsd', or 'hed'
        """
        self.method = method
        self.hed_net = None
        
        # LSD detector instance
        if method == self.METHOD_LSD:
            self.lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        
        # HED network
        if method == self.METHOD_HED:
            self._init_hed()

    def _init_hed(self):
        """Initialize HED network if available."""
        try:
            if os.path.exists(self.HED_PROTOTXT) and os.path.exists(self.HED_CAFFEMODEL):
                self.hed_net = cv2.dnn.readNetFromCaffe(self.HED_PROTOTXT, self.HED_CAFFEMODEL)
                print("HED model loaded successfully")
            else:
                print(f"HED model files not found. Will fallback to Canny.")
                print(f"Expected: {self.HED_PROTOTXT}")
                print(f"Expected: {self.HED_CAFFEMODEL}")
        except Exception as e:
            print(f"HED init error: {e}")
            self.hed_net = None

    def detect_edges(
        self,
        image: np.ndarray,
        low_threshold=DEFAULT_CANNY_LOW_THRESHOLD,
        high_threshold=DEFAULT_CANNY_HIGH_THRESHOLD,
    ) -> np.ndarray:
        """
        Detect edges using selected method.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            color = image
        else:
            gray = image
            color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if self.method == self.METHOD_LSD:
            edges = self._detect_lsd(gray)
        elif self.method == self.METHOD_HED:
            edges = self._detect_hed(color, gray)
        else:
            edges = self._detect_canny(gray, low_threshold, high_threshold)
            
        # SKELETONIZATION: Ensure edges are 1px wide
        # This prevents "walking inside the edge" and reduces jitter
        try:
            # Normalize to binary (0 or 1)
            binary = edges > self.EDGE_PRESENCE_THRESHOLD
            
            # Skeletonize
            skeleton = skeletonize(binary)
            
            # Convert back to uint8 (0 or 255)
            edges = (skeleton * self.EDGE_MAX_VALUE).astype(np.uint8)
        except Exception as e:
            print(f"Skeletonize error: {e}")
            
        return edges

    def _detect_canny(
        self,
        gray: np.ndarray,
        low_threshold=DEFAULT_CANNY_LOW_THRESHOLD,
        high_threshold=DEFAULT_CANNY_HIGH_THRESHOLD,
    ) -> np.ndarray:
        """Canny + Adaptive threshold method."""
        # Adaptive threshold for dark lines
        dark_mask = cv2.adaptiveThreshold(
            gray,
            self.EDGE_MAX_VALUE,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.CANNY_ADAPTIVE_BLOCK_SIZE,
            self.CANNY_ADAPTIVE_C,
        )
        
        # Canny for fine edges
        blurred = cv2.GaussianBlur(gray, self.CANNY_BLUR_KERNEL, 0)
        canny = cv2.Canny(blurred, low_threshold, high_threshold)
        
        # Combine
        combined = cv2.bitwise_or(dark_mask, canny)
        
        # Clean up
        kernel = np.ones(self.CANNY_CLOSE_KERNEL, np.uint8)
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
                cv2.line(edge_mask, (x1, y1), (x2, y2), self.EDGE_MAX_VALUE, self.LSD_LINE_WIDTH)
        
        # Also add dark line detection for completeness
        dark_mask = cv2.adaptiveThreshold(
            gray,
            self.EDGE_MAX_VALUE,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.LSD_ADAPTIVE_BLOCK_SIZE,
            self.LSD_ADAPTIVE_C,
        )
        
        # Combine LSD with dark detection
        combined = cv2.bitwise_or(edge_mask, dark_mask)
        
        # Morphological closing for continuity
        kernel = np.ones(self.LSD_CLOSE_KERNEL, np.uint8)
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
                mean=self.HED_MEAN,
                swapRB=False, 
                crop=False
            )
            
            self.hed_net.setInput(blob)
            hed_output = self.hed_net.forward()
            
            # Post-process: convert to 0-255 range
            hed_edges = hed_output[0, 0]
            hed_edges = cv2.resize(hed_edges, (w, h))
            hed_edges = (self.EDGE_MAX_VALUE * hed_edges).astype(np.uint8)
            
            # Threshold to binary
            _, binary = cv2.threshold(
                hed_edges,
                self.HED_BINARY_THRESHOLD,
                self.EDGE_MAX_VALUE,
                cv2.THRESH_BINARY,
            )
            
            # Optional: thin the edges
            kernel = np.ones(self.HED_CLOSE_KERNEL, np.uint8)
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
        dist = cv2.distanceTransform(inverted, cv2.DIST_L2, self.DIST_TRANSFORM_MASK_SIZE)
        
        # Lower multiplier = more freedom
        multiplier = self.EDGE_COST_BASE_MULTIPLIER + edge_weight * self.EDGE_COST_WEIGHT_SCALE
        cost_map = 1.0 + dist * multiplier
        
        return cost_map.astype(np.float32)

    @classmethod
    def is_hed_available(cls):
        """Check if HED model files are available."""
        return os.path.exists(cls.HED_PROTOTXT) and os.path.exists(cls.HED_CAFFEMODEL)

    @classmethod
    def get_hed_download_info(cls):
        """Get info about downloading HED model."""
        return {
            'prototxt_url': cls.HED_PROTOTXT_URL,
            'caffemodel_url': cls.HED_CAFFEMODEL_URL,
            'prototxt_path': cls.HED_PROTOTXT,
            'caffemodel_path': cls.HED_CAFFEMODEL,
            'size_mb': cls.HED_MODEL_SIZE_MB
        }
