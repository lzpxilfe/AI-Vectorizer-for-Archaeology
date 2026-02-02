# -*- coding: utf-8 -*-
"""
Edge Detection Module for AI Vectorizer
Uses OpenCV to detect contour lines in historical maps.
"""

import cv2
import numpy as np

class EdgeDetector:
    def __init__(self):
        """Initialize Edge Detector."""
        pass

    def detect_edges(self, image: np.ndarray, low_threshold=50, high_threshold=150) -> np.ndarray:
        """
        Detect edges using Canny edge detection.
        
        Args:
            image (np.ndarray): Input image (BGR or Grayscale).
            low_threshold (int): Lower bound for hysteresis thresholding.
            high_threshold (int): Upper bound for hysteresis thresholding.
            
        Returns:
            np.ndarray: Binary edge map (0 or 255).
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny Edge Detection
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        
        return edges

    def enhance_contour_lines(self, image: np.ndarray, lower_color=None, upper_color=None) -> np.ndarray:
        """
        Enhance contour lines based on color segmentation.
        Default colors assume brownish/blackish contours common in old maps.
        
        Args:
            image (np.ndarray): Input image (BGR).
            lower_color (tuple): Lower HSV bound (default: brown/dark).
            upper_color (tuple): Upper HSV bound.
            
        Returns:
            np.ndarray: Enhanced binary mask.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Default to removing paper background (usually light) and keeping dark lines
        # This is a simple heuristic; can be tuned.
        if lower_color is None:
            # Dark/Brownish range (very broad approximation)
            lower_color = np.array([0, 0, 0])
        if upper_color is None:
            upper_color = np.array([180, 255, 150]) # Limit brightness to filter out paper

        mask = cv2.inRange(hsv, lower_color, upper_color)
        
        # Morphological close to fill gaps
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask

    def get_edge_cost_map(self, edges: np.ndarray) -> np.ndarray:
        """
        Create a cost map for pathfinding. 
        Edges should have LOW cost, background HIGH cost.
        
        Args:
            edges (np.ndarray): Binary edge map (255=edge, 0=bg).
            
        Returns:
            np.ndarray: Cost map (float32).
        """
        # Invert edges: 0 (edge) -> 255 (bg)
        # But we want edge pixels to be low cost.
        # Let's use Distance Transform: distance to nearest non-edge pixel?
        # No, distance to nearest edge pixel is standard for "following" edges.
        # If we are ON an edge, distance is 0.
        
        # Invert: white(255) becomes black(0). 
        # So we want distance from white pixels.
        inverted = cv2.bitwise_not(edges)
        
        # Distance transform: value is distance to nearest 0 (black) pixel.
        # So we want distance to nearest edge (255 in original, 0 in inverted).
        dist = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)
        
        # Normalize/Scale cost
        # Edges (dist=0) -> Cost 1 (min move cost)
        # Far from edges -> High cost
        cost_map = 1.0 + dist * 5.0 # Weight factor can be tuned
        
        return cost_map.astype(np.float32)
