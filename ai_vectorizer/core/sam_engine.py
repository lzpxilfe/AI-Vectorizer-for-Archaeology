# -*- coding: utf-8 -*-
"""
SAM Engine Module for AI Vectorizer
Uses MobileSAM for efficient segmentation on historical maps.
"""

import os
import numpy as np
import requests

try:
    import torch
    from mobile_sam import sam_model_registry, SamPredictor
    MOBILE_SAM_AVAILABLE = True
except ImportError:
    MOBILE_SAM_AVAILABLE = False
    print("MobileSAM or Torch not installed. Standard Mode unavailable.")

class SAMEngine:
    def __init__(self, model_type="vit_t", device=None):
        """
        Initialize SAM Engine.
        
        Args:
            model_type (str): 'vit_t' for MobileSAM.
            device (str): 'cuda' or 'cpu'. Auto-detect if None.
        """
        self.predictor = None
        self.is_ready = False
        
        if not MOBILE_SAM_AVAILABLE:
            return

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model_type = model_type
        # Default weights path (in models/ directory under plugin)
        self.weights_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'models', 
            'mobile_sam.pt'
        )

    def load_model(self):
        """Load the MobileSAM model from weights file."""
        if not MOBILE_SAM_AVAILABLE:
            return False, "MobileSAM library not installed."
            
        if not os.path.exists(self.weights_path):
            return False, f"Model weights not found at {self.weights_path}"
            
        try:
            sam = sam_model_registry[self.model_type](checkpoint=self.weights_path)
            sam.to(device=self.device)
            sam.eval()
            self.predictor = SamPredictor(sam)
            self.is_ready = True
            return True, "Model loaded successfully."
        except Exception as e:
            return False, f"Error loading model: {str(e)}"

    def download_weights(self):
        """Download MobileSAM weights if missing."""
        url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
        
        models_dir = os.path.dirname(self.weights_path)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        try:
            print(f"Downloading MobileSAM weights from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(self.weights_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False

    def set_image(self, image: np.ndarray):
        """
        Set image for embedding calculation.
        
        Args:
            image (np.ndarray): RGB image (H, W, 3).
        """
        if self.predictor:
            self.predictor.set_image(image)

    def predict_point(self, points, labels):
        """
        Predict mask from point prompts.
        
        Args:
            points (np.ndarray or list): [[x, y], ...]
            labels (np.ndarray or list): [1, 0, ...] (1: fg, 0: bg)
            
        Returns:
            mask (np.ndarray): Best binary mask.
        """
        if not self.predictor:
            return None
            
        masks, scores, logits = self.predictor.predict(
            point_coords=np.array(points),
            point_labels=np.array(labels),
            multimask_output=False # We want the best single mask for contour
        )
        
        # masks shape: (1, H, W) -> return (H, W)
        return masks[0]
