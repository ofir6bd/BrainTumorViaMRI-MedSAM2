"""FCM segmentation pipeline package.

Whole-tumor detection from FLAIR FCM segmentation, plus a step-through web UI.
See PIPELINE.md for the spec.
"""
from .pipeline import FcmSegmentationPipeline, PARAMS

__all__ = ["FcmSegmentationPipeline", "PARAMS"]
