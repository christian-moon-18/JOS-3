"""
Export functionality for images, videos, and 3D models

Implements: PRD Section 3.3 - Export Capabilities
"""

from .image_export import ImageExporter, create_filename_pattern, get_journal_requirements

# Optional imports based on dependencies
try:
    from .video_export import VideoExporter
    VIDEO_EXPORT_AVAILABLE = True
except ImportError:
    VIDEO_EXPORT_AVAILABLE = False

try:
    from .model_export import ModelExporter
    MODEL_EXPORT_AVAILABLE = True
except ImportError:
    MODEL_EXPORT_AVAILABLE = False

# Base exports always available
__all__ = ['ImageExporter', 'create_filename_pattern', 'get_journal_requirements']

# Add conditional exports
if VIDEO_EXPORT_AVAILABLE:
    __all__.append('VideoExporter')

if MODEL_EXPORT_AVAILABLE:
    __all__.append('ModelExporter')