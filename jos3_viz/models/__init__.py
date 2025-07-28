"""
3D model and body segment definitions

Implements: TDD Section 3.3 - 3D Model Generation
"""

from .body_segments import BODY_SEGMENTS
from .scaling import get_segment_mass, SEGMENT_MASS_PERCENTAGES

__all__ = ['BODY_SEGMENTS', 'get_segment_mass', 'SEGMENT_MASS_PERCENTAGES']