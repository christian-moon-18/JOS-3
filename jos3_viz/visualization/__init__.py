"""
Visualization rendering components

Implements: TDD Section 3.4 - Visualization Renderer
"""

from .color_mapping import HeatColorMapper, get_available_colormaps
from .renderer_2d import HeatRenderer2D

__all__ = ['HeatColorMapper', 'HeatRenderer2D', 'get_available_colormaps']

# Only import 3D renderer if VTK is available
try:
    import vtk
    from .renderer_3d import HeatRenderer3D
    __all__.append('HeatRenderer3D')
except ImportError:
    # VTK not available - 3D renderer not accessible
    pass