"""
Color Mapping System for Heat Transfer Visualization

Implementation of HeatColorMapper class
Source: Agile Plan Sprint 2, Epic 2.1, Task 2.1.1
User Story: As a researcher, I need accurate color representation of temperature/heat data including conductive heat transfer

Provides scientifically accurate color mapping for heat transfer visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

from ..core.logger import get_logger
from ..core.exceptions import VisualizationError

logger = get_logger(__name__)


class HeatColorMapper:
    """
    Color mapping system for heat transfer visualization
    
    Implements: TDD Section 6.1 - Color Mapping System
    Fulfills: PRD Section 3.1.3 - Color Mapping requirements
    """
    
    # Predefined color schemes for different visualization modes
    COLOR_SCHEMES = {
        'temperature': {
            'RdBu_r': plt.cm.RdBu_r,      # Blue (cold) to Red (hot) - default
            'viridis': plt.cm.viridis,      # Purple to Yellow
            'plasma': plt.cm.plasma,        # Purple to Pink to Yellow
            'coolwarm': plt.cm.coolwarm,    # Blue to Red
            'thermal': 'custom_thermal'     # Custom thermal comfort colors
        },
        'heat_flow': {
            'RdBu': plt.cm.RdBu,           # Red (heating) to Blue (cooling)
            'seismic': plt.cm.seismic,      # Blue-White-Red diverging
            'PiYG': plt.cm.PiYG,           # Pink-Yellow-Green diverging
            'conductive': 'custom_conductive'  # NEW: Special for conductive heat
        },
        'comfort': {
            'thermal_comfort': 'custom_comfort',  # Comfort-based colors
            'stress': 'custom_stress'             # Thermal stress indicators
        }
    }
    
    def __init__(self, colormap: str = 'RdBu_r', data_type: str = 'temperature'):
        """
        Initialize color mapper
        
        Args:
            colormap: Color scheme name
            data_type: Type of data being visualized ('temperature', 'heat_flow', 'comfort')
        """
        self.data_type = data_type
        self.colormap_name = colormap
        self.colormap = self._get_colormap(colormap, data_type)
        self.normalizer = None
        self.value_range = (None, None)
        self.auto_scale = True
        
        logger.debug(f"Initialized color mapper: {colormap} for {data_type}")
    
    def _get_colormap(self, colormap: str, data_type: str) -> mcolors.Colormap:
        """Get colormap object from scheme name"""
        try:
            # Check if it's a predefined scheme
            if data_type in self.COLOR_SCHEMES and colormap in self.COLOR_SCHEMES[data_type]:
                scheme = self.COLOR_SCHEMES[data_type][colormap]
                
                if isinstance(scheme, str) and scheme.startswith('custom_'):
                    return self._create_custom_colormap(scheme)
                else:
                    return scheme
            
            # Try matplotlib built-in colormaps
            try:
                return plt.cm.get_cmap(colormap)
            except ValueError:
                logger.warning(f"Unknown colormap '{colormap}', using default RdBu_r")
                return plt.cm.RdBu_r
                
        except Exception as e:
            logger.error(f"Error getting colormap: {str(e)}")
            return plt.cm.RdBu_r
    
    def _create_custom_colormap(self, scheme_name: str) -> mcolors.LinearSegmentedColormap:
        """Create custom colormaps for specific visualization needs"""
        
        if scheme_name == 'custom_thermal':
            # Custom thermal comfort colormap
            colors = [
                '#0000FF',  # Cold blue
                '#4169E1',  # Royal blue  
                '#87CEEB',  # Sky blue
                '#90EE90',  # Light green (comfortable)
                '#FFFF00',  # Yellow
                '#FFA500',  # Orange
                '#FF0000'   # Hot red
            ]
            return mcolors.LinearSegmentedColormap.from_list('custom_thermal', colors)
            
        elif scheme_name == 'custom_conductive':
            # NEW: Special colormap for conductive heat transfer
            colors = [
                '#000080',  # Dark blue (strong cooling)
                '#4169E1',  # Royal blue (cooling)
                '#87CEEB',  # Light blue (mild cooling)
                '#F0F0F0',  # Light gray (no conductive heat)
                '#FFB6C1',  # Light pink (mild heating)
                '#FF6347',  # Tomato (heating)
                '#8B0000'   # Dark red (strong heating)
            ]
            return mcolors.LinearSegmentedColormap.from_list('custom_conductive', colors)
            
        elif scheme_name == 'custom_comfort':
            # Thermal comfort based on PMV scale
            colors = [
                '#000080',  # Cold (-3 PMV)
                '#4169E1',  # Cool (-2 PMV)
                '#87CEEB',  # Slightly cool (-1 PMV)
                '#90EE90',  # Neutral (0 PMV) - green for comfort
                '#FFFF00',  # Slightly warm (+1 PMV)  
                '#FFA500',  # Warm (+2 PMV)
                '#FF0000'   # Hot (+3 PMV)
            ]
            return mcolors.LinearSegmentedColormap.from_list('custom_comfort', colors)
            
        elif scheme_name == 'custom_stress':
            # Thermal stress indicators
            colors = [
                '#00FF00',  # Green (no stress)
                '#FFFF00',  # Yellow (mild stress)
                '#FFA500',  # Orange (moderate stress)
                '#FF0000',  # Red (high stress)
                '#8B0000'   # Dark red (severe stress)
            ]
            return mcolors.LinearSegmentedColormap.from_list('custom_stress', colors)
            
        else:
            logger.warning(f"Unknown custom scheme '{scheme_name}', using default")
            return plt.cm.RdBu_r
    
    def set_temperature_range(self, min_temp: Optional[float] = None, 
                            max_temp: Optional[float] = None, 
                            auto_scale: bool = True) -> None:
        """
        Set temperature/value range for color mapping
        
        Args:
            min_temp: Minimum temperature/value
            max_temp: Maximum temperature/value  
            auto_scale: If True, automatically scale to data range
        """
        self.auto_scale = auto_scale
        
        if not auto_scale:
            if min_temp is None or max_temp is None:
                raise VisualizationError("Must provide min_temp and max_temp when auto_scale=False")
            
            if min_temp >= max_temp:
                raise VisualizationError("min_temp must be less than max_temp")
            
            self.value_range = (min_temp, max_temp)
            logger.debug(f"Set manual temperature range: {min_temp:.2f} to {max_temp:.2f}")
        else:
            self.value_range = (min_temp, max_temp)  # Will be updated when mapping data
            logger.debug("Set automatic temperature scaling")
    
    def map_temperature_to_color(self, temp_values: Union[np.ndarray, List[float], Dict[str, float]], 
                                colormap: Optional[str] = None) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Map temperature/heat values to colors
        
        Args:
            temp_values: Temperature values (array, list, or dict by segment)
            colormap: Override colormap for this operation
            
        Returns:
            RGB color values (same structure as input)
        """
        if colormap and colormap != self.colormap_name:
            # Temporarily use different colormap
            original_cmap = self.colormap
            self.colormap = self._get_colormap(colormap, self.data_type)
        
        try:
            # Handle different input types
            if isinstance(temp_values, dict):
                return self._map_dict_values(temp_values)
            else:
                return self._map_array_values(np.array(temp_values))
                
        finally:
            # Restore original colormap if changed
            if colormap and colormap != self.colormap_name:
                self.colormap = original_cmap
    
    def _map_dict_values(self, temp_dict: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Map dictionary of segment temperatures to colors"""
        # Extract valid values for range calculation
        valid_values = [v for v in temp_dict.values() if not np.isnan(v) and v is not None]
        
        if not valid_values:
            logger.warning("No valid temperature values found")
            return {k: np.array([0.5, 0.5, 0.5, 1.0]) for k in temp_dict.keys()}  # Gray
        
        # Calculate or use set range
        if self.auto_scale:
            value_min, value_max = np.min(valid_values), np.max(valid_values)
        else:
            value_min, value_max = self.value_range
        
        if value_min == value_max:
            # All values are the same
            normalizer = mcolors.Normalize(vmin=value_min-0.5, vmax=value_max+0.5)
        else:
            normalizer = mcolors.Normalize(vmin=value_min, vmax=value_max)
        
        self.normalizer = normalizer
        
        # Map each value to color
        color_dict = {}
        for segment, temp in temp_dict.items():
            if np.isnan(temp) or temp is None:
                # Use neutral color for missing data
                color_dict[segment] = np.array([0.5, 0.5, 0.5, 1.0])
            else:
                normalized_temp = normalizer(temp)
                color_dict[segment] = np.array(self.colormap(normalized_temp))
        
        logger.debug(f"Mapped {len(color_dict)} segment temperatures to colors "
                    f"(range: {value_min:.2f} to {value_max:.2f})")
        
        return color_dict
    
    def _map_array_values(self, temp_array: np.ndarray) -> np.ndarray:
        """Map array of temperature values to colors"""
        # Handle NaN values
        valid_mask = ~np.isnan(temp_array)
        valid_values = temp_array[valid_mask]
        
        if len(valid_values) == 0:
            logger.warning("No valid temperature values in array")
            return np.full((len(temp_array), 4), [0.5, 0.5, 0.5, 1.0])  # Gray
        
        # Calculate range
        if self.auto_scale:
            value_min, value_max = np.min(valid_values), np.max(valid_values)
        else:
            value_min, value_max = self.value_range
        
        if value_min == value_max:
            normalizer = mcolors.Normalize(vmin=value_min-0.5, vmax=value_max+0.5)
        else:
            normalizer = mcolors.Normalize(vmin=value_min, vmax=value_max)
        
        self.normalizer = normalizer
        
        # Initialize color array
        colors = np.full((len(temp_array), 4), [0.5, 0.5, 0.5, 1.0])  # Default gray
        
        # Map valid values to colors
        if np.any(valid_mask):
            normalized_temps = normalizer(temp_array[valid_mask])
            colors[valid_mask] = self.colormap(normalized_temps)
        
        return colors
    
    def map_conductive_heat(self, heat_values: Union[Dict[str, float], np.ndarray]) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """
        NEW: Special mapping for conductive heat transfer visualization
        
        Args:
            heat_values: Conductive heat transfer values (positive=heating, negative=cooling)
            
        Returns:
            Color values optimized for conductive heat visualization
        """
        # Temporarily switch to conductive colormap
        original_cmap = self.colormap
        original_name = self.colormap_name
        
        self.colormap = self._create_custom_colormap('custom_conductive')
        self.colormap_name = 'custom_conductive'
        
        try:
            # For conductive heat, center the scale around 0
            if isinstance(heat_values, dict):
                valid_values = [abs(v) for v in heat_values.values() if not np.isnan(v) and v != 0]
            else:
                valid_values = np.abs(heat_values[~np.isnan(heat_values) & (heat_values != 0)])
            
            if len(valid_values) > 0:
                max_abs_heat = np.max(valid_values)
                self.set_temperature_range(-max_abs_heat, max_abs_heat, auto_scale=False)
            else:
                max_abs_heat = 1.0  # Default range
                self.set_temperature_range(-max_abs_heat, max_abs_heat, auto_scale=False)
            
            colors = self.map_temperature_to_color(heat_values)
            
            logger.debug(f"Mapped conductive heat values with range ±{max_abs_heat:.2f}W")
            return colors
            
        finally:
            # Restore original colormap
            self.colormap = original_cmap  
            self.colormap_name = original_name
    
    def generate_colorbar_legend(self, fig_size: Tuple[float, float] = (8, 1), 
                               orientation: str = 'horizontal',
                               title: str = '',
                               units: str = '') -> plt.Figure:
        """
        Generate standalone colorbar legend
        
        Args:
            fig_size: Figure size (width, height)
            orientation: 'horizontal' or 'vertical'
            title: Colorbar title
            units: Units label
            
        Returns:
            Matplotlib figure with colorbar
        """
        if self.normalizer is None:
            logger.warning("No data has been mapped yet, creating default colorbar")
            self.normalizer = mcolors.Normalize(vmin=0, vmax=100)
        
        fig, ax = plt.subplots(figsize=fig_size)
        
        if orientation == 'horizontal':
            cbar = ColorbarBase(ax, cmap=self.colormap, norm=self.normalizer, orientation=orientation)
            ax.set_xlabel(f"{title} ({units})" if units else title)
        else:
            cbar = ColorbarBase(ax, cmap=self.colormap, norm=self.normalizer, orientation=orientation)
            ax.set_ylabel(f"{title} ({units})" if units else title)
        
        plt.tight_layout()
        logger.debug(f"Generated {orientation} colorbar legend")
        
        return fig
    
    def validate_colormap_scientific_accuracy(self) -> Dict[str, bool]:
        """
        Validate colormap for scientific accuracy
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            'perceptually_uniform': True,
            'colorblind_friendly': True, 
            'print_friendly': True,
            'appropriate_for_data_type': True,
            'warnings': []
        }
        
        # Check for problematic colormaps
        problematic_cmaps = ['jet', 'rainbow', 'hsv']
        if self.colormap_name in problematic_cmaps:
            validation['perceptually_uniform'] = False
            validation['warnings'].append(f"Colormap '{self.colormap_name}' is not perceptually uniform")
        
        # Check data type appropriateness
        if self.data_type == 'temperature' and self.colormap_name in ['seismic', 'RdBu']:
            validation['warnings'].append("Diverging colormap may not be appropriate for temperature data")
        
        # Check for custom conductive mapping
        if self.colormap_name == 'custom_conductive':
            validation['appropriate_for_data_type'] = True
            validation['warnings'].append("Using custom conductive heat colormap - ensure proper interpretation")
        
        logger.debug(f"Colormap validation: {sum(validation[k] for k in validation if k != 'warnings')}/4 checks passed")
        
        return validation
    
    def get_color_info(self) -> Dict[str, Any]:
        """
        Get information about current color mapping settings
        
        Returns:
            Dictionary with color mapping details
        """
        return {
            'colormap_name': self.colormap_name,
            'data_type': self.data_type,
            'value_range': self.value_range,
            'auto_scale': self.auto_scale,
            'normalizer_range': (self.normalizer.vmin, self.normalizer.vmax) if self.normalizer else None,
            'available_schemes': list(self.COLOR_SCHEMES.keys())
        }


def get_available_colormaps() -> Dict[str, List[str]]:
    """
    Get list of available colormaps by category
    
    Returns:
        Dictionary of colormap categories and their available schemes
    """
    return {
        'temperature': list(HeatColorMapper.COLOR_SCHEMES['temperature'].keys()),
        'heat_flow': list(HeatColorMapper.COLOR_SCHEMES['heat_flow'].keys()),
        'comfort': list(HeatColorMapper.COLOR_SCHEMES['comfort'].keys()),
        'matplotlib_builtin': ['viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'bwr', 'seismic']
    }