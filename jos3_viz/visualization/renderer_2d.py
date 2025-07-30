"""
2D Heat Map Renderer for JOS3 Visualization

Implementation of HeatRenderer2D class
Source: Agile Plan Sprint 2, Epic 2.2, Task 2.2.1
User Story: As a biomedical engineer, I need 2D body heat maps showing all heat transfer mechanisms

Creates 2D visualizations of heat transfer data on human body segments.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, Ellipse, FancyBboxPatch
from matplotlib.collections import PatchCollection
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

from ..core.logger import get_logger
from ..core.exceptions import VisualizationError
from ..models.body_segments import BODY_SEGMENTS, get_bilateral_segments, get_torso_segments
from .color_mapping import HeatColorMapper

logger = get_logger(__name__)


class HeatRenderer2D:
    """
    2D heat map renderer for body segment visualization
    
    Implements: TDD Section 6.2 - 2D Visualization Engine  
    Fulfills: PRD Section 3.1.1 - Humanoid Model requirements
    """
    
    # Body segment positions in 2D layout (x, y, width, height) - normalized coordinates
    SEGMENT_LAYOUT = {
        # Head and neck - top center
        'Head': (0.35, 0.85, 0.3, 0.12),
        'Neck': (0.42, 0.78, 0.16, 0.07),
        
        # Torso - center
        'Chest': (0.25, 0.55, 0.5, 0.23),
        'Back': (0.25, 0.55, 0.5, 0.23),  # Same position as chest, different visualization
        'Pelvis': (0.30, 0.40, 0.4, 0.15),
        
        # Arms - left and right of torso
        'LShoulder': (0.15, 0.65, 0.12, 0.15),
        'RShoulder': (0.73, 0.65, 0.12, 0.15),
        'LArm': (0.05, 0.45, 0.12, 0.25),
        'RArm': (0.83, 0.45, 0.12, 0.25),
        'LHand': (0.02, 0.35, 0.08, 0.12),
        'RHand': (0.90, 0.35, 0.08, 0.12),
        
        # Legs - bottom
        'LThigh': (0.32, 0.15, 0.16, 0.25),
        'RThigh': (0.52, 0.15, 0.16, 0.25),
        'LLeg': (0.34, 0.05, 0.12, 0.18),
        'RLeg': (0.54, 0.05, 0.12, 0.18),
        'LFoot': (0.30, 0.01, 0.08, 0.06),
        'RFoot': (0.62, 0.01, 0.08, 0.06)
    }
    
    # Contact area indicators for conductive heat transfer visualization
    CONTACT_INDICATORS = {
        'Chest': {'type': 'rectangle', 'outline_color': 'blue', 'line_width': 3},
        'Back': {'type': 'rectangle', 'outline_color': 'red', 'line_width': 3},
        'LArm': {'type': 'rectangle', 'outline_color': 'green', 'line_width': 2},
        'RArm': {'type': 'rectangle', 'outline_color': 'green', 'line_width': 2}
    }
    
    def __init__(self, figure_size: Tuple[float, float] = (10, 12)):
        """
        Initialize 2D heat renderer
        
        Args:
            figure_size: Figure size (width, height) in inches
        """
        self.figure_size = figure_size
        self.color_mapper = HeatColorMapper()
        self.current_figure = None
        self.current_axes = None
        self.visualization_mode = 'temperature'  # Default mode
        
        logger.debug(f"Initialized 2D heat renderer with figure size {figure_size}")
    
    def render_body_heatmap(self, heat_data: Dict[str, float], 
                          config: Optional[Dict[str, Any]] = None) -> plt.Figure:
        """
        Render 2D body heat map
        
        Args:
            heat_data: Dictionary mapping segment names to heat/temperature values
            config: Visualization configuration options
            
        Returns:
            Matplotlib figure with heat map
        """
        # Set default configuration
        default_config = {
            'colormap': 'RdBu_r',
            'title': 'Body Heat Map',
            'show_labels': True,
            'show_values': True,
            'show_colorbar': True,
            'units': '°C',
            'mode': 'temperature',
            'show_contact_areas': False
        }
        
        if config:
            default_config.update(config)
        config = default_config
        
        self.visualization_mode = config['mode']
        
        # Validate heat data
        self._validate_heat_data(heat_data)
        
        # Setup color mapper
        data_type = 'temperature' if config['mode'] == 'temperature' else 'heat_flow'
        self.color_mapper = HeatColorMapper(config['colormap'], data_type)
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=self.figure_size)
        self.current_figure = fig
        self.current_axes = ax
        
        # Map data to colors
        if config['mode'] == 'external_conduction':
            segment_colors = self.color_mapper.map_conductive_heat(heat_data)
        else:
            segment_colors = self.color_mapper.map_temperature_to_color(heat_data)
        
        # Create segment layout
        patches_list = self._create_segment_layout(segment_colors, config)
        
        # Add patches to axes
        collection = PatchCollection(patches_list, match_original=True)
        ax.add_collection(collection)
        
        # Add annotations if requested
        if config['show_labels'] or config['show_values']:
            self._add_annotations(heat_data, config)
        
        # Add contact area indicators for conductive heat
        if config['show_contact_areas'] or config['mode'] == 'external_conduction':
            self._add_contact_indicators(heat_data, config)
        
        # Configure axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')  # Hide axes for clean look
        
        # Add title
        if config['title']:
            ax.set_title(config['title'], fontsize=14, fontweight='bold', pad=20)
        
        # Add colorbar
        if config['show_colorbar']:
            self._add_colorbar(fig, config)
        
        plt.tight_layout()
        
        logger.info(f"Rendered 2D heat map with {len(heat_data)} segments in {config['mode']} mode")
        
        return fig
    
    def _validate_heat_data(self, heat_data: Dict[str, float]) -> None:
        """Validate heat data input"""
        if not isinstance(heat_data, dict):
            raise VisualizationError("Heat data must be a dictionary")
        
        if not heat_data:
            raise VisualizationError("Heat data cannot be empty")
        
        # Check for valid segment names
        invalid_segments = set(heat_data.keys()) - set(BODY_SEGMENTS)
        if invalid_segments:
            logger.warning(f"Invalid segment names found: {invalid_segments}")
        
        # Check for valid numeric values
        for segment, value in heat_data.items():
            if not isinstance(value, (int, float, np.number)) and not np.isnan(value):
                logger.warning(f"Non-numeric value for segment {segment}: {value}")
    
    def _create_segment_layout(self, segment_colors: Dict[str, np.ndarray], 
                             config: Dict[str, Any]) -> List[patches.Patch]:
        """Create 2D layout of body segments with colors"""
        patches_list = []
        
        for segment in BODY_SEGMENTS:
            if segment not in self.SEGMENT_LAYOUT:
                logger.warning(f"No layout defined for segment: {segment}")
                continue
            
            x, y, width, height = self.SEGMENT_LAYOUT[segment]
            color = segment_colors.get(segment, [0.5, 0.5, 0.5, 1.0])  # Default gray
            
            # Create patch based on segment type
            patch = self._create_segment_patch(segment, x, y, width, height, color, config)
            if patch:
                patches_list.append(patch)
        
        return patches_list
    
    def _create_segment_patch(self, segment: str, x: float, y: float, 
                            width: float, height: float, color: np.ndarray,
                            config: Dict[str, Any]) -> Optional[patches.Patch]:
        """Create a patch for a specific body segment"""
        
        # Choose patch type based on segment
        if segment == 'Head':
            # Use circle for head
            center_x = x + width/2
            center_y = y + height/2
            radius = min(width, height) / 2
            patch = Circle((center_x, center_y), radius, 
                          facecolor=color, edgecolor='black', linewidth=1)
            
        elif segment in ['LHand', 'RHand', 'LFoot', 'RFoot']:
            # Use ellipse for hands and feet
            center_x = x + width/2
            center_y = y + height/2
            patch = Ellipse((center_x, center_y), width, height,
                           facecolor=color, edgecolor='black', linewidth=1)
            
        elif segment in ['LArm', 'RArm', 'LLeg', 'RLeg']:
            # Use rounded rectangle for limbs
            patch = FancyBboxPatch((x, y), width, height,
                                  boxstyle="round,pad=0.01",
                                  facecolor=color, edgecolor='black', linewidth=1)
        else:
            # Use rectangle for torso segments
            patch = Rectangle((x, y), width, height,
                            facecolor=color, edgecolor='black', linewidth=1)
        
        return patch
    
    def _add_annotations(self, heat_data: Dict[str, float], 
                        config: Dict[str, Any]) -> None:
        """Add labels and values to segments"""
        for segment in BODY_SEGMENTS:
            if segment not in self.SEGMENT_LAYOUT:
                continue
            
            x, y, width, height = self.SEGMENT_LAYOUT[segment]
            center_x = x + width/2
            center_y = y + height/2
            
            # Add segment label
            if config['show_labels']:
                # Use shorter names for display
                display_name = self._get_display_name(segment)
                self.current_axes.text(center_x, center_y + height*0.15, display_name,
                                     ha='center', va='center', fontsize=8, fontweight='bold')
            
            # Add value
            if config['show_values'] and segment in heat_data:
                value = heat_data[segment]
                if not np.isnan(value):
                    value_text = f"{value:.1f}{config['units']}"
                    self.current_axes.text(center_x, center_y - height*0.15, value_text,
                                         ha='center', va='center', fontsize=7)
    
    def _get_display_name(self, segment: str) -> str:
        """Get shorter display name for segment"""
        name_mapping = {
            'LShoulder': 'L.Shldr', 'RShoulder': 'R.Shldr',
            'LArm': 'L.Arm', 'RArm': 'R.Arm',
            'LHand': 'L.Hand', 'RHand': 'R.Hand',
            'LThigh': 'L.Thigh', 'RThigh': 'R.Thigh',
            'LLeg': 'L.Leg', 'RLeg': 'R.Leg',
            'LFoot': 'L.Foot', 'RFoot': 'R.Foot'
        }
        return name_mapping.get(segment, segment)
    
    def _add_contact_indicators(self, heat_data: Dict[str, float], 
                              config: Dict[str, Any]) -> None:
        """Add visual indicators for segments with contact/conductive heat transfer"""
        for segment, value in heat_data.items():
            # Skip if no significant conductive heat transfer
            if abs(value) < 0.1:
                continue
            
            if segment not in self.SEGMENT_LAYOUT:
                continue
            
            x, y, width, height = self.SEGMENT_LAYOUT[segment]
            
            # Choose indicator style based on heating vs cooling  
            if value > 0:  # Heating
                outline_color = 'orange'
                line_style = '-'
            else:  # Cooling
                outline_color = 'cyan' 
                line_style = '--'
            
            # Add contact area outline
            contact_patch = Rectangle((x-0.01, y-0.01), width+0.02, height+0.02,
                                    facecolor='none', edgecolor=outline_color,
                                    linewidth=3, linestyle=line_style, alpha=0.8)
            self.current_axes.add_patch(contact_patch)
    
    def _add_colorbar(self, fig: plt.Figure, config: Dict[str, Any]) -> None:
        """Add colorbar to figure"""
        # Create colorbar axes
        cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
        
        # Create colorbar
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=self.color_mapper.normalizer, 
                                                  cmap=self.color_mapper.colormap),
                           cax=cbar_ax)
        
        # Set colorbar label
        if config['mode'] == 'temperature':
            cbar.set_label(f"Temperature ({config['units']})", rotation=270, labelpad=15)
        elif config['mode'] == 'external_conduction':
            cbar.set_label("Conductive Heat (W)", rotation=270, labelpad=15)
        else:
            cbar.set_label(f"Heat Transfer ({config['units']})", rotation=270, labelpad=15)
    
    def create_segment_layout(self, custom_positions: Optional[Dict[str, Tuple[float, float, float, float]]] = None) -> Dict[str, Tuple[float, float, float, float]]:
        """
        Create or customize segment layout positions
        
        Args:
            custom_positions: Override default positions for specific segments
            
        Returns:
            Dictionary of segment positions (x, y, width, height)
        """
        layout = self.SEGMENT_LAYOUT.copy()
        
        if custom_positions:
            layout.update(custom_positions)
        
        return layout
    
    def apply_heat_visualization_modes(self, heat_data: Dict[str, float], 
                                     mode: str) -> Dict[str, float]:
        """
        Apply mode-specific processing to heat data
        
        Args:
            heat_data: Raw heat transfer data
            mode: Visualization mode ('temperature', 'sensible', 'latent', 'external_conduction')
            
        Returns:
            Processed heat data for visualization
        """
        if mode == 'temperature':
            # No processing needed for temperature data
            return heat_data
        
        elif mode == 'external_conduction':
            # Filter for only segments with conductive heat transfer
            return {k: v for k, v in heat_data.items() if abs(v) > 0.01}
        
        elif mode == 'sensible':
            # Ensure positive values for heat loss visualization
            return {k: abs(v) if v < 0 else v for k, v in heat_data.items()}
        
        elif mode == 'latent':
            # Similar processing for latent heat
            return {k: abs(v) if v < 0 else v for k, v in heat_data.items()}
        
        else:
            logger.warning(f"Unknown visualization mode: {mode}")
            return heat_data
    
    def render_conductive_overlay(self, conductive_data: Dict[str, Dict[str, float]], 
                                base_figure: Optional[plt.Figure] = None) -> plt.Figure:
        """
        NEW: Render conductive heat transfer as overlay on existing figure
        
        Args:
            conductive_data: Dictionary with conductive heat parameters per segment
            base_figure: Existing figure to overlay on, or None to create new
            
        Returns:
            Figure with conductive heat overlay
        """
        if base_figure is None:
            fig, ax = plt.subplots(figsize=self.figure_size)
        else:
            fig = base_figure
            ax = fig.gca()
        
        # Extract heat transfer values for overlay
        heat_values = {}
        contact_areas = {}
        
        for segment, data in conductive_data.items():
            heat_values[segment] = data.get('heat_transfer', 0.0)
            contact_areas[segment] = data.get('contact_area', 0.0)
        
        # Add arrows showing direction and magnitude of heat transfer
        for segment in BODY_SEGMENTS:
            if segment not in self.SEGMENT_LAYOUT or segment not in heat_values:
                continue
            
            heat_value = heat_values[segment]
            contact_area = contact_areas[segment]
            
            # Skip if no significant heat transfer or contact
            if abs(heat_value) < 0.1 or contact_area < 0.01:
                continue
            
            x, y, width, height = self.SEGMENT_LAYOUT[segment]
            center_x = x + width/2
            center_y = y + height/2
            
            # Arrow properties based on heat transfer
            arrow_length = min(0.05, abs(heat_value) * 0.01)  # Scale with heat magnitude
            
            if heat_value > 0:  # Heat into body (warming)
                arrow_color = 'red'
                # Arrow pointing into segment
                dx, dy = 0, arrow_length
            else:  # Heat out of body (cooling)
                arrow_color = 'blue'  
                # Arrow pointing out of segment
                dx, dy = 0, -arrow_length
            
            # Add arrow
            ax.arrow(center_x, center_y, dx, dy, 
                    head_width=0.02, head_length=0.01,
                    fc=arrow_color, ec=arrow_color, alpha=0.8, linewidth=2)
            
            # Add heat value annotation
            heat_text = f"{heat_value:.1f}W"
            ax.text(center_x + width*0.4, center_y, heat_text,
                   fontsize=8, color=arrow_color, fontweight='bold')
        
        logger.info(f"Added conductive heat overlay for {len([v for v in heat_values.values() if abs(v) > 0.1])} segments")
        
        return fig
    
    def get_segment_boundaries(self) -> Dict[str, Dict[str, float]]:
        """
        Get boundary information for all segments
        
        Returns:
            Dictionary with boundary coordinates for each segment
        """
        boundaries = {}
        
        for segment, (x, y, width, height) in self.SEGMENT_LAYOUT.items():
            boundaries[segment] = {
                'left': x,
                'right': x + width,
                'bottom': y,
                'top': y + height,
                'center_x': x + width/2,
                'center_y': y + height/2,
                'area': width * height
            }
        
        return boundaries