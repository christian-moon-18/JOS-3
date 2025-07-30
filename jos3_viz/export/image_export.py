"""
Image Export System for JOS3 Visualizations

Implementation of ImageExporter class
Source: Agile Plan Sprint 2, Epic 2.3, Task 2.3.1  
User Story: As a researcher, I need to export visualizations for reports and publications

Handles export of visualization figures in multiple formats.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as backend_pdf
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

from ..core.logger import get_logger
from ..core.exceptions import ExportError

logger = get_logger(__name__)


class ImageExporter:
    """
    Export system for visualization figures
    
    Implements: TDD Section 6.3 - Export System
    Fulfills: PRD Section 3.3.1 - Static Exports requirements
    """
    
    # Publication-ready styling presets
    PUBLICATION_STYLES = {
        'ieee': {
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 10,
            'axes.labelsize': 10,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'lines.linewidth': 1.5,
            'patch.linewidth': 0.5,
            'axes.linewidth': 0.8
        },
        'nature': {
            'font.family': 'serif',
            'font.serif': ['Arial'],
            'font.size': 8,
            'axes.labelsize': 8,
            'axes.titlesize': 9,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'legend.fontsize': 7,
            'figure.titlesize': 10,
            'lines.linewidth': 1.0,
            'patch.linewidth': 0.3,
            'axes.linewidth': 0.5
        },
        'presentation': {
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica'],
            'font.size': 14,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18,
            'lines.linewidth': 2.0,
            'patch.linewidth': 1.0,
            'axes.linewidth': 1.2
        }
    }
    
    def __init__(self, output_directory: Optional[Union[str, Path]] = None):
        """
        Initialize image exporter
        
        Args:
            output_directory: Default output directory for exports
        """
        self.output_dir = Path(output_directory) if output_directory else Path.cwd()
        self.ensure_output_directory()
        
        # Store original matplotlib settings to restore later
        self.original_rcparams = plt.rcParams.copy()
        
        logger.debug(f"Initialized image exporter with output directory: {self.output_dir}")
    
    def ensure_output_directory(self) -> None:
        """Ensure output directory exists"""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ExportError(f"Could not create output directory {self.output_dir}: {str(e)}")
    
    def export_png(self, figure: plt.Figure, filepath: Union[str, Path], 
                   dpi: int = 300, **kwargs) -> Path:
        """
        Export figure as PNG image
        
        Args:
            figure: Matplotlib figure to export
            filepath: Output file path (with or without .png extension)
            dpi: Resolution in dots per inch
            **kwargs: Additional arguments for plt.savefig()
            
        Returns:
            Path to exported file
        """
        filepath = self._ensure_extension(filepath, '.png')
        
        # Default PNG export settings
        png_kwargs = {
            'dpi': dpi,
            'bbox_inches': 'tight',
            'facecolor': 'white',
            'edgecolor': 'none',
            'transparent': False,
            'pad_inches': 0.1
        }
        png_kwargs.update(kwargs)
        
        try:
            figure.savefig(filepath, format='png', **png_kwargs)
            logger.info(f"Exported PNG: {filepath} ({dpi} DPI)")
            return Path(filepath)
            
        except Exception as e:
            raise ExportError(f"Failed to export PNG: {str(e)}")
    
    def export_svg(self, figure: plt.Figure, filepath: Union[str, Path], 
                   **kwargs) -> Path:
        """
        Export figure as SVG (vector format)
        
        Args:
            figure: Matplotlib figure to export
            filepath: Output file path (with or without .svg extension)
            **kwargs: Additional arguments for plt.savefig()
            
        Returns:
            Path to exported file
        """
        filepath = self._ensure_extension(filepath, '.svg')
        
        # Default SVG export settings
        svg_kwargs = {
            'bbox_inches': 'tight',
            'facecolor': 'white',
            'edgecolor': 'none',
            'pad_inches': 0.1
        }
        svg_kwargs.update(kwargs)
        
        try:
            figure.savefig(filepath, format='svg', **svg_kwargs)
            logger.info(f"Exported SVG: {filepath}")
            return Path(filepath)
            
        except Exception as e:
            raise ExportError(f"Failed to export SVG: {str(e)}")
    
    def export_pdf(self, figure: plt.Figure, filepath: Union[str, Path],
                   **kwargs) -> Path:
        """
        Export figure as PDF (publication ready)
        
        Args:
            figure: Matplotlib figure to export
            filepath: Output file path (with or without .pdf extension)
            **kwargs: Additional arguments for plt.savefig()
            
        Returns:
            Path to exported file
        """
        filepath = self._ensure_extension(filepath, '.pdf')
        
        # Default PDF export settings
        pdf_kwargs = {
            'bbox_inches': 'tight',
            'facecolor': 'white',
            'edgecolor': 'none',
            'pad_inches': 0.1,
            'metadata': {
                'Title': 'JOS3 Heat Transfer Visualization',
                'Author': 'JOS3 Visualization Tool',
                'Creator': 'JOS3-viz Python Package',
                'Producer': 'matplotlib'
            }
        }
        pdf_kwargs.update(kwargs)
        
        try:
            figure.savefig(filepath, format='pdf', **pdf_kwargs)
            logger.info(f"Exported PDF: {filepath}")
            return Path(filepath)
            
        except Exception as e:
            raise ExportError(f"Failed to export PDF: {str(e)}")
    
    def export_multiple_formats(self, figure: plt.Figure, base_filename: Union[str, Path],
                              formats: List[str] = ['png', 'svg', 'pdf'],
                              **kwargs) -> Dict[str, Path]:
        """
        Export figure in multiple formats
        
        Args:
            figure: Matplotlib figure to export
            base_filename: Base filename (without extension)
            formats: List of formats to export ('png', 'svg', 'pdf')
            **kwargs: Additional arguments passed to export functions
            
        Returns:
            Dictionary mapping format names to exported file paths
        """
        base_path = Path(base_filename)
        exported_files = {}
        
        for format_name in formats:
            try:
                if format_name.lower() == 'png':
                    filepath = self.export_png(figure, base_path, **kwargs)
                elif format_name.lower() == 'svg':
                    filepath = self.export_svg(figure, base_path, **kwargs)
                elif format_name.lower() == 'pdf':
                    filepath = self.export_pdf(figure, base_path, **kwargs)
                else:
                    logger.warning(f"Unknown format: {format_name}")
                    continue
                
                exported_files[format_name] = filepath
                
            except Exception as e:
                logger.error(f"Failed to export {format_name}: {str(e)}")
                continue
        
        logger.info(f"Exported {len(exported_files)} formats for {base_filename}")
        return exported_files
    
    def set_publication_styling(self, style: str = 'ieee') -> None:
        """
        Apply publication-ready styling
        
        Args:
            style: Style preset ('ieee', 'nature', 'presentation')
        """
        if style not in self.PUBLICATION_STYLES:
            available_styles = list(self.PUBLICATION_STYLES.keys())
            raise ExportError(f"Unknown style '{style}'. Available: {available_styles}")
        
        # Apply style settings
        style_params = self.PUBLICATION_STYLES[style]
        plt.rcParams.update(style_params)
        
        logger.info(f"Applied publication style: {style}")
    
    def restore_default_styling(self) -> None:
        """Restore original matplotlib styling"""
        plt.rcParams.update(self.original_rcparams)
        logger.debug("Restored default matplotlib styling")
    
    def add_therapeutic_device_annotations(self, figure: plt.Figure, 
                                         device_info: Dict[str, Any]) -> plt.Figure:
        """
        NEW: Add annotations for therapeutic cooling/heating devices
        
        Args:
            figure: Figure to annotate
            device_info: Dictionary with device information
            
        Returns:
            Annotated figure
        """
        ax = figure.gca()
        
        # Add device legend
        device_elements = []
        device_labels = []
        
        if 'cooling_devices' in device_info:
            for device in device_info['cooling_devices']:
                # Add cooling device indicator
                device_elements.append(plt.Line2D([0], [0], color='cyan', lw=3, linestyle='--'))
                device_labels.append(f"Cooling: {device.get('name', 'Device')}")
        
        if 'heating_devices' in device_info:
            for device in device_info['heating_devices']:
                # Add heating device indicator  
                device_elements.append(plt.Line2D([0], [0], color='orange', lw=3, linestyle='-'))
                device_labels.append(f"Heating: {device.get('name', 'Device')}")
        
        if device_elements:
            # Add legend for devices
            device_legend = ax.legend(device_elements, device_labels, 
                                    loc='upper left', bbox_to_anchor=(0.02, 0.98),
                                    frameon=True, fancybox=True, shadow=True)
            device_legend.set_title("Therapeutic Devices", prop={'weight': 'bold'})
            
            # Adjust main legend position if it exists
            main_legend = ax.get_legend()
            if main_legend:
                main_legend.set_bbox_to_anchor((0.02, 0.85))
        
        # Add device specifications text box
        if 'specifications' in device_info:
            specs_text = self._format_device_specifications(device_info['specifications'])
            ax.text(0.98, 0.02, specs_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        logger.debug("Added therapeutic device annotations")
        return figure
    
    def _format_device_specifications(self, specs: Dict[str, Any]) -> str:
        """Format device specifications for display"""
        lines = ["Device Specifications:"]
        
        for key, value in specs.items():
            if key == 'temperature':
                lines.append(f"Temperature: {value}°C")
            elif key == 'power':
                lines.append(f"Power: {value}W")
            elif key == 'contact_area':
                lines.append(f"Contact Area: {value*100:.1f}%")
            elif key == 'duration':
                lines.append(f"Duration: {value}min")
            else:
                lines.append(f"{key.title()}: {value}")
        
        return '\n'.join(lines)
    
    def _ensure_extension(self, filepath: Union[str, Path], extension: str) -> Path:
        """Ensure filepath has the correct extension"""
        filepath = Path(filepath)
        
        # Add extension if missing
        if not filepath.suffix:
            filepath = filepath.with_suffix(extension)
        
        # Make path absolute relative to output directory
        if not filepath.is_absolute():
            filepath = self.output_dir / filepath
        
        return filepath
    
    def export_batch(self, figures: Dict[str, plt.Figure], 
                    base_directory: Optional[Union[str, Path]] = None,
                    formats: List[str] = ['png', 'pdf'],
                    **kwargs) -> Dict[str, Dict[str, Path]]:
        """
        Export multiple figures in batch
        
        Args:
            figures: Dictionary mapping names to figures
            base_directory: Base directory for exports (uses default if None)
            formats: List of formats to export
            **kwargs: Additional arguments for export functions
            
        Returns:
            Dictionary mapping figure names to exported file paths by format
        """
        if base_directory:
            original_output_dir = self.output_dir
            self.output_dir = Path(base_directory)
            self.ensure_output_directory()
        
        try:
            exported_files = {}
            
            for name, figure in figures.items():
                try:
                    figure_exports = self.export_multiple_formats(
                        figure, name, formats, **kwargs
                    )
                    exported_files[name] = figure_exports
                    
                except Exception as e:
                    logger.error(f"Failed to export figure '{name}': {str(e)}")
                    continue
            
            logger.info(f"Batch exported {len(exported_files)} figures")
            return exported_files
            
        finally:
            # Restore original output directory
            if base_directory:
                self.output_dir = original_output_dir
    
    def create_publication_figure(self, width_inches: float, height_inches: float,
                                style: str = 'ieee') -> plt.Figure:
        """
        Create a new figure with publication-ready settings
        
        Args:
            width_inches: Figure width in inches
            height_inches: Figure height in inches
            style: Publication style to apply
            
        Returns:
            Configured matplotlib figure
        """
        # Apply publication styling
        self.set_publication_styling(style)
        
        # Create figure with specified dimensions
        fig = plt.figure(figsize=(width_inches, height_inches))
        
        # Set tight layout
        fig.set_tight_layout(True)
        
        logger.debug(f"Created publication figure: {width_inches}x{height_inches} inches, style={style}")
        
        return fig
    
    def get_export_info(self) -> Dict[str, Any]:
        """
        Get information about export settings
        
        Returns:
            Dictionary with export configuration details
        """
        return {
            'output_directory': str(self.output_dir),
            'supported_formats': ['png', 'svg', 'pdf'],
            'publication_styles': list(self.PUBLICATION_STYLES.keys()),
            'current_rcparams': {k: v for k, v in plt.rcParams.items() 
                               if k.startswith(('font', 'axes', 'figure'))},
            'default_png_dpi': 300
        }


def create_filename_pattern(base_name: str, time_point: Optional[Union[int, float]] = None,
                          mode: Optional[str] = None, suffix: str = '') -> str:
    """
    Create standardized filename patterns for exports
    
    Args:
        base_name: Base filename
        time_point: Time point for time-series data
        mode: Visualization mode
        suffix: Additional suffix
        
    Returns:
        Formatted filename
    """
    parts = [base_name]
    
    if mode:
        parts.append(mode)
    
    if time_point is not None:
        if isinstance(time_point, int):
            parts.append(f"t{time_point:03d}")
        else:
            parts.append(f"t{time_point:.1f}".replace('.', 'p'))
    
    if suffix:
        parts.append(suffix)
    
    return '_'.join(parts)


def get_journal_requirements() -> Dict[str, Dict[str, Any]]:
    """
    Get common journal figure requirements
    
    Returns:
        Dictionary with journal-specific requirements
    """
    return {
        'nature': {
            'max_width_inches': 7.24,  # Full page width
            'max_height_inches': 9.72, # Full page height
            'min_font_size': 8,
            'preferred_formats': ['pdf', 'eps'],
            'dpi_requirement': 300
        },
        'science': {
            'max_width_inches': 7.0,
            'max_height_inches': 9.0,
            'min_font_size': 8,
            'preferred_formats': ['pdf', 'eps'],
            'dpi_requirement': 300
        },
        'ieee': {
            'max_width_inches': 8.5,
            'max_height_inches': 11.0,
            'min_font_size': 10,
            'preferred_formats': ['pdf', 'eps'],
            'dpi_requirement': 300
        }
    }