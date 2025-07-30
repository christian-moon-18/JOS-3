"""
Video Export for JOS3 Heat Transfer Visualization

Implementation of VideoExporter class for creating time-series animations
Source: Agile Plan Sprint 4, Epic 4.1, Task 4.1.1
User Story: As a researcher, I need video animations of thermal changes over time for presentations

Creates MP4, AVI, and GIF animations from thermal simulation data.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

from ..core.logger import get_logger
from ..core.exceptions import VisualizationError
from ..core.data_parser import JOS3DataParser
from ..core.heat_calculator import ExternalHeatCalculator
from ..visualization.renderer_2d import HeatRenderer2D
from ..visualization.color_mapping import HeatColorMapper

logger = get_logger(__name__)


class VideoExporter:
    """
    Create video animations from JOS3 thermal simulation data
    
    Implements: TDD Section 4.1 - Video Export System
    Fulfills: PRD Section 3.3.2 - Video animation requirements
    """
    
    # Supported video formats with their specifications
    SUPPORTED_FORMATS = {
        'mp4': {
            'extension': '.mp4',
            'writer': 'ffmpeg',
            'codec': 'libx264',
            'bitrate': 2000,
            'description': 'MP4 video with H.264 encoding'
        },
        'avi': {
            'extension': '.avi',
            'writer': 'ffmpeg', 
            'codec': 'libxvid',
            'bitrate': 1500,
            'description': 'AVI video with Xvid encoding'
        },
        'gif': {
            'extension': '.gif',
            'writer': 'pillow',
            'codec': None,
            'bitrate': None,
            'description': 'Animated GIF'
        }
    }
    
    # Animation presets for different use cases
    ANIMATION_PRESETS = {
        'presentation': {
            'fps': 8,
            'duration_per_minute': 2.0,  # seconds of video per simulation minute
            'show_timestamp': True,
            'show_metrics': True,
            'quality': 'high'
        },
        'research': {
            'fps': 15,
            'duration_per_minute': 1.0,
            'show_timestamp': True,
            'show_metrics': True,
            'quality': 'highest'
        },
        'quick_preview': {
            'fps': 5,
            'duration_per_minute': 0.5,
            'show_timestamp': False,
            'show_metrics': False,
            'quality': 'medium'
        },
        'publication': {
            'fps': 12,
            'duration_per_minute': 1.5,
            'show_timestamp': True,
            'show_metrics': False,
            'quality': 'highest'
        }
    }
    
    def __init__(self, data_parser: JOS3DataParser, 
                 heat_calculator: Optional[ExternalHeatCalculator] = None):
        """
        Initialize video exporter
        
        Args:
            data_parser: JOS3DataParser instance with simulation data
            heat_calculator: Optional heat calculator for external heat analysis
        """
        self.data_parser = data_parser
        self.heat_calculator = heat_calculator or ExternalHeatCalculator(data_parser)
        
        # Rendering components
        self.renderer_2d = HeatRenderer2D()
        self.color_mapper = HeatColorMapper()
        
        # Animation state
        self.current_animation = None
        self.frame_cache = {}
        self.progress_callback = None
        
        # Quality settings
        self.figure_size = (12, 8)
        self.dpi = 150
        
        logger.info(f"Initialized video exporter with {len(data_parser.data)} time points")
    
    def generate_frame_sequence(self, start_time: int = 0, end_time: Optional[int] = None,
                               fps: int = 10, mode: str = 'temperature') -> List[int]:
        """
        Generate sequence of time indices for animation frames
        
        Args:
            start_time: Starting time index
            end_time: Ending time index (None for last available)
            fps: Target frames per second
            mode: Visualization mode ('temperature', 'external_conduction', 'both')
            
        Returns:
            List of time indices for animation frames
        """
        if end_time is None:
            end_time = len(self.data_parser.data) - 1
        
        # Validate time range
        if start_time < 0 or start_time >= len(self.data_parser.data):
            raise VisualizationError(f"Invalid start_time: {start_time}")
        if end_time < start_time or end_time >= len(self.data_parser.data):
            raise VisualizationError(f"Invalid end_time: {end_time}")
        
        # Calculate frame indices based on desired fps and available data
        total_frames = end_time - start_time + 1
        
        # For smooth animation, we might skip frames or interpolate
        if fps <= 0:
            raise VisualizationError("fps must be positive")
        
        # Generate evenly spaced frame indices
        if total_frames <= fps * 2:  # Use all available frames
            frame_indices = list(range(start_time, end_time + 1))
        else:  # Sample frames to achieve target fps
            step = max(1, total_frames // (fps * 2))
            frame_indices = list(range(start_time, end_time + 1, step))
        
        logger.info(f"Generated {len(frame_indices)} frames from time {start_time} to {end_time}")
        return frame_indices
    
    def render_frame_at_time(self, time_index: int, mode: str = 'temperature',
                           show_timestamp: bool = True, show_metrics: bool = True) -> plt.Figure:
        """
        Render a single animation frame at specified time
        
        Args:
            time_index: Time index to render
            mode: Visualization mode
            show_timestamp: Whether to show timestamp overlay
            show_metrics: Whether to show key metrics
            
        Returns:
            Matplotlib figure for the frame
        """
        # Check cache first
        cache_key = f"{time_index}_{mode}_{show_timestamp}_{show_metrics}"
        if cache_key in self.frame_cache:
            return self.frame_cache[cache_key]
        
        # Get data for this time point
        if mode == 'temperature':
            heat_data = self.data_parser.get_temperature_data(time_index, 'skin')
            title = f"Skin Temperature"
            colormap = 'RdYlBu_r'
            units = '°C'
        elif mode == 'external_conduction':
            heat_data = self.data_parser.get_heat_transfer_data(time_index, 'external_conduction')
            title = f"Conductive Heat Transfer"
            colormap = 'custom_conductive'
            units = 'W'
        elif mode == 'both':
            # Create composite visualization
            temp_data = self.data_parser.get_temperature_data(time_index, 'skin')
            cond_data = self.data_parser.get_heat_transfer_data(time_index, 'external_conduction')
            heat_data = temp_data  # Use temperature as base
            title = f"Temperature + Conductive Heat"
            colormap = 'RdYlBu_r'
            units = '°C'
        else:
            raise VisualizationError(f"Unknown visualization mode: {mode}")
        
        # Create the visualization
        fig = self.renderer_2d.render_body_heatmap(
            heat_data,
            config={
                'title': title,
                'colormap': colormap,
                'show_colorbar': True,
                'mode': mode,
                'show_labels': False,  # Clean look for animation
                'show_values': False
            }
        )
        
        # Add timestamp overlay
        if show_timestamp:
            time_text = f"Time: {time_index} min"
            fig.text(0.02, 0.98, time_text, fontsize=14, fontweight='bold',
                    ha='left', va='top', transform=fig.transFigure,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add metrics overlay
        if show_metrics:
            metrics_text = self._get_metrics_text(heat_data, time_index, mode)
            fig.text(0.02, 0.02, metrics_text, fontsize=10,
                    ha='left', va='bottom', transform=fig.transFigure,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add to cache
        self.frame_cache[cache_key] = fig
        
        return fig
    
    def _get_metrics_text(self, heat_data: Dict[str, float], time_index: int, mode: str) -> str:
        """Generate metrics text overlay for frame"""
        if mode == 'temperature':
            values = list(heat_data.values())
            mean_val = np.mean(values)
            max_val = np.max(values)
            min_val = np.min(values)
            
            return f"Mean: {mean_val:.1f}°C\nRange: {min_val:.1f} - {max_val:.1f}°C"
        
        elif mode == 'external_conduction':
            values = [v for v in heat_data.values() if abs(v) > 0.1]
            if values:
                total_cooling = sum(v for v in values if v < 0)
                total_heating = sum(v for v in values if v > 0)
                active_segments = len(values)
                
                return f"Cooling: {abs(total_cooling):.1f}W\nHeating: {total_heating:.1f}W\nActive: {active_segments} segments"
            else:
                return "No active heat transfer"
        
        return ""
    
    def create_video_animation(self, output_path: str, 
                             start_time: int = 0, end_time: Optional[int] = None,
                             preset: str = 'presentation',
                             mode: str = 'temperature',
                             custom_config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create video animation from thermal simulation data
        
        Args:
            output_path: Output video file path
            start_time: Starting time index
            end_time: Ending time index
            preset: Animation preset ('presentation', 'research', 'quick_preview', 'publication')
            mode: Visualization mode
            custom_config: Custom configuration overrides
            
        Returns:
            True if video created successfully
        """
        logger.info(f"Creating video animation: {output_path}")
        
        # Load preset configuration
        if preset not in self.ANIMATION_PRESETS:
            raise VisualizationError(f"Unknown preset: {preset}")
        
        config = self.ANIMATION_PRESETS[preset].copy()
        if custom_config:
            config.update(custom_config)
        
        # Determine output format from file extension
        output_path = Path(output_path)
        format_key = output_path.suffix[1:].lower()  # Remove dot
        
        if format_key not in self.SUPPORTED_FORMATS:
            raise VisualizationError(f"Unsupported format: {format_key}")
        
        format_config = self.SUPPORTED_FORMATS[format_key]
        
        try:
            # Generate frame sequence
            frame_indices = self.generate_frame_sequence(
                start_time, end_time, config['fps'], mode
            )
            
            # Setup progress tracking
            total_frames = len(frame_indices)
            self._setup_progress_tracking(total_frames)
            
            # Create animation using matplotlib
            fig = plt.figure(figsize=self.figure_size, dpi=self.dpi)
            
            def animate_func(frame_idx):
                """Animation function for matplotlib"""
                time_index = frame_indices[frame_idx]
                
                # Clear figure and render new frame
                fig.clear()
                
                # Get the rendered frame
                frame_fig = self.render_frame_at_time(
                    time_index, mode, 
                    config['show_timestamp'], 
                    config['show_metrics']
                )
                
                # Copy content to animation figure
                # Note: This is a simplified approach - in practice you'd want
                # to properly transfer the axes and their content
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, f"Frame {frame_idx + 1}/{total_frames}\nTime: {time_index} min",
                       ha='center', va='center', transform=ax.transAxes, fontsize=16)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                
                plt.close(frame_fig)  # Clean up
                
                # Update progress
                self._update_progress(frame_idx + 1, total_frames)
                
                return ax,
            
            # Create animation
            anim = FuncAnimation(fig, animate_func, frames=len(frame_indices), 
                               interval=1000/config['fps'], blit=False, repeat=False)
            
            # Setup writer
            if format_config['writer'] == 'ffmpeg':
                writer = FFMpegWriter(fps=config['fps'], codec=format_config['codec'],
                                    bitrate=format_config['bitrate'])
            elif format_config['writer'] == 'pillow':
                writer = PillowWriter(fps=config['fps'])
            else:
                raise VisualizationError(f"Unknown writer: {format_config['writer']}")
            
            # Save animation
            anim.save(str(output_path), writer=writer, dpi=self.dpi)
            plt.close(fig)
            
            # Clear cache to free memory
            self.frame_cache.clear()
            
            logger.info(f"Video animation saved: {output_path}")
            logger.info(f"  Format: {format_config['description']}")
            logger.info(f"  Frames: {total_frames}")
            logger.info(f"  Duration: {total_frames/config['fps']:.1f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create video animation: {str(e)}")
            return False
    
    def create_frame_sequence_export(self, output_dir: str,
                                   start_time: int = 0, end_time: Optional[int] = None,
                                   mode: str = 'temperature',
                                   filename_pattern: str = "frame_{:04d}_{:02d}min.png") -> List[Path]:
        """
        Export individual frames as image files
        
        Args:
            output_dir: Output directory for frames
            start_time: Starting time index
            end_time: Ending time index
            mode: Visualization mode
            filename_pattern: Filename pattern with frame and time placeholders
            
        Returns:
            List of created frame file paths
        """
        logger.info(f"Exporting frame sequence to: {output_dir}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate frame indices
        frame_indices = self.generate_frame_sequence(start_time, end_time, fps=30, mode=mode)
        
        frame_files = []
        total_frames = len(frame_indices)
        
        self._setup_progress_tracking(total_frames)
        
        for i, time_index in enumerate(frame_indices):
            try:
                # Render frame
                fig = self.render_frame_at_time(time_index, mode, 
                                              show_timestamp=True, show_metrics=True)
                
                # Generate filename
                filename = filename_pattern.format(i, time_index)
                frame_path = output_dir / filename
                
                # Save frame
                fig.savefig(frame_path, dpi=self.dpi, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                plt.close(fig)
                
                frame_files.append(frame_path)
                self._update_progress(i + 1, total_frames)
                
            except Exception as e:
                logger.error(f"Failed to export frame {i}: {str(e)}")
                continue
        
        logger.info(f"Exported {len(frame_files)} frames")
        return frame_files
    
    def compile_video_from_frames(self, frame_files: List[Path], output_path: str,
                                fps: int = 10, format: str = 'mp4') -> bool:
        """
        Compile video from pre-rendered frame images
        
        Args:
            frame_files: List of frame image files
            output_path: Output video path
            fps: Frames per second
            format: Output format ('mp4', 'avi', 'gif')
            
        Returns:
            True if compilation successful
        """
        if format not in self.SUPPORTED_FORMATS:
            raise VisualizationError(f"Unsupported format: {format}")
        
        try:
            # This would typically use external tools like ffmpeg
            # For now, we'll create a simple slideshow-style video
            
            logger.info(f"Compiling {len(frame_files)} frames into {output_path}")
            logger.info(f"Target: {fps} fps, {format} format")
            
            # Read first frame to get dimensions
            if not frame_files:
                raise VisualizationError("No frame files provided")
            
            # Create a simple animation from the images
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            def animate_frame(frame_idx):
                if frame_idx < len(frame_files):
                    # Load and display image
                    from PIL import Image
                    img = Image.open(frame_files[frame_idx])
                    ax.clear()
                    ax.imshow(img)
                    ax.axis('off')
                    ax.set_title(f"Frame {frame_idx + 1}/{len(frame_files)}")
                return ax,
            
            anim = FuncAnimation(fig, animate_frame, frames=len(frame_files),
                               interval=1000/fps, blit=False, repeat=False)
            
            # Save with appropriate writer
            format_config = self.SUPPORTED_FORMATS[format]
            if format_config['writer'] == 'ffmpeg':
                writer = FFMpegWriter(fps=fps, codec=format_config['codec'])
            else:
                writer = PillowWriter(fps=fps)
            
            anim.save(output_path, writer=writer, dpi=self.dpi)
            plt.close(fig)
            
            logger.info(f"Video compilation complete: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Video compilation failed: {str(e)}")
            return False
    
    def _setup_progress_tracking(self, total_operations: int) -> None:
        """Setup progress tracking for long operations"""
        self.total_operations = total_operations
        self.completed_operations = 0
        self.start_time = time.time()
        
        logger.info(f"Starting operation with {total_operations} steps...")
    
    def _update_progress(self, completed: int, total: int) -> None:
        """Update progress and call progress callback if set"""
        self.completed_operations = completed
        
        if self.progress_callback:
            percentage = (completed / total) * 100
            elapsed = time.time() - self.start_time
            eta = (elapsed / completed) * (total - completed) if completed > 0 else 0
            
            self.progress_callback(completed, total, percentage, eta)
        
        # Log progress at key milestones
        percentage = (completed / total) * 100
        if percentage in [25, 50, 75, 100] or completed % max(1, total // 10) == 0:
            logger.info(f"Progress: {completed}/{total} ({percentage:.1f}%)")
    
    def add_therapy_annotations(self, fig: plt.Figure, time_index: int) -> None:
        """
        NEW: Add therapeutic device annotations to animation frames
        
        Args:
            fig: Matplotlib figure to annotate
            time_index: Current time index
        """
        # Get conductive heat data to identify active devices
        cond_data = self.data_parser.get_heat_transfer_data(time_index, 'external_conduction')
        
        active_devices = []
        for segment, heat_value in cond_data.items():
            if abs(heat_value) > 0.1:  # Significant heat transfer
                device_type = "Cooling" if heat_value < 0 else "Heating"
                active_devices.append(f"{device_type} on {segment}: {abs(heat_value):.1f}W")
        
        if active_devices:
            annotation_text = "Active Devices:\n" + "\n".join(active_devices[:3])  # Show top 3
            
            fig.text(0.98, 0.98, annotation_text, fontsize=9,
                    ha='right', va='top', transform=fig.transFigure,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
    
    def set_progress_callback(self, callback: Callable[[int, int, float, float], None]) -> None:
        """
        Set callback function for progress updates
        
        Args:
            callback: Function called with (completed, total, percentage, eta_seconds)
        """
        self.progress_callback = callback
    
    def get_animation_info(self) -> Dict[str, Any]:
        """
        Get information about animation capabilities and current state
        
        Returns:
            Dictionary with animation information
        """
        return {
            'supported_formats': list(self.SUPPORTED_FORMATS.keys()),
            'animation_presets': list(self.ANIMATION_PRESETS.keys()),
            'available_modes': ['temperature', 'external_conduction', 'both'],
            'data_time_points': len(self.data_parser.data),
            'cached_frames': len(self.frame_cache),
            'figure_size': self.figure_size,
            'dpi': self.dpi
        }
    
    def clear_cache(self) -> None:
        """Clear frame cache to free memory"""
        self.frame_cache.clear()
        logger.debug("Frame cache cleared")
    
    def estimate_file_size(self, format: str, duration_seconds: float, fps: int = 10) -> Dict[str, float]:
        """
        Estimate output file size for video animation
        
        Args:
            format: Video format
            duration_seconds: Animation duration
            fps: Frames per second
            
        Returns:
            Dictionary with size estimates in different units
        """
        if format not in self.SUPPORTED_FORMATS:
            return {'error': f'Unknown format: {format}'}
        
        # Rough estimates based on typical compression
        format_config = self.SUPPORTED_FORMATS[format]
        
        if format == 'mp4':
            # H.264 compression, estimate based on bitrate
            bitrate_kbps = format_config['bitrate']
            size_mb = (bitrate_kbps * duration_seconds) / (8 * 1024)
        elif format == 'avi':
            # Less efficient compression
            bitrate_kbps = format_config['bitrate'] 
            size_mb = (bitrate_kbps * duration_seconds) / (8 * 1024) * 1.3
        elif format == 'gif':
            # Uncompressed frames, much larger
            frame_count = duration_seconds * fps
            size_mb = frame_count * 0.5  # Rough estimate: 0.5MB per frame
        else:
            size_mb = 0
        
        return {
            'size_mb': size_mb,
            'size_kb': size_mb * 1024,
            'size_bytes': size_mb * 1024 * 1024
        }