"""
3D Heat Transfer Renderer using VTK

Implementation of HeatRenderer3D class
Source: Agile Plan Sprint 3, Epic 3.2, Task 3.2.1
User Story: As a researcher, I need interactive 3D visualization of heat transfer including conductive cooling/heating

Creates interactive 3D visualizations with VTK rendering pipeline.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import warnings

try:
    import vtk
    from vtk.util import numpy_support
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False
    warnings.warn("VTK not available. 3D visualization will not work.")

from ..core.logger import get_logger
from ..core.exceptions import VisualizationError
from ..models import BODY_SEGMENTS
from ..models.mannequin import MannequinGenerator
from .color_mapping import HeatColorMapper

logger = get_logger(__name__)


class HeatRenderer3D:
    """
    3D heat transfer visualization renderer using VTK
    
    Implements: TDD Section 6.4 - VTK 3D Rendering Engine
    Fulfills: PRD Section 3.1.1 - Interactive 3D visualization requirements
    """
    
    # Predefined camera positions for different views
    CAMERA_PRESETS = {
        'front': {
            'position': (0, -2, 1),
            'focal_point': (0, 0, 1),
            'view_up': (0, 0, 1)
        },
        'back': {
            'position': (0, 2, 1),
            'focal_point': (0, 0, 1),
            'view_up': (0, 0, 1)
        },
        'left': {
            'position': (-2, 0, 1),
            'focal_point': (0, 0, 1),
            'view_up': (0, 0, 1)
        },
        'right': {
            'position': (2, 0, 1),
            'focal_point': (0, 0, 1),
            'view_up': (0, 0, 1)
        },
        'isometric': {
            'position': (1.5, -1.5, 1.5),
            'focal_point': (0, 0, 1),
            'view_up': (0, 0, 1)
        },
        'top': {
            'position': (0, 0, 3),
            'focal_point': (0, 0, 1),
            'view_up': (0, 1, 0)
        }
    }
    
    # Lighting configurations
    LIGHTING_PRESETS = {
        'standard': [
            {'position': (1, -1, 1), 'intensity': 0.8, 'color': (1, 1, 1)},
            {'position': (-1, -1, 0.5), 'intensity': 0.4, 'color': (1, 1, 1)},
            {'position': (0, 1, 1), 'intensity': 0.3, 'color': (1, 1, 1)}
        ],
        'medical': [
            {'position': (0, -1, 1), 'intensity': 0.9, 'color': (1, 1, 0.9)},
            {'position': (1, 0, 0.5), 'intensity': 0.3, 'color': (0.9, 0.9, 1)},
            {'position': (-1, 0, 0.5), 'intensity': 0.3, 'color': (0.9, 0.9, 1)}
        ],
        'presentation': [
            {'position': (0, -1, 1), 'intensity': 1.0, 'color': (1, 1, 1)},
            {'position': (0.5, 0.5, 1), 'intensity': 0.5, 'color': (1, 1, 1)}
        ]
    }
    
    def __init__(self, window_size: Tuple[int, int] = (800, 600), 
                 background_color: Tuple[float, float, float] = (0.1, 0.1, 0.1)):
        """
        Initialize 3D heat renderer
        
        Args:
            window_size: Rendering window size (width, height)
            background_color: Background color (R, G, B) in range 0-1
        """
        if not VTK_AVAILABLE:
            raise VisualizationError("VTK is required for 3D visualization")
        
        self.window_size = window_size
        self.background_color = background_color
        
        # VTK pipeline components
        self.renderer = None
        self.render_window = None
        self.interactor = None
        self.mannequin_generator = None
        self.color_mapper = HeatColorMapper()
        
        # Data storage
        self.current_heat_data = {}
        self.segment_actors = {}
        self.contact_actors = {}
        self.heat_data_arrays = {}
        
        # Interaction and animation
        self.interaction_enabled = True
        self.animation_timer = None
        self.time_series_data = None
        self.current_time_index = 0
        
        logger.info(f"Initialized 3D heat renderer ({window_size[0]}x{window_size[1]})")
    
    def setup_vtk_pipeline(self, anthropometry: Optional[Dict[str, float]] = None) -> None:
        """
        Setup the VTK rendering pipeline
        
        Args:
            anthropometry: Subject anthropometric data
        """
        logger.info("Setting up VTK rendering pipeline...")
        
        # Create renderer
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(*self.background_color)
        
        # Create render window
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(*self.window_size)
        self.render_window.SetWindowName("JOS3 3D Heat Transfer Visualization")
        
        # Create mannequin generator
        self.mannequin_generator = MannequinGenerator(anthropometry)
        
        # Generate 3D body segments
        self.segment_actors = self.mannequin_generator.generate_body_segments(
            include_contact_patches=True
        )
        
        # Add segment actors to renderer
        for segment_name, actor in self.segment_actors.items():
            self.renderer.AddActor(actor)
        
        # Setup camera and lighting
        self.configure_lighting_and_camera()
        
        # Create interactor for user interaction
        if self.interaction_enabled:
            self.interactor = vtk.vtkRenderWindowInteractor()
            self.interactor.SetRenderWindow(self.render_window)
            
            # Setup interaction style
            style = vtk.vtkInteractorStyleTrackballCamera()
            self.interactor.SetInteractorStyle(style)
        
        logger.info("VTK pipeline setup complete")
    
    def render_heat_on_model(self, heat_data: Dict[str, float], 
                           mode: str = 'temperature',
                           colormap: str = 'RdBu_r') -> None:
        """
        Apply heat transfer data to 3D model surfaces
        
        Args:
            heat_data: Dictionary mapping segment names to heat/temperature values
            mode: Visualization mode ('temperature', 'heat_flow', 'external_conduction', 'composite')
            colormap: Color scheme to use
        """
        if not self.segment_actors:
            raise VisualizationError("VTK pipeline not initialized. Call setup_vtk_pipeline() first.")
        
        logger.info(f"Rendering heat data in {mode} mode with {colormap} colormap")
        
        # Store current data
        self.current_heat_data = heat_data.copy()
        
        # Setup color mapper
        data_type = 'temperature' if mode == 'temperature' else 'heat_flow'
        if mode == 'external_conduction':
            self.color_mapper = HeatColorMapper('custom_conductive', 'heat_flow')
        else:
            self.color_mapper = HeatColorMapper(colormap, data_type)
        
        # Map data to colors
        if mode == 'external_conduction':
            segment_colors = self.color_mapper.map_conductive_heat(heat_data)
        else:
            segment_colors = self.color_mapper.map_temperature_to_color(heat_data)
        
        # Apply colors to segment actors
        for segment_name, actor in self.segment_actors.items():
            if segment_name in segment_colors:
                color = segment_colors[segment_name][:3]  # RGB only
                actor.GetProperty().SetColor(*color)
                
                # Add scalar data for more advanced visualization
                mapper = actor.GetMapper()
                polydata = mapper.GetInput()
                
                if polydata and segment_name in heat_data:
                    self._add_scalar_data_to_surface(polydata, heat_data[segment_name], segment_name)
            else:
                # Default color for missing data
                actor.GetProperty().SetColor(0.5, 0.5, 0.5)
        
        # Update contact visualizations if in conductive mode
        if mode == 'external_conduction':
            self._update_contact_visualizations(heat_data)
        
        # Refresh rendering
        if self.render_window:
            self.render_window.Render()
        
        logger.debug(f"Applied heat visualization to {len(segment_colors)} segments")
    
    def _add_scalar_data_to_surface(self, polydata: vtk.vtkPolyData, 
                                   value: float, segment_name: str) -> None:
        """Add scalar data to polydata for advanced visualization"""
        try:
            num_points = polydata.GetNumberOfPoints()
            
            # Create scalar array
            scalars = vtk.vtkFloatArray()
            scalars.SetName(f"{segment_name}_heat")
            scalars.SetNumberOfTuples(num_points)
            
            # Fill with uniform value (could be enhanced with gradients)
            for i in range(num_points):
                scalars.SetValue(i, value)
            
            # Add to polydata
            polydata.GetPointData().SetScalars(scalars)
            self.heat_data_arrays[segment_name] = scalars
            
        except Exception as e:
            logger.warning(f"Failed to add scalar data to {segment_name}: {str(e)}")
    
    def _update_contact_visualizations(self, heat_data: Dict[str, float]) -> None:
        """Update contact area visualizations for conductive heat transfer"""
        # Remove existing contact actors
        for actor in self.contact_actors.values():
            self.renderer.RemoveActor(actor)
        self.contact_actors.clear()
        
        # Create contact data structure
        contact_data = {}
        for segment_name, heat_value in heat_data.items():
            if abs(heat_value) > 0.1:  # Significant heat transfer
                contact_data[segment_name] = {
                    'heat_transfer': heat_value,
                    'contact_area': 0.8,  # Placeholder - would come from actual data
                    'material_temperature': 20.0 if heat_value < 0 else 40.0  # Placeholder
                }
        
        # Generate new contact indicators
        if contact_data and self.mannequin_generator:
            contact_indicators = self.mannequin_generator.add_contact_indicators(contact_data)
            
            # Add to renderer
            for name, actor in contact_indicators.items():
                self.renderer.AddActor(actor)
                self.contact_actors[name] = actor
    
    def configure_lighting_and_camera(self, lighting_preset: str = 'standard',
                                    camera_preset: str = 'isometric') -> None:
        """
        Configure lighting and camera for optimal 3D visualization
        
        Args:
            lighting_preset: Lighting configuration ('standard', 'medical', 'presentation')
            camera_preset: Camera position preset ('front', 'back', 'isometric', etc.)
        """
        if not self.renderer:
            raise VisualizationError("Renderer not initialized")
        
        # Remove existing lights
        lights = vtk.vtkLightCollection()
        lights = self.renderer.GetLights()
        lights.RemoveAllItems()
        
        # Add lights based on preset
        if lighting_preset in self.LIGHTING_PRESETS:
            for light_config in self.LIGHTING_PRESETS[lighting_preset]:
                light = vtk.vtkLight()
                light.SetPosition(*light_config['position'])
                light.SetIntensity(light_config['intensity'])
                light.SetColor(*light_config['color'])
                light.SetLightTypeToSceneLight()
                self.renderer.AddLight(light)
        
        # Configure camera
        camera = self.renderer.GetActiveCamera()
        if camera_preset in self.CAMERA_PRESETS:
            preset = self.CAMERA_PRESETS[camera_preset]
            camera.SetPosition(*preset['position'])
            camera.SetFocalPoint(*preset['focal_point'])
            camera.SetViewUp(*preset['view_up'])
        
        # Auto-adjust camera to show full model
        self.renderer.ResetCamera()
        
        logger.debug(f"Applied {lighting_preset} lighting and {camera_preset} camera view")
    
    def enable_interaction(self, zoom: bool = True, rotate: bool = True, pan: bool = True) -> None:
        """
        Enable/configure user interaction with 3D model
        
        Args:
            zoom: Allow zooming
            rotate: Allow rotation
            pan: Allow panning
        """
        if not self.interactor:
            return
        
        # Create custom interaction style
        style = vtk.vtkInteractorStyleTrackballCamera()
        
        # Disable specific interactions if requested
        if not zoom:
            style.SetMotionFactor(0.0)
        if not rotate:
            style.SetEventPosition(0, 0)  # This would need more complex implementation
        if not pan:
            pass  # Would need custom style implementation
        
        self.interactor.SetInteractorStyle(style)
        
        # Add keyboard shortcuts
        self.interactor.AddObserver("KeyPressEvent", self._handle_key_press)
        
        logger.debug(f"Interaction enabled - zoom:{zoom}, rotate:{rotate}, pan:{pan}")
    
    def _handle_key_press(self, obj, event) -> None:
        """Handle keyboard shortcuts for 3D interaction"""
        if not self.interactor:
            return
        
        key = self.interactor.GetKeySym()
        
        # Camera presets
        if key == '1':
            self.configure_lighting_and_camera(camera_preset='front')
        elif key == '2':
            self.configure_lighting_and_camera(camera_preset='back')
        elif key == '3':
            self.configure_lighting_and_camera(camera_preset='left')
        elif key == '4':
            self.configure_lighting_and_camera(camera_preset='right')
        elif key == '5':
            self.configure_lighting_and_camera(camera_preset='isometric')
        elif key == '6':
            self.configure_lighting_and_camera(camera_preset='top')
        
        # Visualization modes
        elif key == 't':
            # Toggle transparency
            for actor in self.segment_actors.values():
                current_opacity = actor.GetProperty().GetOpacity()
                new_opacity = 0.7 if current_opacity == 1.0 else 1.0
                actor.GetProperty().SetOpacity(new_opacity)
        
        elif key == 'w':
            # Toggle wireframe
            for actor in self.segment_actors.values():
                prop = actor.GetProperty()
                if prop.GetRepresentation() == 2:  # Surface
                    prop.SetRepresentationToWireframe()
                else:
                    prop.SetRepresentationToSurface()
        
        elif key == 'r':
            # Reset camera
            self.renderer.ResetCamera()
        
        # Refresh display
        self.render_window.Render()
    
    def render_contact_devices(self, device_specifications: Dict[str, Dict[str, Any]]) -> None:
        """
        NEW: Render therapeutic cooling/heating devices on the 3D model
        
        Args:
            device_specifications: Dictionary with device information per segment
        """
        logger.info("Rendering therapeutic devices on 3D model...")
        
        for device_name, specs in device_specifications.items():
            segments = specs.get('segments', [])
            device_type = specs.get('type', 'cooling')  # 'cooling' or 'heating'
            
            for segment_name in segments:
                if segment_name not in self.segment_actors:
                    continue
                
                # Create device geometry
                device_actor = self._create_device_geometry(segment_name, specs)
                if device_actor:
                    self.renderer.AddActor(device_actor)
                    self.contact_actors[f"device_{device_name}_{segment_name}"] = device_actor
        
        # Refresh rendering
        if self.render_window:
            self.render_window.Render()
    
    def _create_device_geometry(self, segment_name: str, 
                               device_specs: Dict[str, Any]) -> Optional[vtk.vtkActor]:
        """Create 3D geometry for a therapeutic device"""
        if segment_name not in self.segment_actors:
            return None
        
        segment_actor = self.segment_actors[segment_name]
        
        # Get segment bounds for device sizing
        bounds = segment_actor.GetBounds()
        
        # Create device geometry (simplified as a shell around the segment)
        if 'vest' in device_specs.get('name', '').lower():
            # Cooling vest - covers torso
            source = vtk.vtkCubeSource()
            source.SetXLength(bounds[1] - bounds[0] + 0.02)  # Slightly larger
            source.SetYLength(bounds[3] - bounds[2] + 0.01)
            source.SetZLength(bounds[5] - bounds[4])
        else:
            # Generic device - sphere around segment
            source = vtk.vtkSphereSource()
            max_dim = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
            source.SetRadius(max_dim / 2 + 0.01)
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        
        device_actor = vtk.vtkActor()
        device_actor.SetMapper(mapper)
        
        # Position at segment center
        center = segment_actor.GetCenter()
        device_actor.SetPosition(*center)
        
        # Set device appearance
        property = device_actor.GetProperty()
        device_type = device_specs.get('type', 'cooling')
        
        if device_type == 'cooling':
            property.SetColor(0.0, 0.7, 1.0)  # Cyan
        else:
            property.SetColor(1.0, 0.5, 0.0)  # Orange
        
        property.SetOpacity(0.3)  # Semi-transparent
        property.SetSpecular(0.1)
        
        return device_actor
    
    def start_interactive_session(self) -> None:
        """Start interactive 3D visualization session"""
        if not self.render_window or not self.interactor:
            raise VisualizationError("VTK pipeline not properly initialized")
        
        logger.info("Starting interactive 3D visualization session...")
        logger.info("Keyboard shortcuts:")
        logger.info("  1-6: Camera presets (front, back, left, right, isometric, top)")
        logger.info("  't': Toggle transparency")
        logger.info("  'w': Toggle wireframe")
        logger.info("  'r': Reset camera")
        logger.info("  'q': Quit")
        
        # Start the interaction
        self.render_window.Render()
        self.interactor.Start()
    
    def save_screenshot(self, filepath: str, magnification: int = 1) -> bool:
        """
        Save screenshot of current 3D visualization
        
        Args:
            filepath: Output file path
            magnification: Image magnification factor
            
        Returns:
            True if save successful
        """
        if not self.render_window:
            return False
        
        try:
            # Create window to image filter
            window_to_image = vtk.vtkWindowToImageFilter()
            window_to_image.SetInput(self.render_window)
            window_to_image.SetMagnification(magnification)
            window_to_image.ReadFrontBufferOff()  # Read from back buffer
            window_to_image.Update()
            
            # Write image
            if filepath.lower().endswith('.png'):
                writer = vtk.vtkPNGWriter()
            elif filepath.lower().endswith('.jpg') or filepath.lower().endswith('.jpeg'):
                writer = vtk.vtkJPEGWriter()
            elif filepath.lower().endswith('.tiff') or filepath.lower().endswith('.tif'):
                writer = vtk.vtkTIFFWriter()
            else:
                writer = vtk.vtkPNGWriter()  # Default to PNG
                filepath += '.png'
            
            writer.SetFileName(filepath)
            writer.SetInputConnection(window_to_image.GetOutputPort())
            writer.Write()
            
            logger.info(f"Screenshot saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save screenshot: {str(e)}")
            return False
    
    def create_time_series_animation(self, time_series_data: List[Dict[str, float]],
                                   output_filepath: str, fps: int = 10) -> bool:
        """
        Create animation of heat transfer over time
        
        Args:
            time_series_data: List of heat data dictionaries for each time point
            output_filepath: Output video file path
            fps: Frames per second
            
        Returns:
            True if animation created successfully
        """
        logger.info(f"Creating time series animation with {len(time_series_data)} frames...")
        
        # This would require additional libraries like ffmpeg or imageio
        # For now, create individual frame images
        frame_files = []
        
        for i, heat_data in enumerate(time_series_data):
            # Update visualization
            self.render_heat_on_model(heat_data)
            
            # Save frame
            frame_file = output_filepath.replace('.mp4', f'_frame_{i:04d}.png')
            if self.save_screenshot(frame_file):
                frame_files.append(frame_file)
        
        logger.info(f"Created {len(frame_files)} animation frames")
        logger.info("Use ffmpeg or similar tool to combine frames into video:")
        logger.info(f"ffmpeg -r {fps} -i {output_filepath.replace('.mp4', '_frame_%04d.png')} {output_filepath}")
        
        return len(frame_files) > 0
    
    def render_composite_visualization(self, temperature_data: Dict[str, float],
                                     conductive_data: Dict[str, float],
                                     convective_data: Optional[Dict[str, float]] = None,
                                     layer_weights: Optional[Dict[str, float]] = None) -> None:
        """
        NEW: Create composite 3D visualization combining multiple heat transfer modes
        
        Args:
            temperature_data: Skin temperature data
            conductive_data: Conductive heat transfer data
            convective_data: Optional convective heat transfer data
            layer_weights: Weights for combining layers (temperature, conductive, convective)
        """
        if not self.segment_actors:
            raise VisualizationError("VTK pipeline not initialized")
        
        logger.info("Creating composite 3D visualization...")
        
        # Default layer weights
        if layer_weights is None:
            layer_weights = {
                'temperature': 0.6,
                'conductive': 0.3,
                'convective': 0.1 if convective_data else 0.0
            }
        
        # Clear existing contact actors
        for actor in self.contact_actors.values():
            self.renderer.RemoveActor(actor)
        self.contact_actors.clear()
        
        # Process each segment
        for segment_name, actor in self.segment_actors.items():
            # Base temperature visualization
            temp_value = temperature_data.get(segment_name, 37.0)
            temp_color = self._get_temperature_color(temp_value)
            
            # Conductive overlay
            cond_value = conductive_data.get(segment_name, 0.0)
            if abs(cond_value) > 0.1:
                # Apply conductive heat color mixing
                cond_color = self._get_conductive_color(cond_value)
                
                # Blend colors based on weights
                blended_color = self._blend_colors(
                    temp_color, cond_color,
                    layer_weights['temperature'], layer_weights['conductive']
                )
                
                # Create contact indicator for significant conductive transfer
                contact_indicator = self._create_composite_contact_indicator(
                    segment_name, cond_value, temp_value
                )
                if contact_indicator:
                    self.renderer.AddActor(contact_indicator)
                    self.contact_actors[f"composite_{segment_name}"] = contact_indicator
            else:
                blended_color = temp_color
            
            # Apply final color to segment
            actor.GetProperty().SetColor(*blended_color[:3])
            
            # Add convective effects if provided
            if convective_data and segment_name in convective_data:
                conv_value = convective_data[segment_name]
                if abs(conv_value) > 0.5:
                    # Modify opacity based on convective heat transfer
                    opacity = min(1.0, 0.7 + abs(conv_value) * 0.02)
                    actor.GetProperty().SetOpacity(opacity)
        
        # Add composite visualization legend/indicators
        self._add_composite_legend()
        
        # Refresh rendering
        if self.render_window:
            self.render_window.Render()
        
        logger.info("Composite 3D visualization complete")
    
    def _get_temperature_color(self, temperature: float) -> Tuple[float, float, float]:
        """Get color for temperature value"""
        # Normalize temperature to 0-1 range (30-45°C typical range)
        normalized = (temperature - 30.0) / 15.0
        normalized = max(0.0, min(1.0, normalized))
        
        # Blue to red gradient
        return (normalized, 0.2, 1.0 - normalized)
    
    def _get_conductive_color(self, heat_value: float) -> Tuple[float, float, float]:
        """Get color for conductive heat transfer"""
        if heat_value < 0:  # Cooling
            intensity = min(1.0, abs(heat_value) / 50.0)
            return (0.0, intensity, 1.0)  # Cyan
        else:  # Heating
            intensity = min(1.0, heat_value / 50.0)
            return (1.0, intensity * 0.5, 0.0)  # Orange
    
    def _blend_colors(self, color1: Tuple[float, float, float], 
                     color2: Tuple[float, float, float],
                     weight1: float, weight2: float) -> Tuple[float, float, float]:
        """Blend two colors with specified weights"""
        total_weight = weight1 + weight2
        if total_weight == 0:
            return color1
        
        w1 = weight1 / total_weight
        w2 = weight2 / total_weight
        
        return (
            color1[0] * w1 + color2[0] * w2,
            color1[1] * w1 + color2[1] * w2,
            color1[2] * w1 + color2[2] * w2
        )
    
    def _create_composite_contact_indicator(self, segment_name: str, 
                                          heat_value: float, temp_value: float) -> Optional[vtk.vtkActor]:
        """Create composite contact indicator showing both temperature and heat transfer"""
        if segment_name not in self.segment_actors:
            return None
        
        segment_actor = self.segment_actors[segment_name]
        
        # Create a wireframe outline
        outline = vtk.vtkOutlineFilter()
        mapper = segment_actor.GetMapper()
        
        # Apply transform
        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputConnection(mapper.GetInputConnection(0, 0))
        transform_filter.SetTransform(segment_actor.GetUserTransform())
        
        outline.SetInputConnection(transform_filter.GetOutputPort())
        
        # Create indicator actor
        outline_mapper = vtk.vtkPolyDataMapper()
        outline_mapper.SetInputConnection(outline.GetOutputPort())
        
        indicator_actor = vtk.vtkActor()
        indicator_actor.SetMapper(outline_mapper)
        
        # Set appearance based on dominant effect
        property = indicator_actor.GetProperty()
        if abs(heat_value) > 20:  # Strong conductive effect
            if heat_value < 0:
                property.SetColor(0.0, 1.0, 1.0)  # Bright cyan
            else:
                property.SetColor(1.0, 0.5, 0.0)  # Orange
            property.SetLineWidth(4.0)
        else:
            # Moderate effect - use temperature color
            temp_color = self._get_temperature_color(temp_value)
            property.SetColor(*temp_color)
            property.SetLineWidth(2.0)
        
        property.SetOpacity(0.7)
        
        return indicator_actor
    
    def _add_composite_legend(self) -> None:
        """Add visual legend for composite visualization"""
        # This could create text actors or color bars
        # For now, just log the legend information
        logger.debug("Composite visualization legend:")
        logger.debug("  Base colors: Blue (cold) to Red (hot) temperature")
        logger.debug("  Outlines: Cyan (cooling), Orange (heating)")
        logger.debug("  Line width: Proportional to heat transfer intensity")
    
    def get_render_info(self) -> Dict[str, Any]:
        """
        Get information about current 3D rendering state
        
        Returns:
            Dictionary with rendering information
        """
        info = {
            'vtk_available': VTK_AVAILABLE,
            'pipeline_initialized': self.renderer is not None,
            'num_segments': len(self.segment_actors),
            'num_contact_actors': len(self.contact_actors),
            'interaction_enabled': self.interaction_enabled,
            'window_size': self.window_size,
            'background_color': self.background_color
        }
        
        if self.renderer:
            info.update({
                'num_actors': self.renderer.GetActors().GetNumberOfItems(),
                'num_lights': self.renderer.GetLights().GetNumberOfItems(),
                'camera_position': self.renderer.GetActiveCamera().GetPosition(),
                'camera_focal_point': self.renderer.GetActiveCamera().GetFocalPoint()
            })
        
        if self.mannequin_generator:
            info['anthropometry'] = self.mannequin_generator.anthropometry
        
        return info
    
    def cleanup(self) -> None:
        """Clean up VTK resources"""
        if self.interactor:
            self.interactor.TerminateApp()
        
        # Clear actors
        if self.renderer:
            self.renderer.RemoveAllViewProps()
        
        # Clear references
        self.segment_actors.clear()
        self.contact_actors.clear()
        self.heat_data_arrays.clear()
        
        logger.debug("3D renderer cleanup complete")