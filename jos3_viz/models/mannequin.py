"""
3D Mannequin Generation for JOS3 Visualization

Implementation of MannequinGenerator class
Source: Agile Plan Sprint 3, Epic 3.1, Task 3.1.1
User Story: As a thermal engineer, I need accurate 3D human models for spatial heat analysis including contact areas

Creates 3D humanoid models with accurate anthropometric scaling.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

try:
    import vtk
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False
    warnings.warn("VTK not available. 3D functionality will be limited.")
    # Create dummy vtk module for type hints
    class _DummyVTK:
        class vtkActor: pass
        class vtkPolyData: pass
        class vtkSource: pass
    vtk = _DummyVTK()

from ..core.logger import get_logger
from ..core.exceptions import VisualizationError
from .body_segments import BODY_SEGMENTS, get_bilateral_segments, get_torso_segments
from .scaling import get_segment_dimensions, get_segment_mass

logger = get_logger(__name__)


class MannequinGenerator:
    """
    Generate 3D humanoid mannequin models for heat transfer visualization
    
    Implements: TDD Section 3.2 - 3D Mannequin Generation
    Fulfills: PRD Section 3.1.1 - Humanoid Model requirements
    """
    
    # Default anthropometric parameters (adult male)
    DEFAULT_ANTHROPOMETRY = {
        'height': 1.75,      # meters
        'weight': 70.0,      # kg
        'age': 30,           # years
        'body_fat': 15.0,    # percentage
        'sex': 'male'
    }
    
    # Segment positioning in 3D space (relative coordinates)
    SEGMENT_3D_POSITIONS = {
        # Head and neck
        'Head': {'center': (0.0, 0.0, 1.65), 'orientation': (0, 0, 0)},
        'Neck': {'center': (0.0, 0.0, 1.55), 'orientation': (0, 0, 0)},
        
        # Torso
        'Chest': {'center': (0.0, 0.0, 1.35), 'orientation': (0, 0, 0)},
        'Back': {'center': (0.0, 0.08, 1.35), 'orientation': (0, 0, 0)},  # Slightly behind chest
        'Pelvis': {'center': (0.0, 0.0, 1.0), 'orientation': (0, 0, 0)},
        
        # Arms - Left side
        'LShoulder': {'center': (-0.25, 0.0, 1.45), 'orientation': (0, 0, -15)},
        'LArm': {'center': (-0.35, 0.0, 1.2), 'orientation': (0, 0, -10)},
        'LHand': {'center': (-0.4, 0.0, 0.95), 'orientation': (0, 0, 0)},
        
        # Arms - Right side  
        'RShoulder': {'center': (0.25, 0.0, 1.45), 'orientation': (0, 0, 15)},
        'RArm': {'center': (0.35, 0.0, 1.2), 'orientation': (0, 0, 10)},
        'RHand': {'center': (0.4, 0.0, 0.95), 'orientation': (0, 0, 0)},
        
        # Legs - Left side
        'LThigh': {'center': (-0.12, 0.0, 0.75), 'orientation': (0, 0, 0)},
        'LLeg': {'center': (-0.12, 0.0, 0.45), 'orientation': (0, 0, 0)},
        'LFoot': {'center': (-0.12, -0.08, 0.1), 'orientation': (90, 0, 0)},
        
        # Legs - Right side
        'RThigh': {'center': (0.12, 0.0, 0.75), 'orientation': (0, 0, 0)},
        'RLeg': {'center': (0.12, 0.0, 0.45), 'orientation': (0, 0, 0)},
        'RFoot': {'center': (0.12, -0.08, 0.1), 'orientation': (90, 0, 0)}
    }
    
    # Contact patch specifications for therapeutic devices
    CONTACT_PATCH_SPECS = {
        'chest_cooling_vest': {
            'segments': ['Chest'],
            'coverage': 0.8,
            'thickness': 0.005,  # 5mm thickness
            'color': (0.0, 0.7, 1.0, 0.6)  # Semi-transparent cyan
        },
        'back_cooling_pad': {
            'segments': ['Back'],
            'coverage': 0.9,
            'thickness': 0.003,  # 3mm thickness
            'color': (0.0, 0.5, 1.0, 0.6)  # Semi-transparent blue
        },
        'arm_heating_sleeves': {
            'segments': ['LArm', 'RArm'],
            'coverage': 0.7,
            'thickness': 0.002,  # 2mm thickness
            'color': (1.0, 0.5, 0.0, 0.6)  # Semi-transparent orange
        }
    }
    
    def __init__(self, anthropometry: Optional[Dict[str, float]] = None):
        """
        Initialize mannequin generator
        
        Args:
            anthropometry: Dictionary with height, weight, age, body_fat, sex
        """
        if not VTK_AVAILABLE:
            raise VisualizationError("VTK is required for 3D mannequin generation")
        
        # Use provided anthropometry or defaults
        self.anthropometry = self.DEFAULT_ANTHROPOMETRY.copy()
        if anthropometry:
            self.anthropometry.update(anthropometry)
        
        # Scale factor based on height
        self.height_scale = self.anthropometry['height'] / self.DEFAULT_ANTHROPOMETRY['height']
        
        # Storage for generated segments
        self.segment_actors = {}
        self.contact_actors = {}
        self.segment_mappers = {}
        
        logger.info(f"Initialized mannequin generator for {self.anthropometry['height']:.2f}m, "
                   f"{self.anthropometry['weight']:.1f}kg subject")
    
    def generate_body_segments(self, include_contact_patches: bool = False) -> Dict[str, vtk.vtkActor]:
        """
        Generate 3D geometry for all body segments
        
        Args:
            include_contact_patches: Whether to include contact area visualizations
            
        Returns:
            Dictionary mapping segment names to VTK actors
        """
        logger.info("Generating 3D body segments...")
        
        self.segment_actors = {}
        
        for segment in BODY_SEGMENTS:
            try:
                actor = self.create_segment_geometry(segment)
                if actor:
                    self.segment_actors[segment] = actor
                    logger.debug(f"Created 3D geometry for {segment}")
                
            except Exception as e:
                logger.error(f"Failed to create geometry for {segment}: {str(e)}")
                continue
        
        # Add contact patches if requested
        if include_contact_patches:
            self._generate_contact_patches()
        
        logger.info(f"Generated {len(self.segment_actors)} body segment geometries")
        return self.segment_actors.copy()
    
    def create_segment_geometry(self, segment_name: str) -> Optional[vtk.vtkActor]:
        """
        Create 3D geometry for a specific body segment
        
        Args:
            segment_name: Name of body segment
            
        Returns:
            VTK actor for the segment
        """
        if segment_name not in BODY_SEGMENTS:
            raise VisualizationError(f"Unknown segment: {segment_name}")
        
        if segment_name not in self.SEGMENT_3D_POSITIONS:
            logger.warning(f"No 3D position defined for {segment_name}")
            return None
        
        # Get segment dimensions
        dimensions = get_segment_dimensions(
            segment_name, 
            self.anthropometry['height'], 
            self.anthropometry['weight']
        )
        
        # Create appropriate primitive based on segment
        if segment_name == 'Head':
            source = self._create_head_geometry(dimensions)
        elif segment_name == 'Neck':
            source = self._create_neck_geometry(dimensions)
        elif segment_name in ['Chest', 'Back', 'Pelvis']:
            source = self._create_torso_geometry(segment_name, dimensions)
        elif 'Arm' in segment_name or 'Leg' in segment_name:
            source = self._create_limb_geometry(segment_name, dimensions)
        elif 'Hand' in segment_name or 'Foot' in segment_name:
            source = self._create_extremity_geometry(segment_name, dimensions)
        elif 'Shoulder' in segment_name:
            source = self._create_shoulder_geometry(dimensions)
        else:
            # Default to box
            source = self._create_box_geometry(dimensions)
        
        if not source:
            return None
        
        # Create mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        self.segment_mappers[segment_name] = mapper
        
        # Create actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Position and orient the segment
        self.apply_anthropometric_scaling(actor, segment_name, dimensions)
        
        # Set default appearance
        self._set_default_appearance(actor, segment_name)
        
        return actor
    
    def _create_head_geometry(self, dimensions: Dict[str, float]) -> vtk.vtkSphereSource:
        """Create sphere for head"""
        source = vtk.vtkSphereSource()
        radius = min(dimensions['width'], dimensions['depth']) / 2
        source.SetRadius(radius)
        source.SetThetaResolution(20)
        source.SetPhiResolution(20)
        return source
    
    def _create_neck_geometry(self, dimensions: Dict[str, float]) -> vtk.vtkCylinderSource:
        """Create cylinder for neck"""
        source = vtk.vtkCylinderSource()
        source.SetRadius(dimensions['width'] / 2)
        source.SetHeight(dimensions['length'])
        source.SetResolution(12)
        return source
    
    def _create_torso_geometry(self, segment_name: str, dimensions: Dict[str, float]) -> vtk.vtkCubeSource:
        """Create box for torso segments with rounded edges"""
        source = vtk.vtkCubeSource()
        source.SetXLength(dimensions['width'])
        source.SetYLength(dimensions['depth'])
        source.SetZLength(dimensions['length'])
        return source
    
    def _create_limb_geometry(self, segment_name: str, dimensions: Dict[str, float]) -> vtk.vtkCylinderSource:
        """Create cylinder for arm and leg segments"""
        source = vtk.vtkCylinderSource()
        source.SetRadius(dimensions['width'] / 2)
        source.SetHeight(dimensions['length'])
        source.SetResolution(12)
        return source
    
    def _create_extremity_geometry(self, segment_name: str, dimensions: Dict[str, float]) -> vtk.vtkSource:
        """Create geometry for hands and feet"""
        if 'Hand' in segment_name:
            # Use ellipsoid for hands
            source = vtk.vtkSphereSource()
            source.SetRadius(dimensions['width'] / 2)
            source.SetThetaResolution(12)
            source.SetPhiResolution(8)
            return source
        else:
            # Use box for feet
            source = vtk.vtkCubeSource()
            source.SetXLength(dimensions['length'])  # Foot length
            source.SetYLength(dimensions['depth'])   # Foot width
            source.SetZLength(dimensions['width'])   # Foot height
            return source
    
    def _create_shoulder_geometry(self, dimensions: Dict[str, float]) -> vtk.vtkSphereSource:
        """Create sphere for shoulder joints"""
        source = vtk.vtkSphereSource()
        source.SetRadius(dimensions['width'] / 2)
        source.SetThetaResolution(12)
        source.SetPhiResolution(8)
        return source
    
    def _create_box_geometry(self, dimensions: Dict[str, float]) -> vtk.vtkCubeSource:
        """Create default box geometry"""
        source = vtk.vtkCubeSource()
        source.SetXLength(dimensions['width'])
        source.SetYLength(dimensions['depth'])
        source.SetZLength(dimensions['length'])
        return source
    
    def apply_anthropometric_scaling(self, actor: vtk.vtkActor, segment_name: str, 
                                   dimensions: Dict[str, float]) -> None:
        """
        Apply anthropometric scaling and positioning to segment
        
        Args:
            actor: VTK actor to transform
            segment_name: Name of body segment
            dimensions: Segment dimensions
        """
        if segment_name not in self.SEGMENT_3D_POSITIONS:
            return
        
        position_info = self.SEGMENT_3D_POSITIONS[segment_name]
        center = position_info['center']
        orientation = position_info['orientation']
        
        # Scale position by height
        scaled_center = (
            center[0] * self.height_scale,
            center[1] * self.height_scale,
            center[2] * self.height_scale
        )
        
        # Apply transformations
        transform = vtk.vtkTransform()
        
        # Scale the geometry itself
        weight_scale = (self.anthropometry['weight'] / self.DEFAULT_ANTHROPOMETRY['weight']) ** (1/3)
        transform.Scale(self.height_scale, weight_scale, self.height_scale)
        
        # Apply rotations
        if orientation[0] != 0:
            transform.RotateX(orientation[0])
        if orientation[1] != 0:
            transform.RotateY(orientation[1])
        if orientation[2] != 0:
            transform.RotateZ(orientation[2])
        
        # Position the segment
        transform.Translate(scaled_center[0], scaled_center[1], scaled_center[2])
        
        actor.SetUserTransform(transform)
    
    def _set_default_appearance(self, actor: vtk.vtkActor, segment_name: str) -> None:
        """Set default visual appearance for segment"""
        property = actor.GetProperty()
        
        # Default flesh color
        property.SetColor(0.9, 0.8, 0.7)  # Light flesh tone
        
        # Different colors for different segment types
        if segment_name == 'Head':
            property.SetColor(0.95, 0.85, 0.75)  # Slightly lighter for head
        elif 'Hand' in segment_name or 'Foot' in segment_name:
            property.SetColor(0.85, 0.75, 0.65)  # Slightly darker for extremities
        
        # Surface properties
        property.SetSpecular(0.3)
        property.SetSpecularPower(30)
        property.SetDiffuse(0.8)
        property.SetAmbient(0.2)
    
    def _generate_contact_patches(self) -> None:
        """Generate visualization elements for contact areas"""
        logger.debug("Generating contact patch visualizations...")
        
        self.contact_actors = {}
        
        for device_name, specs in self.CONTACT_PATCH_SPECS.items():
            for segment in specs['segments']:
                if segment not in self.segment_actors:
                    continue
                
                try:
                    contact_actor = self._create_contact_patch(segment, specs)
                    if contact_actor:
                        patch_name = f"{device_name}_{segment}"
                        self.contact_actors[patch_name] = contact_actor
                        logger.debug(f"Created contact patch: {patch_name}")
                        
                except Exception as e:
                    logger.error(f"Failed to create contact patch for {segment}: {str(e)}")
                    continue
    
    def _create_contact_patch(self, segment_name: str, specs: Dict[str, Any]) -> Optional[vtk.vtkActor]:
        """Create a contact patch for a specific segment"""
        if segment_name not in self.segment_actors:
            return None
        
        # Get the original segment geometry
        segment_actor = self.segment_actors[segment_name]
        
        # Create a slightly larger geometry for the contact patch
        dimensions = get_segment_dimensions(
            segment_name,
            self.anthropometry['height'],
            self.anthropometry['weight']
        )
        
        # Increase dimensions slightly for contact patch
        thickness = specs['thickness']
        patch_dimensions = {
            'width': dimensions['width'] + thickness,
            'depth': dimensions['depth'] + thickness,
            'length': dimensions['length']
        }
        
        # Create contact patch geometry
        if segment_name in ['Chest', 'Back', 'Pelvis']:
            source = vtk.vtkCubeSource()
            source.SetXLength(patch_dimensions['width'])
            source.SetYLength(patch_dimensions['depth'])
            source.SetZLength(patch_dimensions['length'])
        else:
            # Use cylinder for limbs
            source = vtk.vtkCylinderSource()
            source.SetRadius(patch_dimensions['width'] / 2)
            source.SetHeight(patch_dimensions['length'])
            source.SetResolution(12)
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Copy transform from original segment
        actor.SetUserTransform(segment_actor.GetUserTransform())
        
        # Set contact patch appearance
        property = actor.GetProperty()
        color = specs['color']
        property.SetColor(color[0], color[1], color[2])
        property.SetOpacity(color[3])
        property.SetSpecular(0.1)
        property.SetDiffuse(0.9)
        
        return actor
    
    def merge_segments_into_body(self, segments_to_merge: Optional[List[str]] = None) -> vtk.vtkActor:
        """
        Merge multiple segments into a single body actor
        
        Args:
            segments_to_merge: List of segment names to merge, or None for all
            
        Returns:
            Single VTK actor representing the merged body
        """
        if not self.segment_actors:
            raise VisualizationError("No segments have been generated yet")
        
        segments = segments_to_merge or list(self.segment_actors.keys())
        
        # Create append filter to merge geometries
        append_filter = vtk.vtkAppendPolyData()
        
        for segment_name in segments:
            if segment_name not in self.segment_actors:
                logger.warning(f"Segment {segment_name} not found, skipping")
                continue
            
            # Get the polydata from the segment
            actor = self.segment_actors[segment_name]
            mapper = actor.GetMapper()
            
            # Apply transform to the polydata
            transform_filter = vtk.vtkTransformPolyDataFilter()
            transform_filter.SetInputConnection(mapper.GetInputConnection(0, 0))
            transform_filter.SetTransform(actor.GetUserTransform())
            transform_filter.Update()
            
            append_filter.AddInputConnection(transform_filter.GetOutputPort())
        
        append_filter.Update()
        
        # Create mapper for merged geometry
        merged_mapper = vtk.vtkPolyDataMapper()
        merged_mapper.SetInputConnection(append_filter.GetOutputPort())
        
        # Create merged actor
        merged_actor = vtk.vtkActor()
        merged_actor.SetMapper(merged_mapper)
        
        # Set default appearance
        property = merged_actor.GetProperty()
        property.SetColor(0.9, 0.8, 0.7)
        property.SetSpecular(0.3)
        property.SetSpecularPower(30)
        
        logger.info(f"Merged {len(segments)} segments into single body actor")
        
        return merged_actor
    
    def add_contact_indicators(self, contact_data: Dict[str, Dict[str, float]]) -> Dict[str, vtk.vtkActor]:
        """
        NEW: Add visual indicators for segments with contact/conductive heat transfer
        
        Args:
            contact_data: Dictionary with contact information per segment
            
        Returns:
            Dictionary of contact indicator actors
        """
        indicators = {}
        
        for segment_name, data in contact_data.items():
            contact_area = data.get('contact_area', 0.0)
            material_temp = data.get('material_temperature', float('nan'))
            heat_transfer = data.get('heat_transfer', 0.0)
            
            # Skip if no significant contact
            if contact_area < 0.01 or np.isnan(material_temp):
                continue
            
            try:
                # Create contact indicator
                indicator = self._create_contact_indicator(segment_name, data)
                if indicator:
                    indicators[f"contact_{segment_name}"] = indicator
                    
            except Exception as e:
                logger.error(f"Failed to create contact indicator for {segment_name}: {str(e)}")
                continue
        
        logger.info(f"Created {len(indicators)} contact indicators")
        return indicators
    
    def _create_contact_indicator(self, segment_name: str, contact_data: Dict[str, float]) -> Optional[vtk.vtkActor]:
        """Create a visual indicator for contact area"""
        if segment_name not in self.segment_actors:
            return None
        
        heat_transfer = contact_data.get('heat_transfer', 0.0)
        contact_area = contact_data.get('contact_area', 0.0)
        
        # Create outline around the segment
        outline = vtk.vtkOutlineFilter()
        
        # Get the segment's polydata
        segment_actor = self.segment_actors[segment_name]
        mapper = segment_actor.GetMapper()
        
        # Apply transform
        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputConnection(mapper.GetInputConnection(0, 0))
        transform_filter.SetTransform(segment_actor.GetUserTransform())
        
        outline.SetInputConnection(transform_filter.GetOutputPort())
        
        # Create mapper and actor for outline
        outline_mapper = vtk.vtkPolyDataMapper()
        outline_mapper.SetInputConnection(outline.GetOutputPort())
        
        indicator_actor = vtk.vtkActor()
        indicator_actor.SetMapper(outline_mapper)
        
        # Set appearance based on heating vs cooling
        property = indicator_actor.GetProperty()
        if heat_transfer > 0:  # Heating
            property.SetColor(1.0, 0.5, 0.0)  # Orange
        else:  # Cooling
            property.SetColor(0.0, 0.7, 1.0)  # Cyan
        
        # Line width based on contact area
        property.SetLineWidth(max(2.0, contact_area * 10))
        property.SetOpacity(0.8)
        
        return indicator_actor
    
    def get_segment_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about generated segments
        
        Returns:
            Dictionary with segment information
        """
        info = {}
        
        for segment_name, actor in self.segment_actors.items():
            mapper = actor.GetMapper()
            polydata = mapper.GetInput()
            
            info[segment_name] = {
                'num_points': polydata.GetNumberOfPoints(),
                'num_cells': polydata.GetNumberOfCells(),
                'bounds': actor.GetBounds(),
                'center': actor.GetCenter(),
                'has_contact_patch': any(segment_name in name for name in self.contact_actors.keys())
            }
        
        return info
    
    def export_geometry(self, filepath: str, format: str = 'stl') -> bool:
        """
        Export mannequin geometry to file
        
        Args:
            filepath: Output file path
            format: Export format ('stl', 'obj', 'ply')
            
        Returns:
            True if export successful
        """
        if not self.segment_actors:
            raise VisualizationError("No segments to export")
        
        # Merge all segments
        merged_actor = self.merge_segments_into_body()
        polydata = merged_actor.GetMapper().GetInput()
        
        try:
            if format.lower() == 'stl':
                writer = vtk.vtkSTLWriter()
            elif format.lower() == 'obj':
                writer = vtk.vtkOBJWriter()  
            elif format.lower() == 'ply':
                writer = vtk.vtkPLYWriter()
            else:
                raise VisualizationError(f"Unsupported export format: {format}")
            
            writer.SetFileName(filepath)
            writer.SetInputData(polydata)
            writer.Write()
            
            logger.info(f"Exported mannequin geometry to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {str(e)}")
            return False