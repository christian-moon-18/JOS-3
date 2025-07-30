"""
3D Model Export for JOS3 Heat Transfer Visualization

Implementation of ModelExporter class for CAD software integration
Source: Agile Plan Sprint 4, Epic 4.2, Task 4.2.1
User Story: As an engineer, I need to export 3D models for CAD software integration and device design

Exports 3D thermal models in STL, OBJ, and VTK formats with embedded heat data.
"""

import json
import struct 
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import numpy as np

try:
    import vtk
    from vtk.util import numpy_support
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False
    warnings.warn("VTK not available. 3D model export will be limited.")

from ..core.logger import get_logger
from ..core.exceptions import VisualizationError
from ..core.data_parser import JOS3DataParser
from ..core.heat_calculator import ExternalHeatCalculator

logger = get_logger(__name__)


class ModelExporter:
    """
    Export 3D thermal models for CAD software integration
    
    Implements: TDD Section 4.2 - 3D Model Export System
    Fulfills: PRD Section 3.3.3 - CAD integration requirements
    """
    
    # Supported export formats
    SUPPORTED_FORMATS = {
        'stl': {
            'extension': '.stl',
            'description': 'STL format for 3D printing and CAD',
            'supports_colors': False,
            'supports_scalars': False,
            'binary': True
        },
        'obj': {
            'extension': '.obj', 
            'description': 'Wavefront OBJ format for 3D software',
            'supports_colors': True,
            'supports_scalars': False,
            'binary': False
        },
        'ply': {
            'extension': '.ply',
            'description': 'PLY format with vertex colors',
            'supports_colors': True,
            'supports_scalars': True,
            'binary': True
        },
        'vtk': {
            'extension': '.vtk',
            'description': 'VTK format with full data support',
            'supports_colors': True,
            'supports_scalars': True,
            'binary': True
        }
    }
    
    # CAD software specific settings
    CAD_PRESETS = {
        'solidworks': {
            'units': 'meters',
            'scale_factor': 1.0,
            'coordinate_system': 'right_handed',
            'preferred_format': 'stl'
        },
        'autocad': {
            'units': 'meters', 
            'scale_factor': 1.0,
            'coordinate_system': 'right_handed',
            'preferred_format': 'obj'
        },
        'blender': {
            'units': 'meters',
            'scale_factor': 1.0,
            'coordinate_system': 'right_handed', 
            'preferred_format': 'obj'
        },
        'fusion360': {
            'units': 'meters',
            'scale_factor': 1.0,
            'coordinate_system': 'right_handed',
            'preferred_format': 'stl'
        }
    }
    
    def __init__(self, data_parser: Optional[JOS3DataParser] = None,
                 heat_calculator: Optional[ExternalHeatCalculator] = None):
        """
        Initialize model exporter
        
        Args:
            data_parser: JOS3DataParser instance (optional for geometry-only export)
            heat_calculator: Heat calculator for thermal data embedding
        """
        self.data_parser = data_parser
        self.heat_calculator = heat_calculator
        
        # Model generation components (if VTK available)
        if VTK_AVAILABLE:
            from ..models.mannequin import MannequinGenerator
            self.mannequin_generator = None  # Will be created when needed
        else:
            self.mannequin_generator = None
        
        # Export settings
        self.default_units = 'meters'
        self.default_scale = 1.0
        
        logger.info("Initialized 3D model exporter")
    
    def export_stl(self, mannequin_data: Dict[str, Any], filepath: str,
                  binary: bool = True, include_contact_geometry: bool = False) -> bool:
        """
        Export 3D model in STL format for 3D printing and CAD
        
        Args:
            mannequin_data: 3D mannequin data or VTK actors
            filepath: Output STL file path
            binary: Use binary STL format (smaller files)
            include_contact_geometry: Include therapeutic device contact areas
            
        Returns:
            True if export successful
        """
        if not VTK_AVAILABLE:
            logger.error("VTK required for STL export")
            return False
        
        try:
            logger.info(f"Exporting STL: {filepath}")
            
            # Create merged geometry from mannequin data
            merged_polydata = self._create_merged_geometry(mannequin_data, include_contact_geometry)
            
            # Setup STL writer
            writer = vtk.vtkSTLWriter()
            writer.SetFileName(filepath)
            writer.SetInputData(merged_polydata)
            
            if binary:
                writer.SetFileTypeToBinary()
            else:
                writer.SetFileTypeToASCII()
            
            # Write file
            writer.Write()
            
            # Add metadata file
            self._write_metadata_file(filepath, 'stl', mannequin_data)
            
            logger.info(f"STL export complete: {Path(filepath).stat().st_size / (1024*1024):.1f} MB")
            return True
            
        except Exception as e:
            logger.error(f"STL export failed: {str(e)}")
            return False
    
    def export_obj(self, mannequin_data: Dict[str, Any], filepath: str,
                  include_materials: bool = True, include_heat_colors: bool = False) -> bool:
        """
        Export 3D model in OBJ format for general 3D software
        
        Args:
            mannequin_data: 3D mannequin data
            filepath: Output OBJ file path  
            include_materials: Create accompanying MTL material file
            include_heat_colors: Color vertices based on thermal data
            
        Returns:
            True if export successful
        """
        if not VTK_AVAILABLE:
            logger.error("VTK required for OBJ export")
            return False
        
        try:
            logger.info(f"Exporting OBJ: {filepath}")
            
            # Create merged geometry
            merged_polydata = self._create_merged_geometry(mannequin_data, include_contact_geometry=True)
            
            # Add heat data colors if requested
            if include_heat_colors and self.data_parser:
                merged_polydata = self._add_heat_data_colors(merged_polydata)
            
            # Setup OBJ writer (VTK doesn't have built-in OBJ writer, so we'll create our own)
            success = self._write_obj_file(merged_polydata, filepath, include_materials)
            
            if success:
                # Add metadata
                self._write_metadata_file(filepath, 'obj', mannequin_data)
                logger.info(f"OBJ export complete: {filepath}")
            
            return success
            
        except Exception as e:
            logger.error(f"OBJ export failed: {str(e)}")
            return False
    
    def export_ply(self, mannequin_data: Dict[str, Any], filepath: str,  
                  include_heat_scalars: bool = True, binary: bool = True) -> bool:
        """
        Export 3D model in PLY format with vertex colors and scalars
        
        Args:
            mannequin_data: 3D mannequin data
            filepath: Output PLY file path
            include_heat_scalars: Include thermal data as vertex scalars
            binary: Use binary PLY format
            
        Returns:
            True if export successful
        """
        if not VTK_AVAILABLE:
            logger.error("VTK required for PLY export")
            return False
        
        try:
            logger.info(f"Exporting PLY: {filepath}")
            
            # Create merged geometry
            merged_polydata = self._create_merged_geometry(mannequin_data, include_contact_geometry=True)
            
            # Add heat data if available
            if include_heat_scalars and self.data_parser:
                merged_polydata = self._add_heat_data_scalars(merged_polydata)
                merged_polydata = self._add_heat_data_colors(merged_polydata)
            
            # Setup PLY writer
            writer = vtk.vtkPLYWriter()
            writer.SetFileName(filepath)
            writer.SetInputData(merged_polydata)
            
            if binary:
                writer.SetFileTypeToBinary()
            else:
                writer.SetFileTypeToASCII()
            
            # Enable color and scalar writing
            writer.SetColorModeToDefault()
            writer.SetArrayName("temperature")  # Use temperature as main scalar
            
            writer.Write()
            
            # Add metadata
            self._write_metadata_file(filepath, 'ply', mannequin_data)
            
            logger.info(f"PLY export complete: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"PLY export failed: {str(e)}")
            return False
    
    def export_vtk(self, mannequin_data: Dict[str, Any], filepath: str,
                  include_all_data: bool = True) -> bool:
        """
        Export 3D model in VTK format with full thermal data support
        
        Args:
            mannequin_data: 3D mannequin data
            filepath: Output VTK file path
            include_all_data: Include all available thermal data arrays
            
        Returns:
            True if export successful
        """
        if not VTK_AVAILABLE:
            logger.error("VTK required for VTK export")
            return False
        
        try:
            logger.info(f"Exporting VTK: {filepath}")
            
            # Create merged geometry with all data
            merged_polydata = self._create_merged_geometry(mannequin_data, include_contact_geometry=True)
            
            if include_all_data and self.data_parser:
                # Add multiple data arrays
                merged_polydata = self._add_comprehensive_thermal_data(merged_polydata)
            
            # Setup VTK writer
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName(filepath)
            writer.SetInputData(merged_polydata)
            writer.SetFileTypeToBinary()  # Binary for efficiency
            
            writer.Write()
            
            # Add comprehensive metadata
            self._write_metadata_file(filepath, 'vtk', mannequin_data, comprehensive=True)
            
            logger.info(f"VTK export complete: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"VTK export failed: {str(e)}")
            return False
    
    def _create_merged_geometry(self, mannequin_data: Dict[str, Any], 
                               include_contact_geometry: bool = False) -> 'vtk.vtkPolyData':
        """Create merged VTK geometry from mannequin data"""
        if not VTK_AVAILABLE:
            raise VisualizationError("VTK required for geometry operations")
        
        # Initialize mannequin generator if needed
        if not self.mannequin_generator:
            from ..models.mannequin import MannequinGenerator
            anthropometry = mannequin_data.get('anthropometry', {})
            self.mannequin_generator = MannequinGenerator(anthropometry)
        
        # Generate body segments
        segment_actors = self.mannequin_generator.generate_body_segments(include_contact_geometry)
        
        # Merge all segments into single polydata
        append_filter = vtk.vtkAppendPolyData()
        
        for segment_name, actor in segment_actors.items():
            # Get polydata from actor
            mapper = actor.GetMapper()
            
            # Apply actor transform to geometry
            transform_filter = vtk.vtkTransformPolyDataFilter()
            transform_filter.SetInputConnection(mapper.GetInputConnection(0, 0))
            
            if actor.GetUserTransform():
                transform_filter.SetTransform(actor.GetUserTransform())
            
            transform_filter.Update()
            append_filter.AddInputData(transform_filter.GetOutput())
        
        append_filter.Update()
        
        return append_filter.GetOutput()
    
    def _add_heat_data_scalars(self, polydata: 'vtk.vtkPolyData') -> 'vtk.vtkPolyData':
        """Add thermal data as vertex scalars"""
        if not self.data_parser:
            return polydata
        
        try:
            # Get final time point thermal data
            final_time = -1
            temp_data = self.data_parser.get_temperature_data(final_time, 'skin')
            cond_data = self.data_parser.get_heat_transfer_data(final_time, 'external_conduction')
            
            num_points = polydata.GetNumberOfPoints()
            
            # Create temperature array
            temp_array = vtk.vtkFloatArray()
            temp_array.SetName("temperature")
            temp_array.SetNumberOfTuples(num_points)
            
            # Create conductive heat array  
            cond_array = vtk.vtkFloatArray()
            cond_array.SetName("conductive_heat")
            cond_array.SetNumberOfTuples(num_points)
            
            # Assign values (simplified - in practice would map segment data to vertices)
            mean_temp = np.mean(list(temp_data.values()))
            mean_cond = np.mean([v for v in cond_data.values() if abs(v) > 0.01])
            
            for i in range(num_points):
                temp_array.SetValue(i, mean_temp)
                cond_array.SetValue(i, mean_cond if not np.isnan(mean_cond) else 0.0)
            
            # Add arrays to polydata
            polydata.GetPointData().AddArray(temp_array)
            polydata.GetPointData().AddArray(cond_array)
            polydata.GetPointData().SetActiveScalars("temperature")
            
            logger.debug(f"Added thermal scalars to {num_points} vertices")
            
        except Exception as e:
            logger.warning(f"Failed to add thermal scalars: {str(e)}")
        
        return polydata
    
    def _add_heat_data_colors(self, polydata: 'vtk.vtkPolyData') -> 'vtk.vtkPolyData':
        """Add thermal data as vertex colors"""
        if not self.data_parser:
            return polydata
        
        try:
            # Get thermal data
            final_time = -1
            temp_data = self.data_parser.get_temperature_data(final_time, 'skin')
            
            # Create color mapping
            from ..visualization.color_mapping import HeatColorMapper
            color_mapper = HeatColorMapper('RdYlBu_r', 'temperature')
            segment_colors = color_mapper.map_temperature_to_color(temp_data)
            
            num_points = polydata.GetNumberOfPoints()
            
            # Create color array (RGB)
            colors = vtk.vtkUnsignedCharArray()
            colors.SetName("Colors")
            colors.SetNumberOfComponents(3)
            colors.SetNumberOfTuples(num_points)
            
            # Use mean color for all vertices (simplified)
            if segment_colors:
                mean_color = np.mean(list(segment_colors.values()), axis=0)[:3]
                color_rgb = (mean_color * 255).astype(np.uint8)
            else:
                color_rgb = np.array([128, 128, 128], dtype=np.uint8)  # Gray default
            
            for i in range(num_points):
                colors.SetTuple3(i, color_rgb[0], color_rgb[1], color_rgb[2])
            
            polydata.GetPointData().SetScalars(colors)
            
            logger.debug(f"Added thermal colors to {num_points} vertices")
            
        except Exception as e:
            logger.warning(f"Failed to add thermal colors: {str(e)}")
        
        return polydata
    
    def _add_comprehensive_thermal_data(self, polydata: 'vtk.vtkPolyData') -> 'vtk.vtkPolyData':
        """Add all available thermal data arrays"""
        if not self.data_parser:
            return polydata
        
        try:
            final_time = -1
            num_points = polydata.GetNumberOfPoints()
            
            # Get all thermal data types
            temp_data = self.data_parser.get_temperature_data(final_time, 'skin')
            core_data = self.data_parser.get_temperature_data(final_time, 'core')
            cond_data = self.data_parser.get_heat_transfer_data(final_time, 'external_conduction')
            
            # Temperature arrays
            skin_temp_array = self._create_scalar_array("skin_temperature", temp_data, num_points)
            core_temp_array = self._create_scalar_array("core_temperature", core_data, num_points)
            cond_heat_array = self._create_scalar_array("conductive_heat", cond_data, num_points)
            
            # Add all arrays
            polydata.GetPointData().AddArray(skin_temp_array)
            polydata.GetPointData().AddArray(core_temp_array)
            polydata.GetPointData().AddArray(cond_heat_array)
            
            # Set default active scalar
            polydata.GetPointData().SetActiveScalars("skin_temperature")
            
            # Add colors based on skin temperature
            polydata = self._add_heat_data_colors(polydata)
            
            logger.debug("Added comprehensive thermal data arrays")
            
        except Exception as e:
            logger.warning(f"Failed to add comprehensive thermal data: {str(e)}")
        
        return polydata
    
    def _create_scalar_array(self, name: str, data: Dict[str, float], num_points: int) -> 'vtk.vtkFloatArray':
        """Create VTK scalar array from segment data"""
        array = vtk.vtkFloatArray()
        array.SetName(name)
        array.SetNumberOfTuples(num_points)
        
        # Use mean value for all points (simplified approach)
        if data and len(data) > 0:
            mean_value = np.mean(list(data.values()))
        else:
            mean_value = 0.0
        
        for i in range(num_points):
            array.SetValue(i, mean_value)
        
        return array
    
    def _write_obj_file(self, polydata: 'vtk.vtkPolyData', filepath: str, 
                       include_materials: bool = True) -> bool:
        """Write OBJ file (custom implementation since VTK lacks OBJ writer)"""
        try:
            obj_path = Path(filepath)
            
            with open(obj_path, 'w') as f:
                # Write header
                f.write(f"# OBJ file generated by JOS3 Heat Transfer Visualization\n")
                f.write(f"# Vertices: {polydata.GetNumberOfPoints()}\n")
                f.write(f"# Faces: {polydata.GetNumberOfCells()}\n\n")
                
                # Write vertices
                for i in range(polydata.GetNumberOfPoints()):
                    point = polydata.GetPoint(i)
                    f.write(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
                
                f.write("\n")
                
                # Write faces
                for i in range(polydata.GetNumberOfCells()):
                    cell = polydata.GetCell(i)
                    if cell.GetNumberOfPoints() == 3:  # Triangle
                        ids = [cell.GetPointId(j) + 1 for j in range(3)]  # OBJ is 1-indexed
                        f.write(f"f {ids[0]} {ids[1]} {ids[2]}\n")
                    elif cell.GetNumberOfPoints() == 4:  # Quad
                        ids = [cell.GetPointId(j) + 1 for j in range(4)]
                        f.write(f"f {ids[0]} {ids[1]} {ids[2]} {ids[3]}\n")
            
            # Write material file if requested
            if include_materials:
                mtl_path = obj_path.with_suffix('.mtl')
                self._write_mtl_file(mtl_path)
                
                # Add material reference to OBJ
                with open(obj_path, 'r') as f:
                    content = f.read()
                
                with open(obj_path, 'w') as f:
                    f.write(f"mtllib {mtl_path.name}\n")
                    f.write(content)
                    f.write(f"\nusemtl thermal_material\n")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to write OBJ file: {str(e)}")
            return False
    
    def _write_mtl_file(self, mtl_path: Path) -> None:
        """Write material file for OBJ"""
        with open(mtl_path, 'w') as f:
            f.write("# Material file for JOS3 thermal model\n\n")
            f.write("newmtl thermal_material\n")
            f.write("Ka 0.2 0.2 0.2\n")  # Ambient
            f.write("Kd 0.8 0.6 0.4\n")  # Diffuse (skin color)
            f.write("Ks 0.1 0.1 0.1\n")  # Specular
            f.write("Ns 10.0\n")          # Shininess
            f.write("d 1.0\n")            # Transparency
    
    def _write_metadata_file(self, model_filepath: str, format: str, 
                            mannequin_data: Dict[str, Any], comprehensive: bool = False) -> None:
        """Write metadata file accompanying the 3D model"""
        metadata_path = Path(model_filepath).with_suffix('.json')
        
        metadata = {
            'format': format,
            'created_by': 'JOS3 Heat Transfer Visualization',
            'version': '1.0',
            'units': self.default_units,
            'scale_factor': self.default_scale,
            'coordinate_system': 'right_handed',
            'model_info': {
                'anthropometry': mannequin_data.get('anthropometry', {}),
                'segments_included': list(mannequin_data.get('segments', {}).keys()) if 'segments' in mannequin_data else [],
                'contact_geometry_included': mannequin_data.get('contact_geometry', False)
            }
        }
        
        # Add thermal data info if available
        if self.data_parser and comprehensive:
            final_time = -1
            temp_data = self.data_parser.get_temperature_data(final_time, 'skin')
            cond_data = self.data_parser.get_heat_transfer_data(final_time, 'external_conduction')
            
            metadata['thermal_data'] = {
                'time_point': final_time,
                'temperature_range': [float(np.min(list(temp_data.values()))), 
                                    float(np.max(list(temp_data.values())))],
                'active_cooling_segments': len([v for v in cond_data.values() if v < -0.1]),
                'active_heating_segments': len([v for v in cond_data.values() if v > 0.1]),
                'data_arrays': ['skin_temperature', 'core_temperature', 'conductive_heat'] if format == 'vtk' else []
            }
        
        # Add CAD software compatibility info
        metadata['cad_compatibility'] = {
            software: info for software, info in self.CAD_PRESETS.items() 
            if info['preferred_format'] == format
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.debug(f"Metadata written: {metadata_path}")
    
    def export_contact_geometry(self, mannequin_data: Dict[str, Any], 
                               output_dir: str, format: str = 'stl') -> List[Path]:
        """
        NEW: Export therapeutic device contact geometry separately
        
        Args:
            mannequin_data: Mannequin data with contact information
            output_dir: Output directory for contact geometry files
            format: Export format
            
        Returns:
            List of exported contact geometry files
        """
        if not VTK_AVAILABLE:
            logger.error("VTK required for contact geometry export")
            return []
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        contact_files = []
        
        try:
            # Get contact data from heat calculator if available
            if self.data_parser and self.heat_calculator:
                final_time = -1
                cond_data = self.data_parser.get_heat_transfer_data(final_time, 'external_conduction')
                
                # Create contact geometry for segments with active heat transfer
                active_segments = {seg: heat for seg, heat in cond_data.items() if abs(heat) > 0.1}
                
                for segment_name, heat_value in active_segments.items():
                    device_type = "cooling" if heat_value < 0 else "heating"
                    
                    # Create contact patch geometry (simplified)
                    contact_geometry = self._create_contact_patch_geometry(segment_name, heat_value)
                    
                    if contact_geometry:
                        filename = f"{device_type}_device_{segment_name.lower()}.{format}"
                        filepath = output_dir / filename
                        
                        # Export based on format
                        if format == 'stl':
                            writer = vtk.vtkSTLWriter()
                            writer.SetFileName(str(filepath))
                            writer.SetInputData(contact_geometry)
                            writer.Write()
                        elif format == 'obj':
                            self._write_obj_file(contact_geometry, str(filepath), False)
                        
                        contact_files.append(filepath)
                        logger.debug(f"Exported contact geometry: {filepath}")
            
            logger.info(f"Exported {len(contact_files)} contact geometry files")
            return contact_files
            
        except Exception as e:
            logger.error(f"Contact geometry export failed: {str(e)}")
            return []
    
    def _create_contact_patch_geometry(self, segment_name: str, heat_value: float) -> Optional['vtk.vtkPolyData']:
        """Create geometry for therapeutic device contact patch"""
        if not VTK_AVAILABLE:
            return None
        
        try:
            # Create a simple rectangular patch (placeholder implementation)
            plane = vtk.vtkPlaneSource()
            plane.SetResolution(10, 10)
            
            # Size based on typical contact areas
            if 'Chest' in segment_name or 'Back' in segment_name:
                plane.SetPoint1(0.25, 0, 0)    # Width
                plane.SetPoint2(0, 0.30, 0)    # Height
            elif 'Arm' in segment_name:
                plane.SetPoint1(0.15, 0, 0)
                plane.SetPoint2(0, 0.25, 0)
            else:
                plane.SetPoint1(0.12, 0, 0)
                plane.SetPoint2(0, 0.15, 0)
            
            plane.Update()
            
            # Add thickness using extrusion
            extrude = vtk.vtkLinearExtrusionFilter()
            extrude.SetInputConnection(plane.GetOutputPort())
            extrude.SetExtrusionTypeToNormalExtrusion()
            extrude.SetScaleFactor(0.005)  # 5mm thickness
            extrude.Update()
            
            return extrude.GetOutput()
            
        except Exception as e:
            logger.warning(f"Failed to create contact patch for {segment_name}: {str(e)}")
            return None
    
    def create_cad_package(self, mannequin_data: Dict[str, Any], output_dir: str,
                          cad_software: str = 'solidworks', include_assembly: bool = True) -> bool:
        """
        Create complete CAD package for specific software
        
        Args:
            mannequin_data: Mannequin data
            output_dir: Output directory
            cad_software: Target CAD software
            include_assembly: Include assembly instructions
            
        Returns:
            True if package created successfully
        """
        if cad_software not in self.CAD_PRESETS:
            raise VisualizationError(f"Unsupported CAD software: {cad_software}")
        
        preset = self.CAD_PRESETS[cad_software]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Creating CAD package for {cad_software}")
            
            # Export main body model
            format = preset['preferred_format']
            body_file = output_dir / f"thermal_mannequin.{format}"
            
            if format == 'stl':
                success = self.export_stl(mannequin_data, str(body_file))
            elif format == 'obj':
                success = self.export_obj(mannequin_data, str(body_file))
            else:
                success = False
            
            if not success:
                return False
            
            # Export contact geometry
            contact_files = self.export_contact_geometry(mannequin_data, output_dir / "devices", format)
            
            # Create assembly instructions
            if include_assembly:
                self._create_assembly_instructions(output_dir, cad_software, body_file, contact_files)
            
            # Create CAD-specific readme
            self._create_cad_readme(output_dir, cad_software, preset)
            
            logger.info(f"CAD package created: {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"CAD package creation failed: {str(e)}")
            return False
    
    def _create_assembly_instructions(self, output_dir: Path, cad_software: str,
                                    body_file: Path, contact_files: List[Path]) -> None:
        """Create assembly instructions for CAD software"""
        instructions_file = output_dir / "ASSEMBLY_INSTRUCTIONS.txt"
        
        with open(instructions_file, 'w') as f:
            f.write(f"JOS3 Thermal Mannequin - {cad_software.title()} Assembly Instructions\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("FILES INCLUDED:\n")
            f.write(f"- Main body model: {body_file.name}\n")
            
            if contact_files:
                f.write("- Therapeutic device contact geometry:\n")
                for contact_file in contact_files:
                    f.write(f"  * {contact_file.name}\n")
            
            f.write("\nASSEMBLY PROCEDURE:\n")
            f.write("1. Import the main body model into your CAD software\n")
            f.write("2. Set units to meters (model is in metric units)\n")
            f.write("3. Import contact geometry files\n")
            f.write("4. Position contact devices on appropriate body segments\n")
            f.write("5. Use mate/constraint features to align devices to body surface\n")
            
            f.write("\nCONTACT DEVICE PLACEMENT:\n")
            if self.data_parser:
                final_time = -1
                cond_data = self.data_parser.get_heat_transfer_data(final_time, 'external_conduction')
                
                for segment, heat_value in cond_data.items():
                    if abs(heat_value) > 0.1:
                        device_type = "cooling" if heat_value < 0 else "heating"
                        f.write(f"- {segment}: {device_type.title()} device ({abs(heat_value):.1f}W)\n")
            
            f.write("\nNOTES:\n")
            f.write("- All dimensions are in meters\n")
            f.write("- Contact devices should have 5mm thickness\n")
            f.write("- Maintain proper thermal contact between devices and body\n")
    
    def _create_cad_readme(self, output_dir: Path, cad_software: str, preset: Dict[str, Any]) -> None:
        """Create CAD-specific readme file"""
        readme_file = output_dir / "README.md"
        
        with open(readme_file, 'w') as f:
            f.write(f"# JOS3 Thermal Mannequin - {cad_software.title()} Package\n\n")
            
            f.write("## Overview\n\n")
            f.write("This package contains 3D thermal mannequin models and therapeutic device geometry ")
            f.write(f"optimized for {cad_software.title()}.\n\n")
            
            f.write("## Technical Specifications\n\n")
            f.write(f"- **Units**: {preset['units']}\n")
            f.write(f"- **Scale**: {preset['scale_factor']}\n")
            f.write(f"- **Coordinate System**: {preset['coordinate_system']}\n")
            f.write(f"- **Preferred Format**: {preset['preferred_format']}\n\n")
            
            f.write("## File Structure\n\n")
            f.write("```\n")
            f.write("thermal_mannequin.{ext}     # Main body model\n".format(ext=preset['preferred_format']))
            f.write("devices/                    # Therapeutic device geometry\n")
            f.write("  cooling_device_*.{ext}    # Cooling devices\n".format(ext=preset['preferred_format']))
            f.write("  heating_device_*.{ext}    # Heating devices\n".format(ext=preset['preferred_format']))
            f.write("ASSEMBLY_INSTRUCTIONS.txt   # Assembly guide\n")
            f.write("*.json                      # Metadata files\n")
            f.write("```\n\n")
            
            f.write("## Usage\n\n")
            f.write(f"1. Open {cad_software.title()}\n")
            f.write("2. Import the main body model\n")
            f.write("3. Follow assembly instructions for device placement\n")
            f.write("4. Use thermal data from metadata for analysis\n\n")
            
            f.write("Generated by JOS3 Heat Transfer Visualization System\n")
    
    def get_export_info(self) -> Dict[str, Any]:
        """
        Get information about export capabilities
        
        Returns:
            Dictionary with export information
        """
        return {
            'supported_formats': list(self.SUPPORTED_FORMATS.keys()),
            'cad_software_presets': list(self.CAD_PRESETS.keys()),
            'vtk_available': VTK_AVAILABLE,
            'thermal_data_available': self.data_parser is not None,
            'mannequin_generator_ready': self.mannequin_generator is not None,
            'default_units': self.default_units,
            'default_scale': self.default_scale
        }