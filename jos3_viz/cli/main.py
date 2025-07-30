"""
Main Command Line Interface for JOS3 Heat Transfer Visualization

Implementation of Click-based CLI for easy tool usage
Source: Agile Plan Sprint 4, Epic 4.3, Task 4.3.1
User Story: As a user, I need simple commands to generate visualizations without coding

Provides commands: visualize, calculate-heat, export-video, export-model, analyze-therapy
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import click
import pandas as pd
import numpy as np

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from jos3_viz.core import JOS3DataParser, ExternalHeatCalculator, setup_logger
from jos3_viz.visualization import HeatRenderer2D, HeatColorMapper
from jos3_viz.export import ImageExporter

# Optional imports with availability checks
try:
    from jos3_viz.export.video_export import VideoExporter
    VIDEO_EXPORT_AVAILABLE = True
except ImportError:
    VIDEO_EXPORT_AVAILABLE = False

try:
    from jos3_viz.export.model_export import ModelExporter
    MODEL_EXPORT_AVAILABLE = True
except ImportError:
    MODEL_EXPORT_AVAILABLE = False

try:
    from jos3_viz.visualization.renderer_3d import HeatRenderer3D
    RENDERER_3D_AVAILABLE = True
except ImportError:
    RENDERER_3D_AVAILABLE = False


# Global CLI context
class CLIContext:
    """CLI context for sharing state between commands"""
    def __init__(self):
        self.config = {}
        self.verbose = False
        self.output_dir = Path.cwd()
        self.logger = None


@click.group(invoke_without_command=True)
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Configuration file path (YAML/JSON)')
@click.option('--output-dir', '-o', type=click.Path(),
              default='.', help='Output directory for generated files')
@click.option('--verbose', '-v', is_flag=True, 
              help='Enable verbose logging')
@click.version_option(version='1.0', prog_name='jos3-viz')
@click.pass_context
def cli(ctx, config, output_dir, verbose):
    """
    JOS3 Heat Transfer Visualization - Command Line Interface
    
    A powerful tool for thermal simulation visualization and analysis.
    
    Examples:
        jos3-viz visualize data.csv --mode 2d --time 30
        jos3-viz export-video data.csv --start 0 --end 60 --fps 10
        jos3-viz analyze-therapy data.csv --device cooling-vest
    """
    # Initialize context
    ctx.ensure_object(CLIContext)
    ctx.obj.verbose = verbose
    ctx.obj.output_dir = Path(output_dir)
    ctx.obj.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    ctx.obj.logger = setup_logger("cli", console_output=True, level=log_level)
    
    # Load configuration if provided
    if config:
        ctx.obj.config = load_config(config)
        if verbose:
            click.echo(f"Loaded configuration from: {config}")
    
    # Show help if no command provided
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        
        # Show available capabilities
        click.echo("\n🔧 Available Features:")
        click.echo(f"   • 2D Visualization: ✅ Ready")
        click.echo(f"   • 3D Visualization: {'✅ Ready' if RENDERER_3D_AVAILABLE else '❌ VTK required'}")
        click.echo(f"   • Video Export: {'✅ Ready' if VIDEO_EXPORT_AVAILABLE else '❌ FFmpeg/Pillow required'}")
        click.echo(f"   • Model Export: {'✅ Ready' if MODEL_EXPORT_AVAILABLE else '❌ VTK required'}")
        
        click.echo("\n💡 Quick Start:")
        click.echo("   jos3-viz visualize --help")
        click.echo("   jos3-viz analyze-therapy --help")


@cli.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--mode', '-m', type=click.Choice(['2d', '3d', 'both']), 
              default='2d', help='Visualization mode')
@click.option('--time', '-t', type=int, default=-1,
              help='Time point to visualize (-1 for last)')
@click.option('--viz-type', type=click.Choice(['temperature', 'external_conduction', 'both']),
              default='temperature', help='Type of visualization')
@click.option('--colormap', type=str, default='RdYlBu_r',
              help='Color scheme to use')
@click.option('--output', type=str, help='Output filename (auto-generated if not specified)')
@click.option('--format', 'output_format', type=click.Choice(['png', 'svg', 'pdf']),
              default='png', help='Output format')
@click.option('--dpi', type=int, default=150, help='Resolution in DPI')
@click.pass_context
def visualize(ctx, data_file, mode, time, viz_type, colormap, output, output_format, dpi):
    """
    Create thermal visualization from JOS3 simulation data
    
    DATA_FILE: Path to JOS3 simulation CSV file
    
    Examples:
        jos3-viz visualize simulation.csv --mode 2d --time 30
        jos3-viz visualize data.csv --viz-type external_conduction --colormap custom_conductive
    """
    ctx.obj.logger.info(f"Creating {mode} visualization from: {data_file}")
    
    try:
        # Load and parse data
        results_df = pd.read_csv(data_file)
        parser = JOS3DataParser(results_df)
        
        # Generate output filename if not provided
        if not output:
            base_name = Path(data_file).stem
            output = f"{base_name}_{mode}_{viz_type}_t{time}.{output_format}"
        
        output_path = ctx.obj.output_dir / output
        
        if mode == '2d' or mode == 'both':
            success = create_2d_visualization(parser, time, viz_type, colormap, 
                                            output_path, output_format, dpi)
            if success:
                click.echo(f"✅ 2D visualization saved: {output_path}")
        
        if mode == '3d' or mode == 'both':
            if not RENDERER_3D_AVAILABLE:
                click.echo("❌ 3D visualization requires VTK. Install with: pip install vtk")
                return
            
            if mode == 'both':
                output_path = ctx.obj.output_dir / output.replace('.', '_3d.')
            
            success = create_3d_visualization(parser, time, viz_type, output_path)
            if success:
                click.echo(f"✅ 3D visualization saved: {output_path}")
        
    except Exception as e:
        ctx.obj.logger.error(f"Visualization failed: {str(e)}")
        click.echo(f"❌ Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--segments', type=str, help='Comma-separated segment names (e.g., Chest,Back)')
@click.option('--time', '-t', type=int, default=-1, help='Time point to analyze')
@click.option('--output', type=str, help='Output CSV filename')
@click.option('--summary', is_flag=True, help='Show summary statistics')
@click.pass_context
def calculate_heat(ctx, data_file, segments, time, output, summary):
    """
    Calculate heat transfer values for specific body segments
    
    DATA_FILE: Path to JOS3 simulation CSV file
    
    Examples:
        jos3-viz calculate-heat data.csv --segments Chest,Back --time 30
        jos3-viz calculate-heat data.csv --summary
    """
    ctx.obj.logger.info(f"Calculating heat transfer from: {data_file}")
    
    try:
        # Load and parse data
        results_df = pd.read_csv(data_file)
        parser = JOS3DataParser(results_df)
        heat_calc = ExternalHeatCalculator(parser)
        
        # Get heat transfer data
        temp_data = parser.get_temperature_data(time, 'skin') 
        cond_data = parser.get_heat_transfer_data(time, 'external_conduction')
        
        # Filter segments if specified
        if segments:
            segment_list = [s.strip() for s in segments.split(',')]
            temp_data = {k: v for k, v in temp_data.items() if k in segment_list}
            cond_data = {k: v for k, v in cond_data.items() if k in segment_list}
        
        # Display results
        click.echo(f"\n🌡️ Heat Transfer Analysis (Time: {time})")
        click.echo("=" * 50)
        
        for segment in temp_data.keys():
            temp = temp_data.get(segment, 0)
            cond_heat = cond_data.get(segment, 0)
            
            status = ""
            if abs(cond_heat) > 0.1:
                status = "🔥 Heating" if cond_heat > 0 else "❄️ Cooling"
            
            click.echo(f"{segment:12} | {temp:6.1f}°C | {cond_heat:8.1f}W | {status}")
        
        # Summary statistics
        if summary:
            click.echo(f"\n📊 Summary Statistics:")
            click.echo(f"   Mean temperature: {np.mean(list(temp_data.values())):.1f}°C")
            click.echo(f"   Temperature range: {np.min(list(temp_data.values())):.1f} - {np.max(list(temp_data.values())):.1f}°C")
            
            active_cooling = [v for v in cond_data.values() if v < -0.1]
            active_heating = [v for v in cond_data.values() if v > 0.1]
            
            click.echo(f"   Total cooling power: {abs(sum(active_cooling)):.1f}W")
            click.echo(f"   Total heating power: {sum(active_heating):.1f}W")
            click.echo(f"   Active segments: {len(active_cooling) + len(active_heating)}")
        
        # Save to CSV if requested
        if output:
            output_path = ctx.obj.output_dir / output
            heat_df = pd.DataFrame({
                'Segment': list(temp_data.keys()),
                'Temperature_C': list(temp_data.values()),
                'Conductive_Heat_W': [cond_data.get(seg, 0) for seg in temp_data.keys()]
            })
            heat_df.to_csv(output_path, index=False)
            click.echo(f"✅ Heat data saved: {output_path}")
        
    except Exception as e:
        ctx.obj.logger.error(f"Heat calculation failed: {str(e)}")
        click.echo(f"❌ Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('data_file', type=click.Path(exists=True))  
@click.option('--start', type=int, default=0, help='Start time index')
@click.option('--end', type=int, help='End time index (default: last)')
@click.option('--fps', type=int, default=10, help='Frames per second')
@click.option('--preset', type=click.Choice(['presentation', 'research', 'quick_preview', 'publication']),
              default='presentation', help='Animation preset')
@click.option('--mode', type=click.Choice(['temperature', 'external_conduction', 'both']),
              default='temperature', help='Visualization mode')
@click.option('--format', 'video_format', type=click.Choice(['mp4', 'avi', 'gif']),
              default='mp4', help='Video format')
@click.option('--output', type=str, help='Output video filename')
@click.pass_context
def export_video(ctx, data_file, start, end, fps, preset, mode, video_format, output):
    """
    Create video animation from thermal simulation time series
    
    DATA_FILE: Path to JOS3 simulation CSV file
    
    Examples:
        jos3-viz export-video data.csv --start 0 --end 60 --fps 15
        jos3-viz export-video data.csv --preset research --format mp4
    """
    if not VIDEO_EXPORT_AVAILABLE:
        click.echo("❌ Video export requires additional packages. Install with:")
        click.echo("   pip install imageio[ffmpeg] pillow")
        return
    
    ctx.obj.logger.info(f"Creating video animation from: {data_file}")
    
    try:
        # Load and parse data
        results_df = pd.read_csv(data_file)
        parser = JOS3DataParser(results_df)
        heat_calc = ExternalHeatCalculator(parser)
        
        # Initialize video exporter
        video_exporter = VideoExporter(parser, heat_calc)
        
        # Setup progress callback
        def progress_callback(completed, total, percentage, eta):
            if completed % max(1, total // 20) == 0:  # Update every 5%
                click.echo(f"Progress: {percentage:.1f}% ({completed}/{total}) - ETA: {eta:.1f}s")
        
        video_exporter.set_progress_callback(progress_callback)
        
        # Generate output filename if not provided
        if not output:
            base_name = Path(data_file).stem
            output = f"{base_name}_{preset}_{mode}_animation.{video_format}"
        
        output_path = ctx.obj.output_dir / output
        
        # Create animation
        click.echo(f"🎬 Creating {video_format.upper()} animation...")
        click.echo(f"   Preset: {preset}")
        click.echo(f"   Mode: {mode}")
        click.echo(f"   Time range: {start} to {end or 'end'}")
        
        success = video_exporter.create_video_animation(
            str(output_path), start, end, preset, mode
        )
        
        if success:
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            click.echo(f"✅ Video animation saved: {output_path} ({file_size:.1f} MB)")
        else:
            click.echo("❌ Video creation failed", err=True)
            sys.exit(1)
        
    except Exception as e:
        ctx.obj.logger.error(f"Video export failed: {str(e)}")
        click.echo(f"❌ Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--format', 'model_format', type=click.Choice(['stl', 'obj', 'ply', 'vtk']),
              default='stl', help='3D model format')
@click.option('--time', '-t', type=int, default=-1, help='Time point for thermal data')
@click.option('--include-heat-data', is_flag=True, help='Embed thermal data in model')
@click.option('--include-contacts', is_flag=True, help='Include therapeutic device geometry')
@click.option('--cad-package', type=click.Choice(['solidworks', 'autocad', 'blender', 'fusion360']),
              help='Create complete CAD package for specific software')
@click.option('--output', type=str, help='Output filename or directory')
@click.pass_context  
def export_model(ctx, data_file, model_format, time, include_heat_data, include_contacts, cad_package, output):
    """
    Export 3D thermal model for CAD software integration
    
    DATA_FILE: Path to JOS3 simulation CSV file
    
    Examples:
        jos3-viz export-model data.csv --format stl --include-contacts
        jos3-viz export-model data.csv --cad-package solidworks --output ./cad_files/
    """
    if not MODEL_EXPORT_AVAILABLE:
        click.echo("❌ Model export requires VTK. Install with: pip install vtk")
        return
    
    ctx.obj.logger.info(f"Exporting 3D model from: {data_file}")
    
    try:
        # Load and parse data
        results_df = pd.read_csv(data_file)
        parser = JOS3DataParser(results_df)
        heat_calc = ExternalHeatCalculator(parser)
        
        # Initialize model exporter
        model_exporter = ModelExporter(parser, heat_calc)
        
        # Prepare mannequin data
        anthropometry = {'height': 1.75, 'weight': 70.0, 'age': 30}  # Default values
        mannequin_data = {
            'anthropometry': anthropometry,
            'contact_geometry': include_contacts,
            'thermal_data_time': time
        }
        
        if cad_package:
            # Create complete CAD package
            if not output:
                output = f"cad_package_{cad_package}"
            
            output_dir = ctx.obj.output_dir / output
            
            click.echo(f"🏗️ Creating CAD package for {cad_package.title()}...")
            success = model_exporter.create_cad_package(
                mannequin_data, str(output_dir), cad_package, include_assembly=True
            )
            
            if success:
                click.echo(f"✅ CAD package created: {output_dir}")
                # List created files
                files = list(output_dir.rglob("*.*"))
                click.echo(f"   Generated {len(files)} files:")
                for file in files[:10]:  # Show first 10 files
                    click.echo(f"     • {file.relative_to(output_dir)}")
                if len(files) > 10:
                    click.echo(f"     ... and {len(files) - 10} more files")
            
        else:
            # Export single model file
            if not output:
                base_name = Path(data_file).stem
                output = f"{base_name}_thermal_model.{model_format}"
            
            output_path = ctx.obj.output_dir / output
            
            click.echo(f"📄 Exporting {model_format.upper()} model...")
            
            # Choose export method based on format
            if model_format == 'stl':
                success = model_exporter.export_stl(mannequin_data, str(output_path),
                                                  include_contact_geometry=include_contacts)
            elif model_format == 'obj':
                success = model_exporter.export_obj(mannequin_data, str(output_path),
                                                  include_heat_colors=include_heat_data)
            elif model_format == 'ply':
                success = model_exporter.export_ply(mannequin_data, str(output_path),
                                                  include_heat_scalars=include_heat_data)  
            elif model_format == 'vtk':
                success = model_exporter.export_vtk(mannequin_data, str(output_path),
                                                  include_all_data=include_heat_data)
            
            if success:
                file_size = output_path.stat().st_size / (1024 * 1024)  # MB
                click.echo(f"✅ 3D model exported: {output_path} ({file_size:.1f} MB)")
        
    except Exception as e:
        ctx.obj.logger.error(f"Model export failed: {str(e)}")
        click.echo(f"❌ Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--device', type=click.Choice(['cooling-vest', 'heating-pad', 'comprehensive', 'auto-detect']),
              default='auto-detect', help='Therapeutic device type')
@click.option('--metrics', is_flag=True, help='Calculate effectiveness metrics')
@click.option('--recommendations', is_flag=True, help='Generate optimization recommendations')
@click.option('--report', type=str, help='Generate detailed report file')
@click.pass_context
def analyze_therapy(ctx, data_file, device, metrics, recommendations, report):
    """
    Analyze therapeutic cooling/heating device effectiveness
    
    DATA_FILE: Path to JOS3 simulation CSV file
    
    Examples:
        jos3-viz analyze-therapy data.csv --device cooling-vest --metrics
        jos3-viz analyze-therapy data.csv --recommendations --report therapy_analysis.md
    """
    ctx.obj.logger.info(f"Analyzing therapeutic intervention from: {data_file}")
    
    try:
        # Load and parse data
        results_df = pd.read_csv(data_file)
        parser = JOS3DataParser(results_df)
        heat_calc = ExternalHeatCalculator(parser)
        
        final_time = -1
        
        # Get thermal data
        temp_data = parser.get_temperature_data(final_time, 'skin')
        cond_data = parser.get_heat_transfer_data(final_time, 'external_conduction')
        cooling_summary = heat_calc.get_conductive_heat_summary(final_time)
        
        # Auto-detect device type if requested
        if device == 'auto-detect':
            device = detect_device_type(cond_data)
            click.echo(f"🔍 Detected device type: {device}")
        
        # Basic analysis
        click.echo(f"\n🏥 Therapeutic Device Analysis: {device.replace('-', ' ').title()}")
        click.echo("=" * 60)
        
        click.echo(f"📊 Thermal Status:")
        click.echo(f"   Mean skin temperature: {np.mean(list(temp_data.values())):.1f}°C")
        click.echo(f"   Temperature range: {np.min(list(temp_data.values())):.1f} - {np.max(list(temp_data.values())):.1f}°C")
        
        active_segments = cooling_summary.get('segments_cooling', [])
        total_power = abs(cooling_summary.get('total_conductive_heat', 0))
        
        click.echo(f"\n❄️ Device Performance:")
        click.echo(f"   Active cooling segments: {len(active_segments)}")
        click.echo(f"   Total cooling power: {total_power:.1f}W")
        click.echo(f"   Active segments: {', '.join(active_segments[:5])}")
        
        # Calculate effectiveness metrics
        if metrics:
            effectiveness = calculate_therapy_effectiveness(temp_data, cond_data, device)
            
            click.echo(f"\n📈 Effectiveness Metrics:")
            click.echo(f"   Cooling efficiency: {effectiveness['efficiency']:.1%}")
            click.echo(f"   Coverage ratio: {effectiveness['coverage']:.1%}")
            click.echo(f"   Temperature reduction: {effectiveness['temp_reduction']:.1f}°C")
            click.echo(f"   Power density: {effectiveness['power_density']:.1f}W/m²")
        
        # Generate recommendations
        if recommendations:
            recs = generate_therapy_recommendations(temp_data, cond_data, device)
            
            click.echo(f"\n💡 Optimization Recommendations:")
            for i, rec in enumerate(recs, 1):
                click.echo(f"   {i}. {rec}")
        
        # Generate detailed report
        if report:
            report_path = ctx.obj.output_dir / report
            generate_therapy_report(report_path, data_file, device, temp_data, cond_data, 
                                  cooling_summary, effectiveness if metrics else None,
                                  recs if recommendations else None)
            click.echo(f"✅ Detailed report saved: {report_path}")
        
    except Exception as e:
        ctx.obj.logger.error(f"Therapy analysis failed: {str(e)}")
        click.echo(f"❌ Error: {str(e)}", err=True)
        sys.exit(1)


# Helper functions
def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file"""
    config_path = Path(config_path)
    
    try:
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                import yaml
                return yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
    except Exception as e:
        click.echo(f"❌ Failed to load config: {str(e)}", err=True)
        sys.exit(1)


def create_2d_visualization(parser: JOS3DataParser, time: int, viz_type: str,
                           colormap: str, output_path: Path, output_format: str, dpi: int) -> bool:
    """Create 2D visualization"""
    try:
        renderer_2d = HeatRenderer2D()
        
        if viz_type == 'temperature':
            heat_data = parser.get_temperature_data(time, 'skin')
            mode = 'temperature'
        elif viz_type == 'external_conduction':
            heat_data = parser.get_heat_transfer_data(time, 'external_conduction')
            mode = 'external_conduction'
        else:  # both
            heat_data = parser.get_temperature_data(time, 'skin')
            mode = 'temperature'
        
        # Create visualization
        fig = renderer_2d.render_body_heatmap(
            heat_data,
            config={
                'title': f"{viz_type.replace('_', ' ').title()} (t={time} min)",
                'colormap': colormap,
                'show_colorbar': True,
                'mode': mode
            }
        )
        
        # Save with specified format and DPI
        fig.savefig(str(output_path), format=output_format, dpi=dpi, 
                   bbox_inches='tight', facecolor='white')
        
        return True
        
    except Exception as e:
        click.echo(f"2D visualization failed: {str(e)}", err=True)
        return False


def create_3d_visualization(parser: JOS3DataParser, time: int, viz_type: str, output_path: Path) -> bool:
    """Create 3D visualization"""
    try:
        renderer_3d = HeatRenderer3D()
        
        # Setup 3D pipeline
        anthropometry = {'height': 1.75, 'weight': 70.0, 'age': 30}
        renderer_3d.setup_vtk_pipeline(anthropometry)
        
        # Get data and render
        if viz_type == 'temperature':
            heat_data = parser.get_temperature_data(time, 'skin')
            renderer_3d.render_heat_on_model(heat_data, mode='temperature')
        elif viz_type == 'external_conduction':
            heat_data = parser.get_heat_transfer_data(time, 'external_conduction')
            renderer_3d.render_heat_on_model(heat_data, mode='external_conduction')
        
        # Save screenshot
        success = renderer_3d.save_screenshot(str(output_path), magnification=2)
        return success
        
    except Exception as e:
        click.echo(f"3D visualization failed: {str(e)}", err=True)
        return False


def detect_device_type(cond_data: Dict[str, float]) -> str:
    """Auto-detect therapeutic device type from conductive heat data"""
    active_segments = [seg for seg, heat in cond_data.items() if abs(heat) > 0.1]
    
    # Simple heuristics based on active segments
    torso_segments = ['Chest', 'Back']
    limb_segments = ['LArm', 'RArm', 'LLeg', 'RLeg', 'LThigh', 'RThigh']
    
    torso_active = sum(1 for seg in torso_segments if seg in active_segments)
    limb_active = sum(1 for seg in limb_segments if seg in active_segments)
    
    if torso_active >= 2 and limb_active >= 2:
        return 'comprehensive'
    elif torso_active >= 1:
        return 'cooling-vest'
    elif limb_active >= 1:
        return 'heating-pad'
    else:
        return 'unknown'


def calculate_therapy_effectiveness(temp_data: Dict[str, float], 
                                  cond_data: Dict[str, float], device: str) -> Dict[str, float]:
    """Calculate therapeutic device effectiveness metrics"""
    # Simplified effectiveness calculation
    active_segments = [seg for seg, heat in cond_data.items() if abs(heat) > 0.1]
    total_segments = len(temp_data)
    total_power = sum(abs(heat) for heat in cond_data.values() if abs(heat) > 0.1)
    
    # Estimate temperature reduction (simplified)
    mean_temp = np.mean(list(temp_data.values()))
    baseline_temp = 37.0  # Assumed baseline
    temp_reduction = max(0, baseline_temp - mean_temp)
    
    return {
        'efficiency': min(1.0, temp_reduction / 5.0),  # Max 5°C reduction
        'coverage': len(active_segments) / total_segments,
        'temp_reduction': temp_reduction,
        'power_density': total_power / 2.0  # Rough body surface area estimate
    }


def generate_therapy_recommendations(temp_data: Dict[str, float],
                                   cond_data: Dict[str, float], device: str) -> List[str]:
    """Generate optimization recommendations"""
    recommendations = []
    
    mean_temp = np.mean(list(temp_data.values()))
    active_power = sum(abs(heat) for heat in cond_data.values() if abs(heat) > 0.1)
    
    if mean_temp > 36.5:
        recommendations.append("Consider increasing cooling power or coverage area")
    
    if active_power < 20:
        recommendations.append("Low cooling power detected - check device contact and settings")
    
    hot_segments = [seg for seg, temp in temp_data.items() if temp > 37.0]
    if hot_segments and all(abs(cond_data.get(seg, 0)) < 0.1 for seg in hot_segments):
        recommendations.append(f"Consider adding cooling to hot segments: {', '.join(hot_segments[:3])}")
    
    if not recommendations:
        recommendations.append("Current therapeutic configuration appears optimal")
    
    return recommendations


def generate_therapy_report(report_path: Path, data_file: str, device: str,
                           temp_data: Dict[str, float], cond_data: Dict[str, float],
                           cooling_summary: Dict[str, Any], effectiveness: Optional[Dict[str, float]],
                           recommendations: Optional[List[str]]) -> None:
    """Generate detailed therapy analysis report"""
    
    with open(report_path, 'w') as f:
        f.write(f"# Therapeutic Device Analysis Report\n\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Data Source:** {data_file}\n")
        f.write(f"**Device Type:** {device.replace('-', ' ').title()}\n\n")
        
        f.write(f"## Thermal Status\n\n")
        f.write(f"- Mean skin temperature: {np.mean(list(temp_data.values())):.1f}°C\n")
        f.write(f"- Temperature range: {np.min(list(temp_data.values())):.1f} - {np.max(list(temp_data.values())):.1f}°C\n")
        f.write(f"- Temperature standard deviation: {np.std(list(temp_data.values())):.1f}°C\n\n")
        
        f.write(f"## Device Performance\n\n")
        active_segments = cooling_summary.get('segments_cooling', [])
        total_power = abs(cooling_summary.get('total_conductive_heat', 0))
        
        f.write(f"- Active cooling segments: {len(active_segments)}\n")
        f.write(f"- Total cooling power: {total_power:.1f}W\n")
        f.write(f"- Active segments: {', '.join(active_segments)}\n\n")
        
        if effectiveness:
            f.write(f"## Effectiveness Metrics\n\n")
            f.write(f"- Cooling efficiency: {effectiveness['efficiency']:.1%}\n")
            f.write(f"- Coverage ratio: {effectiveness['coverage']:.1%}\n")
            f.write(f"- Temperature reduction: {effectiveness['temp_reduction']:.1f}°C\n")
            f.write(f"- Power density: {effectiveness['power_density']:.1f}W/m²\n\n")
        
        if recommendations:
            f.write(f"## Optimization Recommendations\n\n")
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")
            f.write("\n")
        
        f.write(f"---\n")
        f.write(f"*Report generated by JOS3 Heat Transfer Visualization CLI v1.0*\n")


if __name__ == '__main__':
    cli()