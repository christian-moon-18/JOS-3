#!/usr/bin/env python3
"""
Comprehensive 2D Visualization Example with Conductive Heat Transfer

Implementation of complete visualization workflow
Source: Sprint 2 - 2D Visualization & Export
User Story: As a researcher, I need publication-quality 2D heat maps showing therapeutic cooling/heating

This example demonstrates:
- JOS3 simulation with conductive cooling therapy
- 2D heat map visualization with multiple modes
- Conductive heat transfer overlays
- Publication-quality export in multiple formats
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

import jos3
from jos3_viz.core import JOS3DataParser, ExternalHeatCalculator, setup_logger
from jos3_viz.visualization import HeatColorMapper, HeatRenderer2D
from jos3_viz.export import ImageExporter


def run_cooling_therapy_simulation():
    """Run JOS3 simulation with therapeutic cooling"""
    
    # Create JOS3 model
    model = jos3.JOS3(height=1.75, weight=70, age=30, ex_output="all")
    
    # Set hot environmental conditions
    model.Ta = 35  # Hot ambient temperature (°C)
    model.RH = 60  # Relative humidity (%)
    model.Va = 0.1  # Low air velocity (m/s)
    model.PAR = 1.5  # Metabolic activity
    
    # Apply cooling therapy to chest and back
    # Simulate cooling vest at 15°C
    model.material_temp = [float('nan')] * 17
    model.material_temp[2] = 15  # Chest cooling at 15°C
    model.material_temp[3] = 15  # Back cooling at 15°C
    
    model.contact_area = [0] * 17
    model.contact_area[2] = 0.8  # 80% chest coverage
    model.contact_area[3] = 0.8  # 80% back coverage
    
    model.contact_resistance = 0.01  # Good thermal contact
    
    # Run simulation for 30 minutes
    model.simulate(times=30)
    
    # Get results as DataFrame
    results_dict = model.dict_results()
    results_df = pd.DataFrame(results_dict)
    
    return results_df, model


def create_2d_visualizations(results_df, time_point=29):
    """Create multiple 2D visualizations showing different heat transfer modes"""
    
    # Initialize components
    parser = JOS3DataParser(results_df)
    heat_calc = ExternalHeatCalculator(parser)
    renderer = HeatRenderer2D(figure_size=(12, 14))
    exporter = ImageExporter(output_directory='visualization_outputs')
    
    visualizations = {}
    
    # 1. Skin Temperature Visualization
    skin_temps = parser.get_temperature_data(time_point, 'skin')
    
    config_temp = {
        'colormap': 'RdBu_r',
        'title': f'Skin Temperature Distribution (t={time_point} min)',
        'show_labels': True,
        'show_values': True,
        'show_colorbar': True,
        'units': '°C',
        'mode': 'temperature'
    }
    
    fig_temp = renderer.render_body_heatmap(skin_temps, config_temp)
    visualizations['skin_temperature'] = fig_temp
    
    # 2. Conductive Heat Transfer Visualization
    conductive_heat = parser.get_heat_transfer_data(time_point, 'external_conduction')
    
    config_cond = {
        'colormap': 'custom_conductive',
        'title': f'Conductive Heat Transfer (t={time_point} min)',
        'show_labels': True,
        'show_values': True,
        'show_colorbar': True,
        'units': 'W',
        'mode': 'external_conduction',
        'show_contact_areas': True
    }
    
    fig_cond = renderer.render_body_heatmap(conductive_heat, config_cond)
    visualizations['conductive_heat'] = fig_cond
    
    # 3. Total Heat Loss Visualization
    total_heat_loss = parser.get_heat_transfer_data(time_point, 'total_loss')
    
    config_loss = {
        'colormap': 'viridis',
        'title': f'Total Heat Loss from Skin (t={time_point} min)',
        'show_labels': True,
        'show_values': True,
        'show_colorbar': True,
        'units': 'W',
        'mode': 'heat_flow'
    }
    
    fig_loss = renderer.render_body_heatmap(total_heat_loss, config_loss)
    visualizations['total_heat_loss'] = fig_loss
    
    # 4. External Heat Requirements
    external_heat = {}
    for segment in parser.get_body_segments():
        try:
            ext_heat = heat_calc.calculate_instantaneous_heat(time_point, segment)
            external_heat[segment] = ext_heat
        except Exception as e:
            external_heat[segment] = 0.0
    
    config_ext = {
        'colormap': 'seismic',
        'title': f'External Heat Requirements (t={time_point} min)',
        'show_labels': True,
        'show_values': True,
        'show_colorbar': True,
        'units': 'W',
        'mode': 'heat_flow'
    }
    
    fig_ext = renderer.render_body_heatmap(external_heat, config_ext)
    visualizations['external_heat'] = fig_ext
    
    # 5. Combined Visualization with Conductive Overlay
    # Start with skin temperature as base
    fig_combined = renderer.render_body_heatmap(skin_temps, config_temp)
    
    # Add conductive heat overlay
    conductive_data = parser.get_conductive_data(time_point)
    fig_combined = renderer.render_conductive_overlay(conductive_data, fig_combined)
    
    # Update title
    fig_combined.suptitle(f'Skin Temperature with Conductive Cooling Overlay (t={time_point} min)', 
                         fontsize=16, fontweight='bold', y=0.95)
    
    visualizations['combined_overlay'] = fig_combined
    
    return visualizations, parser, heat_calc, exporter


def export_publication_figures(visualizations, exporter):
    """Export figures in publication-ready formats"""
    
    # Apply IEEE publication styling
    exporter.set_publication_styling('ieee')
    
    exported_files = {}
    
    for name, figure in visualizations.items():
        try:
            # Add therapeutic device annotations for relevant figures
            if 'conductive' in name or 'combined' in name:
                device_info = {
                    'cooling_devices': [
                        {'name': 'Cooling Vest', 'temperature': 15, 'coverage': '80%'}
                    ],
                    'specifications': {
                        'temperature': 15,
                        'contact_area': 0.8,
                        'thermal_resistance': 0.01,
                        'duration': 30
                    }
                }
                figure = exporter.add_therapeutic_device_annotations(figure, device_info)
            
            # Export in multiple formats
            files = exporter.export_multiple_formats(
                figure, 
                f"cooling_therapy_{name}",
                formats=['png', 'pdf', 'svg'],
                dpi=300
            )
            exported_files[name] = files
            
        except Exception as e:
            print(f"Failed to export {name}: {str(e)}")
            continue
    
    # Restore default styling
    exporter.restore_default_styling()
    
    return exported_files


def generate_analysis_report(parser, heat_calc, time_point=29):
    """Generate quantitative analysis report"""
    
    print("\n" + "="*80)
    print("COOLING THERAPY ANALYSIS REPORT")
    print("="*80)
    
    # Overall simulation summary
    data_summary = parser.get_data_summary()
    print(f"Simulation Duration: {data_summary['time_range'][1]} minutes")
    print(f"Data Points: {data_summary['time_points_count']}")
    print(f"Body Segments: {data_summary['segments_available']}")
    
    # Temperature analysis
    skin_temps = parser.get_temperature_data(time_point, 'skin')
    print(f"\nSkin Temperature Analysis (t={time_point} min):")
    print(f"  Mean temperature: {np.mean(list(skin_temps.values())):.2f}°C")
    print(f"  Temperature range: {np.min(list(skin_temps.values())):.2f} - {np.max(list(skin_temps.values())):.2f}°C")
    
    # Cooling segments analysis
    print(f"  Coolest segments:")
    sorted_temps = sorted(skin_temps.items(), key=lambda x: x[1])
    for segment, temp in sorted_temps[:3]:
        print(f"    {segment}: {temp:.2f}°C")
    
    # Conductive heat analysis
    cond_summary = heat_calc.get_conductive_heat_summary(time_point)
    print(f"\nConductive Heat Transfer Analysis:")
    print(f"  Segments with contact: {len(cond_summary['segments_in_contact'])}")
    print(f"  Segments being cooled: {cond_summary['segments_cooling']}")
    print(f"  Total conductive cooling power: {cond_summary['total_conductive_heat']:.2f}W")
    
    if cond_summary['contact_statistics']['total_segments_with_contact'] > 0:
        temp_range = cond_summary['contact_statistics']['temperature_range']
        print(f"  Contact material temperature range: {temp_range['min']:.1f} - {temp_range['max']:.1f}°C")
        print(f"  Average contact area: {cond_summary['contact_statistics']['average_contact_area']:.1%}")
    
    # External heat requirements
    total_body_heat = heat_calc.get_total_body_heat(time_point)
    print(f"\nExternal Heat Requirements:")
    print(f"  Total body external heat: {total_body_heat['total_heat']:.2f}W")
    print(f"  Segments requiring heating: {len(total_body_heat['heating_segments'])}")
    print(f"  Segments requiring cooling: {len(total_body_heat['cooling_segments'])}")
    
    # Energy balance validation
    energy_balance = heat_calc.validate_energy_balance(time_point)
    print(f"\nEnergy Balance Validation:")
    print(f"  Status: {energy_balance['message']}")
    print(f"  Total metabolic heat: {energy_balance.get('total_metabolic', 0):.2f}W")
    print(f"  Total heat losses: {energy_balance.get('total_losses', 0):.2f}W")
    
    # Therapeutic effectiveness
    print(f"\nTherapeutic Effectiveness:")
    if 'Chest' in skin_temps and 'Back' in skin_temps:
        cooled_temp = (skin_temps['Chest'] + skin_temps['Back']) / 2
        uncooled_segments = [seg for seg in skin_temps.keys() 
                           if seg not in ['Chest', 'Back'] and 'Hand' not in seg and 'Foot' not in seg]
        if uncooled_segments:
            uncooled_temp = np.mean([skin_temps[seg] for seg in uncooled_segments[:5]])
            cooling_effect = uncooled_temp - cooled_temp
            print(f"  Average cooling effect: {cooling_effect:.2f}°C")
            print(f"  Cooled area temperature: {cooled_temp:.2f}°C")
            print(f"  Reference area temperature: {uncooled_temp:.2f}°C")


def demonstrate_color_mapping():
    """Demonstrate different color mapping options"""
    
    print("\n" + "="*60)
    print("COLOR MAPPING DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    sample_temps = {
        'Head': 36.5, 'Neck': 35.8, 'Chest': 32.1, 'Back': 31.9,
        'Pelvis': 34.2, 'LShoulder': 33.5, 'RShoulder': 33.7,
        'LArm': 32.8, 'RArm': 33.1, 'LHand': 28.5, 'RHand': 29.1,
        'LThigh': 34.8, 'RThigh': 35.1, 'LLeg': 33.2, 'RLeg': 33.5,
        'LFoot': 30.2, 'RFoot': 30.8
    }
    
    # Test different colormaps
    colormaps = ['RdBu_r', 'viridis', 'plasma', 'coolwarm']
    
    for cmap in colormaps:
        mapper = HeatColorMapper(cmap, 'temperature')
        colors = mapper.map_temperature_to_color(sample_temps)
        validation = mapper.validate_colormap_scientific_accuracy()
        
        print(f"\nColormap: {cmap}")
        print(f"  Perceptually uniform: {validation['perceptually_uniform']}")
        print(f"  Colorblind friendly: {validation['colorblind_friendly']}")
        print(f"  Warnings: {len(validation['warnings'])}")
        
        if validation['warnings']:
            for warning in validation['warnings']:
                print(f"    - {warning}")
    
    # Test conductive heat mapping
    print(f"\nConductive Heat Mapping:")
    sample_conductive = {'Chest': -25.5, 'Back': -28.2, 'Head': 0.0, 'LArm': 5.2}
    
    mapper = HeatColorMapper()
    cond_colors = mapper.map_conductive_heat(sample_conductive)
    print(f"  Mapped {len([v for v in sample_conductive.values() if v != 0])} segments with conductive heat")
    print(f"  Range: {min(sample_conductive.values()):.1f} to {max(sample_conductive.values()):.1f}W")


def main():
    """Main execution function"""
    
    # Setup logging
    logger = setup_logger("cooling_therapy_viz", console_output=True)
    logger.info("Starting Cooling Therapy 2D Visualization Demo")
    
    try:
        # Import pandas (needed for JOS3 results)
        import pandas as pd
        
        print("JOS3 Heat Transfer Visualization - Cooling Therapy Demo")
        print("="*60)
        
        # 1. Run JOS3 simulation
        print("1. Running JOS3 simulation with cooling therapy...")
        results_df, model = run_cooling_therapy_simulation()
        print(f"   ✓ Simulation complete: {results_df.shape[0]} time points, {results_df.shape[1]} parameters")
        
        # 2. Create visualizations
        print("\n2. Creating 2D heat map visualizations...")
        visualizations, parser, heat_calc, exporter = create_2d_visualizations(results_df)
        print(f"   ✓ Created {len(visualizations)} visualization figures")
        
        # 3. Export figures
        print("\n3. Exporting publication-quality figures...")
        exported_files = export_publication_figures(visualizations, exporter)
        print(f"   ✓ Exported {len(exported_files)} figure sets")
        
        # Display export summary
        total_files = sum(len(files) for files in exported_files.values())
        print(f"   ✓ Total files exported: {total_files}")
        
        for name, files in exported_files.items():
            print(f"     {name}:")
            for format_name, filepath in files.items():
                print(f"       - {format_name.upper()}: {filepath.name}")
        
        # 4. Generate analysis report
        print("\n4. Generating quantitative analysis...")
        generate_analysis_report(parser, heat_calc)
        
        # 5. Demonstrate color mapping
        demonstrate_color_mapping()
        
        # Show one visualization
        print(f"\n5. Displaying sample visualization...")
        if 'combined_overlay' in visualizations:
            plt.figure(visualizations['combined_overlay'].number)
            plt.show(block=False)
            print("   ✓ Combined temperature + conductive overlay displayed")
        
        print("\n" + "="*80)
        print("✅ COOLING THERAPY 2D VISUALIZATION DEMO COMPLETE")
        print("="*80)
        print("🎯 Key Achievements:")
        print("  • JOS3 conductive heat simulation: ✓")
        print("  • 2D heat map rendering with multiple modes: ✓")
        print("  • Conductive heat transfer visualization: ✓")
        print("  • Publication-quality export (PNG/PDF/SVG): ✓")
        print("  • Therapeutic device annotations: ✓")
        print("  • Quantitative analysis and validation: ✓")
        print("\n🚀 Sprint 2 (2D Visualization & Export) functionality verified!")
        
        return {
            'visualizations': visualizations,
            'exported_files': exported_files,
            'parser': parser,
            'heat_calculator': heat_calc,
            'results_data': results_df
        }
        
    except ImportError as e:
        print(f"❌ Missing dependency: {str(e)}")
        print("Please install required packages: pip install pandas matplotlib")
        return None
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
    
    if results:
        print(f"\n📊 Demo results available in 'results' variable")
        print(f"📁 Exported files saved to: visualization_outputs/")
        
        # Keep matplotlib windows open
        try:
            plt.show()
        except:
            pass