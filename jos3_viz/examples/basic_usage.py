"""
Basic usage example for JOS3 Heat Transfer Visualization

Implementation of basic visualization workflow
Source: PRD Section 8 - Example API Usage
User Story: Demonstrate core functionality for biomedical engineers

This example shows how to:
1. Load JOS3 simulation data
2. Calculate external heating/cooling requirements
3. Extract heat transfer data for visualization
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from jos3_viz.core import JOS3DataParser, ExternalHeatCalculator, setup_logger


def create_sample_jos3_data() -> pd.DataFrame:
    """
    Create sample JOS3 simulation data for demonstration
    
    Returns:
        DataFrame with sample JOS3 output format
    """
    # Time points (minutes)
    time_points = np.arange(0, 61, 1)  # 0 to 60 minutes
    n_points = len(time_points)
    
    # Body segments from JOS3
    segments = [
        "Head", "Neck", "Chest", "Back", "Pelvis",
        "LShoulder", "LArm", "LHand", "RShoulder", "RArm", "RHand",
        "LThigh", "LLeg", "LFoot", "RThigh", "RLeg", "RFoot"
    ]
    
    # Create sample data
    data = {}
    
    # Time index
    data['time'] = time_points
    
    # Whole body parameters
    data['Met'] = 80 + 10 * np.sin(time_points / 10)  # Metabolic rate varies
    data['RES'] = 5 + 2 * np.sin(time_points / 15)    # Respiratory heat loss
    
    # Generate realistic data for each segment
    for segment in segments:
        # Core temperatures (°C) - slight variations
        base_core_temp = 37.0
        temp_variation = 0.5 * np.sin(time_points / 20) + 0.2 * np.random.normal(0, 0.1, n_points)
        data[f'Tcr_{segment}'] = base_core_temp + temp_variation
        
        # Skin temperatures (°C) - more variation than core
        base_skin_temp = 34.0 if 'Hand' not in segment and 'Foot' not in segment else 30.0
        skin_variation = 2.0 * np.sin(time_points / 15) + 0.5 * np.random.normal(0, 0.2, n_points)
        data[f'Tsk_{segment}'] = base_skin_temp + skin_variation
        
        # Heat production (W) - varies by segment size
        segment_multiplier = {'Head': 0.8, 'Chest': 2.0, 'Back': 2.0, 'Pelvis': 1.5}.get(segment, 1.0)
        base_qcr = 3.0 * segment_multiplier
        base_qsk = 0.5 * segment_multiplier
        
        data[f'Qcr_{segment}'] = base_qcr + 0.5 * np.sin(time_points / 12)
        data[f'Qsk_{segment}'] = base_qsk + 0.1 * np.sin(time_points / 8)
        
        # Heat losses (W) - environmental dependent
        base_shl = 2.0 * segment_multiplier
        base_lhl = 1.0 * segment_multiplier
        
        data[f'SHLsk_{segment}'] = base_shl + 1.0 * np.sin(time_points / 18) + 0.2 * np.random.normal(0, 0.1, n_points)
        data[f'LHLsk_{segment}'] = base_lhl + 0.5 * np.sin(time_points / 22) + 0.1 * np.random.normal(0, 0.05, n_points)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df = df.set_index('time')
    
    return df


def demonstrate_basic_usage():
    """Demonstrate basic usage of JOS3 visualization tools"""
    
    # Setup logging
    logger = setup_logger("jos3_viz_demo", console_output=True)
    logger.info("Starting JOS3 Heat Transfer Visualization Demo")
    
    try:
        # 1. Create sample data (in real use, load from JOS3 CSV output)
        logger.info("Creating sample JOS3 simulation data...")
        sample_data = create_sample_jos3_data()
        logger.info(f"Created sample data with shape: {sample_data.shape}")
        
        # 2. Initialize data parser
        logger.info("Initializing JOS3 data parser...")
        parser = JOS3DataParser(sample_data)
        
        # Display data summary
        summary = parser.get_data_summary()
        logger.info(f"Data summary: {summary}")
        
        # 3. Initialize heat calculator
        logger.info("Initializing external heat calculator...")
        heat_calc = ExternalHeatCalculator(parser)
        
        # 4. Calculate external heat for specific time and segment
        time_point = 30  # 30 minutes
        segment = "Chest"
        
        logger.info(f"Calculating external heat for {segment} at t={time_point} min...")
        external_heat = heat_calc.calculate_instantaneous_heat(time_point, segment)
        logger.info(f"External heat required for {segment}: {external_heat:.2f} W")
        
        # 5. Calculate total body heat at specific time
        logger.info(f"Calculating total body heat at t={time_point} min...")
        total_body_heat = heat_calc.get_total_body_heat(time_point)
        logger.info(f"Total body external heat: {total_body_heat['total_heat']:.2f} W")
        logger.info(f"Segments requiring heating: {total_body_heat['heating_segments']}")
        logger.info(f"Segments requiring cooling: {total_body_heat['cooling_segments']}")
        
        # 6. Calculate time-averaged heat for a segment
        logger.info(f"Calculating time-averaged heat for {segment} (0-60 min)...")
        avg_heat = heat_calc.calculate_time_averaged_heat(0, 60, segment)
        logger.info(f"Average external heat for {segment}: {avg_heat['average_power']:.2f} W")
        logger.info(f"Total energy over period: {avg_heat['total_energy']:.0f} J")
        logger.info(f"Peak power: {avg_heat['peak_power']:.2f} W")
        
        # 7. Validate energy balance
        logger.info(f"Validating energy balance at t={time_point} min...")
        balance = heat_calc.validate_energy_balance(time_point)
        logger.info(f"Energy balance validation: {balance['message']}")
        
        # 8. Get comprehensive heating/cooling summary
        logger.info("Generating heating/cooling summary...")
        summary = heat_calc.get_heating_cooling_summary((0, 60))
        logger.info(f"Peak heating segment: {summary['critical_segments']['highest_heating']}")
        logger.info(f"Peak cooling segment: {summary['critical_segments']['highest_cooling']}")
        logger.info(f"Whole body average net heat: {summary['whole_body']['average_net']:.2f} W")
        
        # 9. Extract heat transfer data for visualization
        logger.info("Extracting heat transfer data for visualization...")
        sensible_heat = parser.get_heat_transfer_data(time_point, 'sensible')
        latent_heat = parser.get_heat_transfer_data(time_point, 'latent')
        temperatures = parser.get_temperature_data(time_point, 'skin')
        
        logger.info("Sample sensible heat loss data:")
        for segment, heat in list(sensible_heat.items())[:5]:  # Show first 5 segments
            logger.info(f"  {segment}: {heat:.2f} W")
        
        logger.info("Sample skin temperature data:")
        for segment, temp in list(temperatures.items())[:5]:  # Show first 5 segments
            logger.info(f"  {segment}: {temp:.1f} °C")
        
        logger.info("Demo completed successfully!")
        
        return {
            'parser': parser,
            'heat_calculator': heat_calc,
            'sample_data': sample_data,
            'results': {
                'external_heat_chest': external_heat,
                'total_body_heat': total_body_heat,
                'avg_heat_chest': avg_heat,
                'energy_balance': balance,
                'summary': summary
            }
        }
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the demonstration
    demo_results = demonstrate_basic_usage()
    
    print("\n" + "="*60)
    print("JOS3 Heat Transfer Visualization Demo Complete")
    print("="*60)
    print(f"External heat for Chest at 30min: {demo_results['results']['external_heat_chest']:.2f} W")
    print(f"Total body heat at 30min: {demo_results['results']['total_body_heat']['total_heat']:.2f} W")
    print(f"Energy balance: {demo_results['results']['energy_balance']['message']}")
    print("\nCore functionality is working correctly!")
    print("Ready to proceed to Sprint 2: 2D Visualization")