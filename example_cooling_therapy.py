#!/usr/bin/env python3
"""
Example: Cooling Therapy Simulation with JOS-3 Conductive Heat Transfer

This example demonstrates how to use the new conductive heat transfer feature
to simulate therapeutic cooling applications.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib.pyplot as plt
import jos3

def cooling_therapy_example():
    """Demonstrate cooling therapy simulation"""
    
    print("=== JOS-3 Conductive Heat Transfer Example ===")
    print("Simulating therapeutic cooling therapy\n")
    
    # Create model
    model = jos3.JOS3(height=1.75, weight=70, age=30, ex_output="all")
    
    # === Phase 1: Baseline (Hot Environment) ===
    print("Phase 1: Establishing baseline in hot environment...")
    model.Ta = 35      # Hot air temperature (35°C)
    model.RH = 60      # High humidity (60%)
    model.Va = 0.1     # Low air movement
    model.PAR = 1.5    # Light activity
    
    # Run baseline simulation
    model.simulate(times=10)  # 10 minutes to reach steady state
    baseline_temp = model.TskMean
    print(f"Baseline mean skin temperature: {baseline_temp:.2f}°C")
    
    # === Phase 2: Apply Cooling Therapy ===
    print("\nPhase 2: Applying cooling therapy...")
    
    # Apply cooling pads to torso (chest, back, pelvis)
    cooling_temps = np.full(17, np.nan)  # Start with no contact
    cooling_areas = np.zeros(17)         # No contact area initially
    
    # Set cooling for torso segments
    torso_segments = [2, 3, 4]  # Chest, Back, Pelvis indices
    for i in torso_segments:
        cooling_temps[i] = 20    # 20°C cooling pads
        cooling_areas[i] = 0.7   # 70% coverage
    
    # Apply cooling settings
    model.material_temp = cooling_temps
    model.contact_area = cooling_areas
    model.contact_resistance = 0.01  # Good thermal contact
    
    print("Cooling applied to torso:")
    print(f"- Material temperature: 20°C")
    print(f"- Contact area: 70% of torso segments")
    print(f"- Thermal resistance: 0.01 K·m²/W")
    
    # Run cooling simulation
    model.simulate(times=20)  # 20 minutes of cooling
    
    # === Phase 3: Results Analysis ===
    print("\nPhase 3: Analyzing results...")
    
    results = model.dict_results()
    
    # Get temperature data
    time_points = np.array(results['CycleTime']) / 60  # Convert to minutes
    skin_temps = np.array(results['TskMean'])
    
    # Calculate temperature reduction
    final_temp = skin_temps[-1]
    temp_reduction = baseline_temp - final_temp
    
    print(f"Final mean skin temperature: {final_temp:.2f}°C")
    print(f"Temperature reduction: {temp_reduction:.2f}°C")
    
    # Calculate cooling power
    # Get conductive heat for torso segments at final time
    cooling_power = 0
    segment_names = ['Head', 'Neck', 'Chest', 'Back', 'Pelvis', 'LShoulder', 'LArm', 'LHand',
                    'RShoulder', 'RArm', 'RHand', 'LThigh', 'LLeg', 'LFoot', 'RThigh', 'RLeg', 'RFoot']
    
    print("\nCooling power by segment:")
    for i, segment in enumerate(segment_names):
        qcond = results[f'Qcond{segment}'][-1]  # Latest value
        if i in torso_segments:  # Only show cooled segments
            cooling_power += abs(qcond)  # Add absolute value for total cooling
            print(f"  {segment}: {qcond:.1f}W")
    
    print(f"\nTotal cooling power: {cooling_power:.1f}W")
    
    # === Phase 4: Create Visualization ===
    print("\nPhase 4: Creating visualization...")
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Temperature over time
    plt.subplot(2, 2, 1)
    plt.plot(time_points, skin_temps, 'b-', linewidth=2)
    plt.axvline(x=10, color='r', linestyle='--', alpha=0.7, label='Cooling applied')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Mean Skin Temperature (°C)')
    plt.title('Skin Temperature Response')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Cooling power over time
    plt.subplot(2, 2, 2)
    cooling_history = []
    for i in range(len(time_points)):
        if i < 11:  # Before cooling
            cooling_history.append(0)
        else:  # During cooling
            total_cooling = 0
            for seg_idx in torso_segments:
                segment = segment_names[seg_idx]
                total_cooling += abs(results[f'Qcond{segment}'][i])
            cooling_history.append(total_cooling)
    
    plt.plot(time_points, cooling_history, 'g-', linewidth=2)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Cooling Power (W)')
    plt.title('Cooling Power Over Time')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Body segment temperatures (final state)
    plt.subplot(2, 2, 3)
    segment_temps = [results[f'Tsk{seg}'][-1] for seg in segment_names]
    colors = ['red' if i in torso_segments else 'blue' for i in range(17)]
    
    plt.bar(range(17), segment_temps, color=colors, alpha=0.7)
    plt.xlabel('Body Segment')
    plt.ylabel('Skin Temperature (°C)')
    plt.title('Final Skin Temperatures by Segment')
    plt.xticks(range(17), [seg[:4] for seg in segment_names], rotation=45)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.7, label='Cooled segments'),
                      Patch(facecolor='blue', alpha=0.7, label='Non-cooled segments')]
    plt.legend(handles=legend_elements)
    
    # Plot 4: Cooling effectiveness
    plt.subplot(2, 2, 4)
    effectiveness_data = {
        'Baseline': baseline_temp,
        'With Cooling': final_temp,
        'Reduction': temp_reduction
    }
    
    bars = plt.bar(effectiveness_data.keys(), effectiveness_data.values(), 
                   color=['orange', 'lightblue', 'green'])
    plt.ylabel('Temperature (°C)')
    plt.title('Cooling Effectiveness')
    
    # Add value labels on bars
    for bar, value in zip(bars, effectiveness_data.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.1f}°C', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('cooling_therapy_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # === Summary ===
    print("\n" + "="*50)
    print("COOLING THERAPY SIMULATION COMPLETE")
    print("="*50)
    print(f"✓ Baseline temperature: {baseline_temp:.2f}°C")
    print(f"✓ Final temperature: {final_temp:.2f}°C")
    print(f"✓ Temperature reduction: {temp_reduction:.2f}°C")
    print(f"✓ Total cooling power: {cooling_power:.1f}W")
    print(f"✓ Cooling efficiency: {temp_reduction/cooling_power*1000:.1f}°C per kW")
    print("✓ Results saved to 'cooling_therapy_results.png'")
    
    if temp_reduction > 1.0:
        print("\n🎉 Cooling therapy was EFFECTIVE!")
    else:
        print("\n⚠️  Cooling therapy had minimal effect - consider adjusting parameters")

if __name__ == "__main__":
    cooling_therapy_example()