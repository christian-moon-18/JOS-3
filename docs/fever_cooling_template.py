#!/usr/bin/env python3
"""
JOS-3 Conductive Heat Transfer Template: Fever Cooling Simulation

This template demonstrates the complete workflow for simulating fever management
using conductive cooling therapy with the enhanced JOS-3 model.

Features demonstrated:
- Patient model initialization with anthropometric data
- Multi-phase simulation (baseline → fever → cooling therapy)
- Proper conductive heat transfer parameter setup
- Comprehensive results analysis and visualization
- Energy conservation validation

Usage:
    python fever_cooling_template.py

Author: Enhanced JOS-3 Development Team
Version: 1.0
Date: January 2025
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jos3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# =============================================================================
# CONFIGURATION SECTION - Modify these parameters as needed
# =============================================================================

# Patient characteristics
PATIENT_CONFIG = {
    'height': 1.75,          # meters (5'9")
    'weight': 70,            # kg (154 lbs)
    'age': 35,               # years
    'sex': 'male',           # 'male' or 'female'
    'fat': 15,               # body fat percentage
    'ci': 2.6432,            # cardiac index [L/min/m²]
    'bmr_equation': 'harris-benedict',  # or 'japanese'
    'bsa_equation': 'dubois'            # or 'fujimoto', 'kruazumi', 'takahira'
}

# Environmental conditions
ENVIRONMENT_CONFIG = {
    'ambient_temp': 24,      # °C
    'relative_humidity': 50, # %
    'air_velocity': 0.1,     # m/s
    'posture': 'lying'       # 'standing', 'sitting', 'lying'
}

# Simulation phases (time in minutes)
SIMULATION_PHASES = {
    'baseline_duration': 60,    # 1 hour baseline
    'fever_duration': 120,      # 2 hours fever development
    'cooling_duration': 240     # 4 hours cooling therapy
}

# Fever parameters
FEVER_CONFIG = {
    'core_setpoint': 38.8,      # °C - target fever temperature
    'skin_setpoint': 37.2,      # °C - elevated skin temperature
    'metabolic_rate': 1.8       # multiplier (1.0 = normal, >1.0 = elevated)
}

# Cooling device configuration
COOLING_CONFIG = {
    'target_segment': 3,        # 3 = Back (see body segment mapping below)
    'material_temp': 4.0,       # °C - cooling pad temperature
    'contact_area': 0.8,        # fraction (0.0-1.0) - 80% coverage
    'contact_resistance': 0.02, # K⋅m²/W - thermal resistance
    'device_type': 'cooling_pad' # for documentation
}

# Body segment mapping for reference
BODY_SEGMENTS = {
    0: 'Head', 1: 'Neck', 2: 'Chest', 3: 'Back', 4: 'Pelvis',
    5: 'L_Shoulder', 6: 'L_Arm', 7: 'R_Arm', 8: 'L_Forearm', 9: 'R_Forearm',
    10: 'L_Hand', 11: 'R_Hand', 12: 'R_Hand', 13: 'L_Thigh', 14: 'R_Thigh',
    15: 'L_Leg', 16: 'R_Leg'
}

# =============================================================================
# SIMULATION IMPLEMENTATION
# =============================================================================

def initialize_patient_model():
    """Initialize JOS-3 model with patient characteristics"""
    print("🏥 Initializing patient model...")
    
    model = jos3.JOS3(
        height=PATIENT_CONFIG['height'],
        weight=PATIENT_CONFIG['weight'],
        age=PATIENT_CONFIG['age'],
        sex=PATIENT_CONFIG['sex'],
        fat=PATIENT_CONFIG['fat'],
        ci=PATIENT_CONFIG['ci'],
        bmr_equation=PATIENT_CONFIG['bmr_equation'],
        bsa_equation=PATIENT_CONFIG['bsa_equation'],
        ex_output="all"  # Export all parameters for analysis
    )
    
    # Set environmental conditions
    model.Ta = ENVIRONMENT_CONFIG['ambient_temp']
    model.RH = ENVIRONMENT_CONFIG['relative_humidity']
    model.Va = ENVIRONMENT_CONFIG['air_velocity']
    model.posture = ENVIRONMENT_CONFIG['posture']
    
    print(f"   Patient: {PATIENT_CONFIG['height']:.2f}m, {PATIENT_CONFIG['weight']:.1f}kg, {PATIENT_CONFIG['age']}yo {PATIENT_CONFIG['sex']}")
    print(f"   Environment: {ENVIRONMENT_CONFIG['ambient_temp']}°C, {ENVIRONMENT_CONFIG['relative_humidity']}% RH")
    
    return model

def simulate_baseline_phase(model):
    """Simulate baseline physiological state"""
    print(f"📊 Phase 1: Baseline simulation ({SIMULATION_PHASES['baseline_duration']} minutes)...")
    
    # Run baseline simulation to establish normal thermoregulation
    model.simulate(times=SIMULATION_PHASES['baseline_duration'], dtime=60)
    
    # Record baseline metrics
    baseline_tcb = model.Tcb  # Core blood pool temperature (scalar)
    baseline_tsk = np.mean(model.Tsk)  # Average skin temperature
    
    print(f"   Baseline core temperature: {baseline_tcb:.2f}°C")
    print(f"   Baseline average skin temperature: {baseline_tsk:.2f}°C")
    
    return {'tcb': baseline_tcb, 'tsk': baseline_tsk}

def simulate_fever_phase(model):
    """Simulate fever development"""
    print(f"🔥 Phase 2: Fever simulation ({SIMULATION_PHASES['fever_duration']} minutes)...")
    
    # Set fever parameters - elevated setpoints and metabolism
    model.setpt_cr = [FEVER_CONFIG['core_setpoint']] * 17
    model.setpt_sk = [FEVER_CONFIG['skin_setpoint']] * 17
    model.PAR = FEVER_CONFIG['metabolic_rate']
    
    print(f"   Target core setpoint: {FEVER_CONFIG['core_setpoint']}°C")
    print(f"   Metabolic rate multiplier: {FEVER_CONFIG['metabolic_rate']}x")
    
    # Simulate fever development
    model.simulate(times=SIMULATION_PHASES['fever_duration'], dtime=60)
    
    # Record peak fever metrics 
    fever_tcb = model.Tcb  # Scalar value
    fever_tsk = np.mean(model.Tsk)
    
    print(f"   Peak fever core temperature: {fever_tcb:.2f}°C")
    print(f"   Peak fever skin temperature: {fever_tsk:.2f}°C")
    
    return {'tcb': fever_tcb, 'tsk': fever_tsk}

def apply_cooling_therapy(model):
    """Apply conductive cooling therapy"""
    print(f"❄️  Phase 3: Cooling therapy ({SIMULATION_PHASES['cooling_duration']} minutes)...")
    
    target_segment = COOLING_CONFIG['target_segment']
    segment_name = BODY_SEGMENTS.get(target_segment, f"Segment_{target_segment}")
    
    print(f"   Device: {COOLING_CONFIG['device_type']}")
    print(f"   Target: {segment_name} (segment {target_segment})")
    print(f"   Temperature: {COOLING_CONFIG['material_temp']}°C")
    print(f"   Coverage: {COOLING_CONFIG['contact_area']*100:.0f}%")
    print(f"   Thermal resistance: {COOLING_CONFIG['contact_resistance']} K⋅m²/W")
    
    # Set material temperature (NaN for no contact)
    model.material_temp = [float('nan')] * 17
    model.material_temp[target_segment] = COOLING_CONFIG['material_temp']
    
    # Set contact area - CRITICAL: Use proper array assignment method
    contact_areas = [0.0] * 17
    contact_areas[target_segment] = COOLING_CONFIG['contact_area']
    model.contact_area = contact_areas
    
    # Set thermal resistance
    model.contact_resistance[target_segment] = COOLING_CONFIG['contact_resistance']
    
    # Validate setup
    print("   Validating cooling setup...")
    validate_cooling_setup(model, target_segment)
    
    # Simulate cooling therapy
    model.simulate(times=SIMULATION_PHASES['cooling_duration'], dtime=60)
    
    # Analyze cooling effectiveness
    return analyze_cooling_effectiveness(model, target_segment)

def validate_cooling_setup(model, segment_idx):
    """Validate that cooling parameters are set correctly"""
    material_temp = model._material_temp[segment_idx]
    contact_area = model._contact_area[segment_idx]
    contact_resistance = model._contact_resistance[segment_idx]
    
    if np.isnan(material_temp):
        print("   ❌ ERROR: Material temperature is NaN")
        return False
    
    if contact_area <= 0:
        print("   ❌ ERROR: Contact area is zero or negative")
        return False
        
    if contact_resistance <= 0:
        print("   ❌ ERROR: Contact resistance is zero or negative") 
        return False
    
    print(f"   ✅ Setup validated: T={material_temp}°C, A={contact_area:.2f}, R={contact_resistance}")
    return True

def analyze_cooling_effectiveness(model, segment_idx):
    """Analyze the effectiveness of cooling therapy"""
    results = model.dict_results()
    
    # Get segment name for results keys
    segment_name = BODY_SEGMENTS[segment_idx].replace('_', '')
    qcond_key = f"Qcond{segment_name}"
    
    # Extract cooling power data
    if qcond_key in results:
        cooling_power = [q for q in results[qcond_key] if not np.isnan(q)]
        if cooling_power:
            avg_cooling_power = np.mean(cooling_power[-60:])  # Last hour average
            total_cooling_energy = np.sum(cooling_power) * 60 / 1000  # Convert to kJ
            
            print(f"   Average cooling power: {avg_cooling_power:.1f} W")
            print(f"   Total cooling energy: {total_cooling_energy:.1f} kJ")
        else:
            print("   ❌ WARNING: No cooling power detected")
            avg_cooling_power = 0
            total_cooling_energy = 0
    else:
        print(f"   ❌ ERROR: Could not find {qcond_key} in results")
        avg_cooling_power = 0
        total_cooling_energy = 0
    
    return {
        'avg_power': avg_cooling_power,
        'total_energy': total_cooling_energy
    }

def create_visualization(model):
    """Create comprehensive visualization of simulation results"""
    print("📈 Creating visualization...")
    
    results = model.dict_results()
    time_minutes = [t.total_seconds() / 60 for t in results["ModTime"]]
    
    # Calculate phase boundaries
    baseline_end = SIMULATION_PHASES['baseline_duration']
    fever_end = baseline_end + SIMULATION_PHASES['fever_duration']
    cooling_end = fever_end + SIMULATION_PHASES['cooling_duration']
    
    # Create 4-panel plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Panel 1: Core blood pool temperature
    ax1.plot(time_minutes, results["Tcb"], 'r-', linewidth=2.5, label="Core Blood Pool (Tcb)")
    ax1.axvline(x=baseline_end, color='orange', linestyle='--', alpha=0.7, label="Fever onset")
    ax1.axvline(x=fever_end, color='blue', linestyle='--', alpha=0.7, label="Cooling start")
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title("Core Blood Pool Temperature")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Target segment core temperature
    target_segment = COOLING_CONFIG['target_segment']
    segment_name = BODY_SEGMENTS[target_segment].replace('_', '')
    tcr_key = f"Tcr{segment_name}"
    
    if tcr_key in results:
        ax2.plot(time_minutes, results[tcr_key], 'orange', linewidth=2.5, label=f"{segment_name} Core")
        ax2.axvline(x=baseline_end, color='orange', linestyle='--', alpha=0.7)
        ax2.axvline(x=fever_end, color='blue', linestyle='--', alpha=0.7)
        ax2.set_ylabel("Temperature (°C)")
        ax2.set_title(f"{segment_name} Core Temperature")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Panel 3: Target segment skin temperature
    tsk_key = f"Tsk{segment_name}"
    if tsk_key in results:
        ax3.plot(time_minutes, results[tsk_key], 'green', linewidth=2.5, label=f"{segment_name} Skin")
        ax3.axvline(x=baseline_end, color='orange', linestyle='--', alpha=0.7)
        ax3.axvline(x=fever_end, color='blue', linestyle='--', alpha=0.7)
        ax3.set_ylabel("Temperature (°C)")
        ax3.set_title(f"{segment_name} Skin Temperature")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Panel 4: Conductive heat transfer
    qcond_key = f"Qcond{segment_name}"
    if qcond_key in results:
        ax4.plot(time_minutes, results[qcond_key], 'purple', linewidth=2.5, label="Cooling Power")
        ax4.axvline(x=baseline_end, color='orange', linestyle='--', alpha=0.7)
        ax4.axvline(x=fever_end, color='blue', linestyle='--', alpha=0.7)
        ax4.set_ylabel("Heat Transfer (W)")
        ax4.set_title("Conductive Heat Transfer")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Set x-axis labels for all panels
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel("Time (minutes)")
    
    plt.tight_layout()
    
    # Save plot with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"fever_cooling_simulation_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"   Plot saved: {plot_filename}")
    
    plt.show()
    
    return plot_filename

def export_results(model, baseline_metrics, fever_metrics, cooling_metrics):
    """Export comprehensive results to CSV and generate summary report"""
    print("💾 Exporting results...")
    
    # Export full simulation data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"fever_cooling_results_{timestamp}.csv"
    model.to_csv(csv_filename)
    print(f"   Full results: {csv_filename}")
    
    # Generate summary report
    results = model.dict_results()
    
    # Calculate key metrics
    baseline_tcb = baseline_metrics['tcb']
    fever_tcb = fever_metrics['tcb']
    final_tcb = results["Tcb"][-1]
    
    fever_rise = fever_tcb - baseline_tcb
    cooling_reduction = fever_tcb - final_tcb
    cooling_effectiveness = (cooling_reduction / fever_rise * 100) if fever_rise > 0 else 0
    
    # Create summary report
    report_filename = f"fever_cooling_summary_{timestamp}.txt"
    with open(report_filename, 'w') as f:
        f.write("JOS-3 FEVER COOLING SIMULATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Simulation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("PATIENT CONFIGURATION:\n")
        for key, value in PATIENT_CONFIG.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("COOLING DEVICE CONFIGURATION:\n")
        for key, value in COOLING_CONFIG.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("SIMULATION RESULTS:\n")
        f.write(f"  Baseline core temperature: {baseline_tcb:.2f}°C\n")
        f.write(f"  Peak fever temperature: {fever_tcb:.2f}°C\n")
        f.write(f"  Final core temperature: {final_tcb:.2f}°C\n")
        f.write(f"  Fever rise: {fever_rise:.2f}°C\n")
        f.write(f"  Cooling reduction: {cooling_reduction:.2f}°C\n")
        f.write(f"  Cooling effectiveness: {cooling_effectiveness:.1f}%\n\n")
        
        f.write("COOLING PERFORMANCE:\n")
        f.write(f"  Average cooling power: {cooling_metrics['avg_power']:.1f} W\n")
        f.write(f"  Total cooling energy: {cooling_metrics['total_energy']:.1f} kJ\n")
        
        # Energy analysis
        total_duration = sum(SIMULATION_PHASES.values()) / 60  # hours
        # Use BMR as metabolic power estimate if M not available
        if "M" in results:
            avg_metabolic_power = np.mean(results["M"])
        else:
            avg_metabolic_power = model.BMR  # Use base metabolic rate
        total_metabolic_energy = avg_metabolic_power * total_duration * 3.6  # kJ
        cooling_fraction = abs(cooling_metrics['total_energy']) / total_metabolic_energy * 100
        
        f.write(f"  Metabolic energy (total): {total_metabolic_energy:.1f} kJ\n")
        f.write(f"  Cooling energy fraction: {cooling_fraction:.1f}%\n")
    
    print(f"   Summary report: {report_filename}")
    
    return csv_filename, report_filename

def main():
    """Main simulation workflow"""
    print("🏥 JOS-3 CONDUCTIVE HEAT TRANSFER SIMULATION")
    print("=" * 60)
    print(f"Simulation: Fever management with {COOLING_CONFIG['device_type']}")
    print(f"Target: {BODY_SEGMENTS[COOLING_CONFIG['target_segment']]}")
    print("=" * 60)
    
    try:
        # Initialize model
        model = initialize_patient_model()
        
        # Phase 1: Baseline
        baseline_metrics = simulate_baseline_phase(model)
        
        # Phase 2: Fever development
        fever_metrics = simulate_fever_phase(model)
        
        # Phase 3: Cooling therapy
        cooling_metrics = apply_cooling_therapy(model)
        
        # Analysis and visualization
        plot_file = create_visualization(model)
        csv_file, report_file = export_results(model, baseline_metrics, fever_metrics, cooling_metrics)
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎯 SIMULATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        results = model.dict_results()
        baseline_tcb = baseline_metrics['tcb']
        fever_tcb = fever_metrics['tcb']
        final_tcb = results["Tcb"][-1]
        
        fever_rise = fever_tcb - baseline_tcb
        cooling_reduction = fever_tcb - final_tcb
        effectiveness = (cooling_reduction / fever_rise * 100) if fever_rise > 0 else 0
        
        print(f"Fever rise: {fever_rise:.2f}°C")
        print(f"Cooling reduction: {cooling_reduction:.2f}°C")
        print(f"Effectiveness: {effectiveness:.1f}% fever reduction")
        print(f"Average cooling power: {cooling_metrics['avg_power']:.1f} W")
        print("\nGenerated files:")
        print(f"  • {plot_file}")
        print(f"  • {csv_file}")
        print(f"  • {report_file}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ SIMULATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)