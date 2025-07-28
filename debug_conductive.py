#!/usr/bin/env python3
"""
Debug script for conductive heat transfer implementation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import jos3

def debug_conductive():
    print("=== DEBUGGING CONDUCTIVE HEAT TRANSFER ===")
    
    model = jos3.JOS3(ex_output="all")
    
    # Set simple conditions
    model.Ta = 25  # Air temperature 25°C
    model.RH = 50  # 50% humidity
    model.Va = 0.1 # Low air velocity
    model.PAR = 1.25  # Resting metabolism
    
    print(f"Initial material_temp: {model.material_temp}")
    print(f"Initial contact_area: {model.contact_area}")
    print(f"Initial contact_resistance: {model.contact_resistance}")
    
    # Set up cooling scenario
    model.material_temp = 10  # 10°C cooling material
    model.contact_area = 0.5  # 50% contact area
    model.contact_resistance = 0.01  # Low thermal resistance
    
    print(f"After setting:")
    print(f"material_temp: {model.material_temp}")
    print(f"contact_area: {model.contact_area}")
    print(f"contact_resistance: {model.contact_resistance}")
    
    # Check initial skin temperature
    print(f"Initial skin temperatures: {model.Tsk}")
    
    # Check that values persist
    print(f"Just before simulate:")
    print(f"material_temp: {model.material_temp}")
    print(f"contact_area: {model.contact_area}")
    
    # Run simulation for 1 time step
    model.simulate(times=1)
    
    # Check if values were reset during simulate
    print(f"After simulate:")
    print(f"model._material_temp: {model._material_temp}")
    print(f"model._contact_area: {model._contact_area}")
    
    # Let's check what the conductive heat calculation should be manually
    tsk_after = model.Tsk
    print(f"Skin temperatures after: {tsk_after}")
    
    expected_cond = ((model._material_temp - tsk_after) * 
                     model.BSA * model._contact_area / 
                     model._contact_resistance)
    print(f"Expected conductive heat (manual calc): {expected_cond}")
    
    # Get results
    results = model.dict_results()
    
    # Get skin temperatures after simulation
    body_segments = ['Head', 'Neck', 'Chest', 'Back', 'Pelvis', 'LShoulder', 'LArm', 'LHand', 
                     'RShoulder', 'RArm', 'RHand', 'LThigh', 'LLeg', 'LFoot', 'RThigh', 'RLeg', 'RFoot']
    
    print(f"\nResults history length: {len(results['CycleTime'])}")
    print("After simulation (LATEST STEP):")
    for i, segment in enumerate(body_segments):
        tsk = results[f'Tsk{segment}'][-1]  # Use -1 for latest step
        qcond = results[f'Qcond{segment}'][-1]  # Use -1 for latest step
        material_temp = results[f'MaterialTemp{segment}'][-1]
        contact_area = results[f'ContactArea{segment}'][-1]
        contact_resistance = results[f'ContactResistance{segment}'][-1]
        
        expected_qcond = (material_temp - tsk) * model.BSA[i] * contact_area / contact_resistance
        
        print(f"{segment:12}: Tsk={tsk:5.1f}°C, Qcond={qcond:6.2f}W, Expected={expected_qcond:6.2f}W")
        print(f"             MaterialTemp={material_temp:5.1f}°C, Area={contact_area:.2f}, R={contact_resistance:.3f}")
        
        if i >= 5:  # Just show first few for brevity
            break
    
    print(f"\nTotal conductive heat: {sum(results[f'Qcond{segment}'][0] for segment in body_segments):.2f}W")

if __name__ == "__main__":
    debug_conductive()