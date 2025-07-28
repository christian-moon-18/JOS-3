#!/usr/bin/env python3
"""
Simple test to verify conductive heat transfer is working
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import jos3

def simple_test():
    model = jos3.JOS3(ex_output="all")
    
    # Set environmental conditions
    model.Ta = 25
    model.RH = 50
    model.Va = 0.1
    model.PAR = 1.25
    
    # Set up cooling
    model.material_temp = 10  # 10°C cooling
    model.contact_area = 0.5  # 50% contact
    model.contact_resistance = 0.01  # Good contact
    
    # Run simulation
    model.simulate(times=2)  # Run 2 steps
    
    # Get results
    results = model.dict_results()
    
    print(f"History length: {len(results['CycleTime'])}")
    
    # Get latest conductive heat for head segment  
    qcond_head = results['QcondHead'][-1]
    print(f"Head conductive heat: {qcond_head:.2f}W")
    
    # Check if it's working
    if abs(qcond_head) > 10:  # Should be significant cooling
        print("✓ Conductive heat transfer is working!")
        return True
    else:
        print("❌ Conductive heat transfer not working")
        return False

if __name__ == "__main__":
    simple_test()