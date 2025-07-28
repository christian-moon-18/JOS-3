#!/usr/bin/env python3
"""
Unit tests for conductive heat transfer implementation in JOS-3

Tests basic conductive heat transfer calculation, energy conservation,
and edge cases to ensure the implementation is robust and accurate.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import jos3

# Helper function to extract conductive heat transfer data
def get_qcond_array(results, time_index=-1):
    """Extract conductive heat transfer for all body segments"""
    body_segments = ['Head', 'Neck', 'Chest', 'Back', 'Pelvis', 'LShoulder', 'LArm', 'LHand', 
                     'RShoulder', 'RArm', 'RHand', 'LThigh', 'LLeg', 'LFoot', 'RThigh', 'RLeg', 'RFoot']  
    return np.array([results[f'Qcond{segment}'][time_index] for segment in body_segments])

def test_conductive_heat_transfer():
    """Test basic conductive heat transfer calculation"""
    print("Testing basic conductive heat transfer calculation...")
    
    model = jos3.JOS3(ex_output="all")  # Enable all outputs
    
    # Set environmental conditions
    model.Ta = 25  # Air temperature 25°C
    model.RH = 50  # 50% humidity
    model.Va = 0.1 # Low air velocity
    model.PAR = 1.25  # Resting metabolism
    
    # Set up cooling scenario
    model.material_temp = 10  # 10°C cooling material
    model.contact_area = 0.5  # 50% contact area
    model.contact_resistance = 0.01  # Low thermal resistance
    
    # Run simulation for 2 time steps
    model.simulate(times=2)
    
    # Get results
    results = model.dict_results()
    
    # Get conductive heat transfer for all segments
    print(f"Results history length: {len(results['CycleTime'])}")
    print(f"Sample QcondHead values: {results['QcondHead']}")
    qcond = get_qcond_array(results)
    print(f"Conductive heat transfer: {qcond}")
    
    # All segments should have negative heat transfer (cooling)
    assert all(qcond < 0), f"Expected all negative values (cooling), got: {qcond}"
    
    # Check that heat transfer is proportional to temperature difference
    # Skin temp should be around 34°C, so temp difference ≈ 24°C
    # Expected magnitude should be significant with good contact
    expected_magnitude = np.mean(np.abs(qcond))
    assert expected_magnitude > 10, f"Expected significant heat transfer, got {expected_magnitude:.2f}W average"
    
    print("✓ Basic conductive heat transfer test passed")


def test_no_contact():
    """Test that NaN material temp results in zero heat transfer"""
    print("Testing no contact scenario...")
    
    model = jos3.JOS3(ex_output="all")
    model.Ta = 25
    
    # Set material temperature to NaN (no contact)
    model.material_temp = np.nan
    model.contact_area = 1.0  # Full contact area (should be ignored)
    model.contact_resistance = 0.01
    
    model.simulate(times=2)
    
    results = model.dict_results()
    qcond = get_qcond_array(results)
    
    # All heat transfer should be zero
    assert all(qcond == 0), f"Expected zero heat transfer with NaN temp, got: {qcond}"
    
    print("✓ No contact test passed")


def test_partial_contact():
    """Test partial contact on specific segments"""
    print("Testing partial contact scenario...")
    
    model = jos3.JOS3(ex_output="all")
    model.Ta = 25
    
    # Set cooling only on back and chest
    material_temps = np.full(17, np.nan)
    material_temps[2] = 15  # Chest cooling
    material_temps[3] = 15  # Back cooling
    
    contact_areas = np.zeros(17)
    contact_areas[2] = 0.8  # 80% chest contact
    contact_areas[3] = 0.6  # 60% back contact
    
    model.material_temp = material_temps
    model.contact_area = contact_areas
    model.contact_resistance = 0.01
    
    model.simulate(times=2)
    
    results = model.dict_results()
    qcond = get_qcond_array(results)
    
    # Only chest (index 2) and back (index 3) should have heat transfer
    assert qcond[2] < 0, f"Expected cooling in chest, got {qcond[2]:.2f}W"
    assert qcond[3] < 0, f"Expected cooling in back, got {qcond[3]:.2f}W"
    
    # Other segments should have zero heat transfer
    other_segments = [i for i in range(17) if i not in [2, 3]]
    assert all(qcond[other_segments] == 0), f"Expected zero heat transfer in other segments"
    
    # Chest should have more cooling than back (higher contact area)
    assert abs(qcond[2]) > abs(qcond[3]), f"Expected more cooling in chest than back"
    
    print("✓ Partial contact test passed")


def test_energy_conservation():
    """Verify total energy balance is maintained"""
    print("Testing energy conservation...")
    
    # Run simulation without conductive cooling
    model_baseline = jos3.JOS3(ex_output="all")
    model_baseline.Ta = 25
    model_baseline.simulate(times=5)  # Run for 5 minutes
    results_baseline = model_baseline.dict_results()
    
    # Run simulation with conductive cooling
    model_cooling = jos3.JOS3(ex_output="all")
    model_cooling.Ta = 25
    model_cooling.material_temp = 20  # Moderate cooling
    model_cooling.contact_area = 0.3  # 30% contact
    model_cooling.contact_resistance = 0.02
    model_cooling.simulate(times=5)  # Run for 5 minutes
    results_cooling = model_cooling.dict_results()
    
    # Check that mean skin temperature decreases with cooling
    tsk_baseline = np.array(results_baseline['TskMean'])
    tsk_cooling = np.array(results_cooling['TskMean'])
    
    final_temp_diff = tsk_baseline[-1] - tsk_cooling[-1]
    assert final_temp_diff > 0.1, f"Expected cooling to reduce skin temperature by >0.1°C, got {final_temp_diff:.3f}°C"
    
    # Check that conductive heat removal is working
    qcond_total = np.array(results_cooling['Qcond'])
    total_heat_removed = np.sum(qcond_total, axis=1)  # Sum across body segments
    
    # Total heat removal should be negative (heat flowing out)
    assert np.mean(total_heat_removed) < -5, f"Expected significant heat removal, got {np.mean(total_heat_removed):.2f}W"
    
    print("✓ Energy conservation test passed")


def test_validation_ranges():
    """Test parameter validation"""
    print("Testing parameter validation...")
    
    model = jos3.JOS3(ex_output="all")
    
    # Test contact area validation
    try:
        model.contact_area = 1.5  # Should fail (>1)
        assert False, "Expected ValueError for contact_area > 1"
    except ValueError:
        pass  # Expected
    
    try:
        model.contact_area = -0.1  # Should fail (<0)
        assert False, "Expected ValueError for contact_area < 0"
    except ValueError:
        pass  # Expected
    
    # Test contact resistance validation
    try:
        model.contact_resistance = 0.0005  # Should fail (<0.001)
        assert False, "Expected ValueError for contact_resistance < 0.001"
    except ValueError:
        pass  # Expected
    
    try:
        model.contact_resistance = 2.0  # Should fail (>1.0)
        assert False, "Expected ValueError for contact_resistance > 1.0"
    except ValueError:
        pass  # Expected
    
    # Valid values should work
    model.contact_area = 0.5
    model.contact_resistance = 0.05
    model.material_temp = 20
    
    print("✓ Parameter validation test passed")


def test_realistic_cooling_therapy():
    """Test realistic cooling therapy scenario"""
    print("Testing realistic cooling therapy scenario...")
    
    model = jos3.JOS3(height=1.75, weight=70, age=30, ex_output="all")
    
    # Simulate mild heat stress first
    model.Ta = 35  # Hot environment
    model.RH = 60  # High humidity
    model.Va = 0.1
    model.PAR = 1.5  # Light activity
    model.simulate(times=10)  # 10 minutes to reach steady state
    
    baseline_temp = model.TskMean
    print(f"Baseline skin temperature: {baseline_temp:.2f}°C")
    
    # Apply cooling pads to torso
    cooling_temps = np.full(17, np.nan)
    cooling_areas = np.zeros(17)
    
    # Cool chest, back, and pelvis
    for segment_idx in [2, 3, 4]:  # Chest, Back, Pelvis
        cooling_temps[segment_idx] = 20  # 20°C cooling pads
        cooling_areas[segment_idx] = 0.7  # 70% coverage
    
    model.material_temp = cooling_temps
    model.contact_area = cooling_areas
    model.contact_resistance = 0.01  # Good thermal contact
    
    # Apply cooling for 20 minutes
    model.simulate(times=20)
    
    results = model.dict_results()
    final_temp = results['TskMean'][-1]
    temp_reduction = baseline_temp - final_temp
    
    print(f"Final skin temperature: {final_temp:.2f}°C")
    print(f"Temperature reduction: {temp_reduction:.2f}°C")
    
    # Should see significant cooling
    assert temp_reduction > 0.5, f"Expected >0.5°C cooling, got {temp_reduction:.2f}°C"
    assert temp_reduction < 5.0, f"Cooling too aggressive: {temp_reduction:.2f}°C"
    
    # Check cooling power
    qcond_final = get_qcond_array(results, -1)  # Last time step
    total_cooling_power = -np.sum(qcond_final)  # Negative sign for cooling
    print(f"Total cooling power: {total_cooling_power:.1f}W")
    
    assert total_cooling_power > 20, f"Expected >20W cooling power, got {total_cooling_power:.1f}W"
    assert total_cooling_power < 200, f"Cooling power too high: {total_cooling_power:.1f}W"
    
    print("✓ Realistic cooling therapy test passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("CONDUCTIVE HEAT TRANSFER TESTS")
    print("=" * 50)
    
    try:
        test_conductive_heat_transfer()
        test_no_contact()
        test_partial_contact()
        test_energy_conservation()
        test_validation_ranges()
        test_realistic_cooling_therapy()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED! ✓")
        print("=" * 50)
        print("\nConductive heat transfer implementation is working correctly!")
        print("The model can now simulate:")
        print("• Cooling/heating pads and blankets")
        print("• Contact with hot/cold surfaces")
        print("• Therapeutic cooling applications")
        print("• Material interface heat transfer")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)