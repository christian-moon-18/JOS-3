# JOS-3 Conductive Heat Transfer - Comprehensive Guide

## Overview

This document provides comprehensive documentation for the conductive heat transfer feature added to JOS-3, enabling simulation of external cooling/heating devices such as therapeutic cooling pads, heating blankets, and contact surfaces.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Technical Implementation](#technical-implementation)
3. [Parameter Reference](#parameter-reference)
4. [Usage Patterns](#usage-patterns)
5. [Validation and Testing](#validation-and-testing)
6. [Troubleshooting](#troubleshooting)
7. [API Reference](#api-reference)

---

## Quick Start

### Basic Usage

```python
import jos3
import numpy as np

# Initialize enhanced JOS-3 model
model = jos3.JOS3(ex_output="all")

# Set baseline conditions
model.Ta = 24  # Ambient temperature
model.simulate(times=60, dtime=60)  # 1 hour baseline

# Apply cooling pad to back segment
model.material_temp = [float('nan')] * 17
model.material_temp[3] = 4.0  # 4°C cooling pad

# Set contact parameters - IMPORTANT: Use array assignment
contact_areas = [0.0] * 17
contact_areas[3] = 0.7  # 70% coverage of back
model.contact_area = contact_areas

model.contact_resistance[3] = 0.02  # Low thermal resistance

# Simulate cooling therapy
model.simulate(times=120, dtime=60)  # 2 hours cooling

# Extract results
results = model.dict_results()
cooling_power = results["QcondBack"]  # Watts
back_skin_temp = results["TskBack"]   # °C
back_core_temp = results["TcrBack"]   # °C
```

### Key Requirements

⚠️ **CRITICAL:** Always use proper array assignment for `contact_area`:
```python
# ❌ WRONG - This doesn't work
model.contact_area = [0] * 17
model.contact_area[3] = 0.7  # Property indexing fails

# ✅ CORRECT - Modify array then assign
areas = [0] * 17
areas[3] = 0.7
model.contact_area = areas
```

---

## Technical Implementation

### Heat Transfer Equation

The conductive heat transfer is calculated using:

```
Q = (T_material - T_skin) × BSA × contact_area / contact_resistance
```

Where:
- `Q`: Heat transfer rate [W] (negative = cooling, positive = heating)
- `T_material`: Temperature of external material [°C]
- `T_skin`: Skin temperature [°C] 
- `BSA`: Body surface area of segment [m²]
- `contact_area`: Fraction of segment in contact [0-1]
- `contact_resistance`: Thermal resistance [K⋅m²/W]

### Integration with Heat Balance

The conductive heat transfer is integrated into JOS-3's heat balance at the skin layer:

```python
arrQ[INDEX["skin"]] += cond_ht
```

This means:
- **Cooling devices** (T_material < T_skin) remove heat from skin
- **Heating devices** (T_material > T_skin) add heat to skin
- Core temperature changes through thermoregulatory response

### Body Segments

JOS-3 divides the body into 17 segments:

| Index | Segment | Index | Segment | Index | Segment |
|-------|---------|-------|---------|-------|---------|
| 0 | Head | 6 | L_Arm | 12 | R_Hand |
| 1 | Neck | 7 | R_Arm | 13 | L_Thigh |
| 2 | Chest | 8 | L_Forearm | 14 | R_Thigh |
| 3 | **Back** | 9 | R_Forearm | 15 | L_Leg |
| 4 | Pelvis | 10 | L_Hand | 16 | R_Leg |
| 5 | L_Shoulder | 11 | R_Hand | | |

**Common therapeutic targets:**
- Back (index 3): Large surface area, good for cooling pads
- Chest (index 2): Cardiac proximity
- Thighs (13,14): Large muscle mass

---

## Parameter Reference

### material_temp (Material Temperature)

**Type:** `numpy.ndarray(17)` or list  
**Units:** °C  
**Default:** `np.nan` (no contact)  
**Range:** Any temperature value  

```python
# Single cooling pad on back
model.material_temp = [float('nan')] * 17
model.material_temp[3] = 4.0  # 4°C ice pack

# Multiple devices
temps = [float('nan')] * 17
temps[2] = 12.0   # 12°C cooling vest on chest  
temps[3] = 4.0    # 4°C ice pack on back
temps[13] = 8.0   # 8°C cooling pad on left thigh
model.material_temp = temps
```

### contact_area (Contact Area Fraction)

**Type:** `numpy.ndarray(17)` or list  
**Units:** Dimensionless [0-1]  
**Default:** `0.0` (no contact)  
**Range:** 0.0 to 1.0  

```python
# CORRECT method - create array then assign
areas = [0.0] * 17
areas[3] = 0.7  # 70% of back surface
areas[2] = 0.5  # 50% of chest surface  
model.contact_area = areas

# Alternative - direct numpy array
import numpy as np
areas = np.zeros(17)
areas[3] = 0.8
model.contact_area = areas
```

**Typical Values:**
- Small cooling pad: 0.3-0.5
- Large cooling blanket: 0.7-0.9  
- Partial contact: 0.1-0.3
- Full coverage: 0.9-1.0

### contact_resistance (Thermal Resistance)

**Type:** `numpy.ndarray(17)` or scalar→array  
**Units:** K⋅m²/W  
**Default:** `0.01`  
**Range:** 0.001 to 1.0  

```python
# Set resistance for specific segments
model.contact_resistance[3] = 0.02   # Good contact pad
model.contact_resistance[2] = 0.05   # Insulated vest
model.contact_resistance[13] = 0.01  # Direct skin contact
```

**Typical Values:**
- Direct skin contact: 0.001-0.01
- Thin gel pad: 0.01-0.02  
- Thick insulation: 0.05-0.1
- Air gap/poor contact: 0.1-1.0

---

## Usage Patterns

### Pattern 1: Fever Management with Cooling

```python
import jos3
import numpy as np

# Initialize patient model
model = jos3.JOS3(
    height=1.75, weight=70, age=30, sex="male",
    ex_output="all"
)

# Baseline (1 hour)
model.Ta = 24
model.simulate(times=60, dtime=60)

# Simulate fever (2 hours) 
model.setpt_cr = [38.5] * 17  # High fever setpoint
model.setpt_sk = [37.0] * 17  # Elevated skin temp
model.PAR = 1.8               # Increased metabolism
model.simulate(times=120, dtime=60)

# Apply cooling therapy (4 hours)
model.material_temp = [float('nan')] * 17
model.material_temp[3] = 4.0  # Ice pack on back

areas = [0.0] * 17
areas[3] = 0.8  # Large cooling pad
model.contact_area = areas

model.contact_resistance[3] = 0.02  # Good thermal contact
model.simulate(times=240, dtime=60)

# Analyze results
results = model.dict_results()
cooling_power = np.mean([q for q in results["QcondBack"][-240:] if not np.isnan(q)])
temp_reduction = results["Tcb"][-240] - results["Tcb"][-1]

print(f"Average cooling power: {cooling_power:.1f} W")
print(f"Core temperature reduction: {temp_reduction:.2f}°C")
```

### Pattern 2: Hypothermia Prevention with Heating

```python
# Simulate cold exposure with heating blanket
model = jos3.JOS3(ex_output="all")

# Cold environment
model.Ta = 5    # 5°C ambient
model.Va = 2    # Wind
model.simulate(times=60, dtime=60)

# Apply heating blanket to torso
model.material_temp = [float('nan')] * 17
model.material_temp[2] = 40.0  # 40°C heating blanket on chest
model.material_temp[3] = 38.0  # 38°C heating pad on back

areas = [0.0] * 17
areas[2] = 0.9  # Full chest coverage
areas[3] = 0.9  # Full back coverage
model.contact_area = areas

model.contact_resistance[2] = 0.03  # Blanket resistance
model.contact_resistance[3] = 0.03
model.simulate(times=180, dtime=60)  # 3 hours heating
```

### Pattern 3: Multi-Zone Therapy

```python
# Selective cooling of multiple body regions
model = jos3.JOS3(ex_output="all")

# Apply cooling to head, chest, and thighs
temps = [float('nan')] * 17
temps[0] = 8.0   # Head cooling cap
temps[2] = 12.0  # Cooling vest
temps[13] = 6.0  # Left thigh pad
temps[14] = 6.0  # Right thigh pad
model.material_temp = temps

areas = [0.0] * 17
areas[0] = 0.6   # Partial head coverage
areas[2] = 0.8   # Vest coverage
areas[13] = 0.7  # Thigh pad
areas[14] = 0.7  # Thigh pad
model.contact_area = areas

# Different resistances for different devices
model.contact_resistance[0] = 0.04   # Insulated cap
model.contact_resistance[2] = 0.02   # Direct vest contact
model.contact_resistance[13] = 0.01  # Gel pad
model.contact_resistance[14] = 0.01  # Gel pad

model.simulate(times=120, dtime=60)
```

---

## Validation and Testing

### Energy Conservation Test

```python
def validate_energy_conservation(model, duration=60):
    """Verify that energy is conserved in the system"""
    
    # Run simulation
    model.simulate(times=duration, dtime=60)
    results = model.dict_results()
    
    # Calculate total energy flows
    total_conductive = sum([
        np.sum([q for q in results[f"Qcond{seg}"] if not np.isnan(q)])
        for seg in ["Head", "Neck", "Chest", "Back", "Pelvis", 
                   "LShoulder", "RShoulder", "LArm", "RArm",
                   "LForearm", "RForearm", "LHand", "RHand",
                   "LThigh", "RThigh", "LLeg", "RLeg"]
    ])
    
    metabolic_heat = np.mean(results["M"]) * duration * 60  # Convert to J
    
    print(f"Total conductive heat transfer: {total_conductive/1000:.2f} kJ")
    print(f"Average metabolic heat: {metabolic_heat/1000:.2f} kJ")
    
    return abs(total_conductive) < metabolic_heat * 0.1  # Should be reasonable fraction

# Test with cooling pad
model = jos3.JOS3(ex_output="all")
model.material_temp[3] = 4.0
areas = [0.0] * 17
areas[3] = 0.7
model.contact_area = areas
model.contact_resistance[3] = 0.02

is_valid = validate_energy_conservation(model)
print(f"Energy conservation test: {'PASS' if is_valid else 'FAIL'}")
```

### Temperature Response Validation

```python
def validate_temperature_response():
    """Test that cooling produces expected temperature changes"""
    
    model = jos3.JOS3(ex_output="all")
    
    # Baseline
    model.simulate(times=30, dtime=60)
    baseline_skin = model.Tsk[3]  # Back skin temp
    
    # Apply aggressive cooling
    model.material_temp = [float('nan')] * 17
    model.material_temp[3] = 0.0  # 0°C cooling
    
    areas = [0.0] * 17
    areas[3] = 1.0  # Full coverage
    model.contact_area = areas
    
    model.contact_resistance[3] = 0.001  # Minimal resistance
    model.simulate(times=30, dtime=60)
    
    cooled_skin = model.Tsk[3]
    temp_drop = baseline_skin - cooled_skin
    
    print(f"Baseline skin temp: {baseline_skin:.2f}°C")
    print(f"Cooled skin temp: {cooled_skin:.2f}°C") 
    print(f"Temperature drop: {temp_drop:.2f}°C")
    
    return temp_drop > 5.0  # Should see significant cooling

is_responsive = validate_temperature_response()
print(f"Temperature response test: {'PASS' if is_responsive else 'FAIL'}")
```

---

## Troubleshooting

### Common Issues

#### 1. No Heat Transfer Occurring

**Symptoms:** `QcondXXX` values are 0 or NaN

**Causes & Solutions:**
```python
# Check 1: Material temperature set?
print(f"Material temp: {model._material_temp[3]}")  # Should not be NaN

# Check 2: Contact area properly assigned?
print(f"Contact area: {model._contact_area[3]}")    # Should be > 0

# Check 3: Array assignment method
# ❌ Wrong way
model.contact_area = [0] * 17
model.contact_area[3] = 0.7  # This fails!

# ✅ Right way  
areas = [0] * 17
areas[3] = 0.7
model.contact_area = areas
```

#### 2. Unrealistic Heat Transfer Values

**Symptoms:** Extremely high cooling power (>200W) or temperature changes

**Solutions:**
```python
# Check resistance values
print(f"Resistance: {model._contact_resistance[3]}")  # Should be 0.001-1.0

# Validate contact area
print(f"Contact area: {model._contact_area[3]}")      # Should be 0.0-1.0

# Check temperature difference
results = model.dict_results()
temp_diff = model._material_temp[3] - results["TskBack"][-1]
print(f"Temperature difference: {temp_diff:.2f}°C")   # Should be reasonable
```

#### 3. Core Temperature Not Responding

**Symptoms:** Skin temperature changes but core temperature unchanged

**Explanation:** This is normal for:
- Short simulation periods (< 1 hour)
- Mild cooling/heating
- Small contact areas

**Solutions:**
```python
# Increase simulation duration
model.simulate(times=240, dtime=60)  # 4 hours minimum

# Use more aggressive parameters
model.material_temp[3] = 2.0    # Colder temperature
areas[3] = 0.9                  # Larger contact area
model.contact_resistance[3] = 0.01  # Better contact
```

### Debug Utilities

```python
def debug_conductive_setup(model, segment_idx):
    """Debug conductive heat transfer setup for a segment"""
    
    print(f"=== Segment {segment_idx} Debug ===")
    print(f"Material temp: {model._material_temp[segment_idx]}")
    print(f"Contact area: {model._contact_area[segment_idx]}")
    print(f"Contact resistance: {model._contact_resistance[segment_idx]}")
    print(f"BSA: {model._bsa[segment_idx]:.4f} m²")
    
    # Check if conduction will occur
    if np.isnan(model._material_temp[segment_idx]):
        print("❌ No conduction - material temp is NaN")
    elif model._contact_area[segment_idx] == 0:
        print("❌ No conduction - contact area is 0")
    else:
        print("✅ Conduction setup looks valid")
        
        # Estimate heat transfer
        if hasattr(model, 'Tsk'):
            skin_temp = model.Tsk[segment_idx]
            dt = model._material_temp[segment_idx] - skin_temp
            expected_q = (dt * model._bsa[segment_idx] * 
                         model._contact_area[segment_idx] / 
                         model._contact_resistance[segment_idx])
            print(f"Expected heat transfer: {expected_q:.1f} W")

# Usage
model = jos3.JOS3()
model.material_temp[3] = 4.0
areas = [0.0] * 17
areas[3] = 0.7
model.contact_area = areas
debug_conductive_setup(model, 3)
```

---

## API Reference

### Properties

#### material_temp
```python
@property
def material_temp(self) -> np.ndarray
```
Get/set material temperatures for conductive heat transfer.

**Parameters:**
- Input: array-like (17,) or scalar [°C]
- Use `float('nan')` for no contact

**Example:**
```python
model.material_temp = [float('nan')] * 17
model.material_temp[3] = 4.0  # 4°C on back
```

#### contact_area  
```python
@property  
def contact_area(self) -> np.ndarray
```
Get/set contact area fractions.

**Parameters:**
- Input: array-like (17,) or scalar [0-1]
- Fraction of segment surface in contact

**Example:**
```python
areas = [0.0] * 17
areas[3] = 0.7  # 70% coverage
model.contact_area = areas
```

#### contact_resistance
```python
@property
def contact_resistance(self) -> np.ndarray  
```
Get/set thermal contact resistances.

**Parameters:**
- Input: array-like (17,) or scalar [K⋅m²/W]
- Range: 0.001 to 1.0

**Example:**
```python
model.contact_resistance[3] = 0.02  # Low resistance
```

### Output Parameters

New output parameters available in results:

- `QcondHead`, `QcondNeck`, `QcondChest`, `QcondBack`, etc.: Heat transfer rate [W]
- `MaterialTempHead`, `MaterialTempNeck`, etc.: Material temperatures [°C]  
- `ContactAreaHead`, `ContactAreaNeck`, etc.: Contact area fractions [-]
- `ContactResistanceHead`, `ContactResistanceNeck`, etc.: Thermal resistances [K⋅m²/W]

**Usage:**
```python
model = jos3.JOS3(ex_output="all")
# ... simulation ...
results = model.dict_results()

cooling_power = results["QcondBack"]      # Back cooling power [W]
material_temps = results["MaterialTempBack"]  # Material temperature [°C]  
contact_areas = results["ContactAreaBack"]    # Contact area fraction [-]
```

---

## Version Information

**Enhanced JOS-3 Version:** 1.8.1+conductive  
**Base JOS-3 Version:** 1.8.1  
**Feature Added:** January 2025  
**Documentation Version:** 1.0

## References

1. Takahashi, Y., et al. (2021). "Thermoregulation Model JOS-3 with New Body Segment Model and Geometric Modifications." *Building and Environment*, 207, 108495.

2. ASHRAE Standard 55-2020. "Thermal Environmental Conditions for Human Occupancy."

3. Fiala, D., et al. (2012). "Computer prediction of human thermoregulatory and temperature responses to a wide range of environmental conditions." *International Journal of Biometeorology*, 45(3), 143-159.

---

*For technical support or questions, refer to the main README.md or create an issue in the project repository.*