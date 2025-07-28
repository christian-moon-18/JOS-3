# JOS3 Heat Transfer Visualization Module

**Version:** 1.0.0 (Sprint 1 Complete)  
**Status:** Core functionality implemented  

## Overview

The JOS3 Heat Transfer Visualization Module creates visual representations of heat flow through a 3D/2D humanoid model based on JOS3 thermoregulation simulation outputs. This module enables biomedical R&D engineers to better understand thermal dynamics in human physiology.

## Sprint 1 Implementation Status ✅

**Epic 1.1: Project Infrastructure** ✅
- [x] Project structure following Python package best practices
- [x] Development environment and dependencies setup
- [x] Basic logging and error handling framework

**Epic 1.2: JOS3 Data Integration** ✅
- [x] Core data parser (`JOS3DataParser`)
- [x] Data validation system for JOS3 columns and consistency
- [x] Segment mapping utilities with 17 body segments

**Epic 1.3: External Heat Calculation** ✅
- [x] External heat calculator (`ExternalHeatCalculator`)
- [x] Anthropometric scaling system with segment mass calculations
- [x] Heat balance validation and energy conservation checks

## Key Features Implemented

### 1. JOS3 Data Parser
```python
from jos3_viz.core import JOS3DataParser

# Load JOS3 simulation data
parser = JOS3DataParser('simulation_output.csv')

# Get data for specific time point
data_30min = parser.get_timestep_data(30)

# Extract heat transfer data by mechanism
sensible_heat = parser.get_heat_transfer_data(30, 'sensible')
temperatures = parser.get_temperature_data(30, 'skin')
```

### 2. External Heat Calculator
```python
from jos3_viz.core import ExternalHeatCalculator

# Initialize calculator
heat_calc = ExternalHeatCalculator(parser)

# Calculate external heating/cooling for specific segment
external_heat = heat_calc.calculate_instantaneous_heat(30, 'Chest')
print(f"External heat required: {external_heat:.2f} W")

# Get total body analysis
total_heat = heat_calc.get_total_body_heat(30)
print(f"Total body heat: {total_heat['total_heat']:.2f} W")

# Time-averaged calculations
avg_heat = heat_calc.calculate_time_averaged_heat(0, 60, 'Chest')
print(f"Average power: {avg_heat['average_power']:.2f} W")
```

### 3. Body Segment Support
17 body segments as defined in JOS3:
- **Torso:** Head, Neck, Chest, Back, Pelvis
- **Arms:** LShoulder, LArm, LHand, RShoulder, RArm, RHand  
- **Legs:** LThigh, LLeg, LFoot, RThigh, RLeg, RFoot

### 4. Heat Balance Validation
```python
# Validate energy conservation
balance = heat_calc.validate_energy_balance(30)
print(f"Energy balance: {balance['message']}")
print(f"Imbalance: {balance['imbalance_watts']:.2f} W")
```

## Installation

### Dependencies
```bash
# Install required packages
pip install -r jos3_viz/requirements.txt
```

**Core Dependencies:**
- numpy>=1.21.0
- pandas>=1.3.0  
- matplotlib>=3.4.0
- vtk>=9.0.0
- pyyaml>=5.4.0
- click>=8.0.0
- imageio>=2.9.0
- trimesh>=3.9.0

## Quick Start

### Running the Demo
```bash
cd jos3_viz/examples
python basic_usage.py
```

This demo creates sample JOS3 data and demonstrates:
- Data loading and validation
- External heat calculations
- Energy balance validation
- Heat transfer data extraction

### Basic Usage
```python
import pandas as pd
from jos3_viz.core import JOS3DataParser, ExternalHeatCalculator

# Load your JOS3 simulation data
data = pd.read_csv('your_jos3_output.csv')
parser = JOS3DataParser(data)

# Calculate external heating/cooling requirements
heat_calc = ExternalHeatCalculator(parser)

# Get external heat for specific segment at 30 minutes
chest_heat = heat_calc.calculate_instantaneous_heat(30, 'Chest')
print(f"Chest requires {chest_heat:.1f} W of external heat")

# Analyze heating/cooling needs for whole body
summary = heat_calc.get_heating_cooling_summary()
print(f"Peak heating needed for: {summary['critical_segments']['highest_heating']}")
```

## External Heat Calculation Methodology

The external heat calculation implements the methodology from **TDD Section 5**:

```
Q_external = Q_metabolic - (Q_stored + Q_sensible + Q_latent + Q_respiratory)

Where:
- Q_metabolic: Metabolic heat production (Qcr + Qsk + Qms + Qfat)  
- Q_stored: Heat storage rate (m*c*dT/dt)
- Q_sensible: Convective + radiative heat loss (SHLsk)
- Q_latent: Evaporative heat loss (LHLsk)
- Q_respiratory: Distributed respiratory heat loss
```

**Positive values** = External heating required  
**Negative values** = External cooling required

## Project Structure

```
jos3_viz/
├── __init__.py                 # Main package imports
├── core/                       # Core functionality
│   ├── data_parser.py         # JOS3 data loading and validation
│   ├── heat_calculator.py     # External heat calculations  
│   ├── logger.py              # Logging configuration
│   └── exceptions.py          # Custom exceptions
├── models/                     # Body segment and scaling models
│   ├── body_segments.py       # 17 segment definitions
│   └── scaling.py             # Anthropometric scaling
├── visualization/             # Rendering components (Sprint 2)
├── export/                    # Export functionality (Sprint 2)
├── cli/                       # Command line interface (Sprint 4)
├── config/                    # Configuration management
├── examples/                  # Usage examples
│   └── basic_usage.py         # Demo script
└── requirements.txt           # Dependencies
```

## Validation and Testing

The implementation includes comprehensive validation:

- **Data Validation:** Checks for required JOS3 columns and realistic value ranges
- **Energy Balance:** Validates conservation of energy across body segments  
- **Anthropometric Scaling:** Uses validated body segment mass percentages
- **Error Handling:** Comprehensive exception handling with informative messages

## Next Steps: Sprint 2

**Sprint 2 Goals (Weeks 3-4):**
- 2D heat map visualization
- Color mapping system  
- Static image export (PNG/SVG)
- Configuration file system

## References

- **PRD:** Product Requirements Document - defines what to build
- **TDD:** Technical Design Document - defines how to build it  
- **Agile Plan:** 12-week development timeline with sprint details

## Support

For issues or questions:
1. Check the examples in `jos3_viz/examples/`
2. Review the comprehensive logging output
3. Validate your JOS3 data format against required columns
4. Ensure anthropometric data (height, weight) is available

---

**Status:** Sprint 1 Complete ✅  
**Next:** Sprint 2 - Basic Visualization & 2D Rendering