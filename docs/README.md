# JOS-3 Enhanced Documentation

This directory contains comprehensive documentation for the enhanced JOS-3 thermoregulation model with conductive heat transfer capabilities.

## 📋 Contents

### Core Documentation

- **[CONDUCTIVE_HEAT_TRANSFER_GUIDE.md](CONDUCTIVE_HEAT_TRANSFER_GUIDE.md)** - Complete technical guide covering:
  - Quick start examples
  - Technical implementation details
  - Parameter reference with typical values
  - Usage patterns for different applications
  - Validation and testing procedures
  - Troubleshooting common issues
  - Full API reference

### Templates and Examples

- **[fever_cooling_template.py](fever_cooling_template.py)** - Comprehensive template for fever management simulations:
  - Multi-phase simulation workflow
  - Configurable patient and device parameters
  - Automatic validation and error checking
  - Professional visualization and reporting
  - Energy analysis and effectiveness metrics

## 🚀 Quick Start

### For New Users

1. Read the **Quick Start** section in `CONDUCTIVE_HEAT_TRANSFER_GUIDE.md`
2. Run the validated template: `python fever_cooling_template.py`
3. Modify the configuration parameters in the template for your use case

### Key Learning Points

⚠️ **Critical:** Always use proper array assignment for contact parameters:
```python
# ❌ WRONG - Property indexing fails
model.contact_area = [0] * 17
model.contact_area[3] = 0.7

# ✅ CORRECT - Modify array then assign
areas = [0] * 17
areas[3] = 0.7
model.contact_area = areas
```

## 📊 Validation Status

The documentation and templates have been validated with:

- ✅ **Functional validation**: All code examples execute without errors
- ✅ **Technical accuracy**: Implementation matches JOS-3 architecture  
- ✅ **Energy conservation**: Heat transfer calculations are physically consistent
- ✅ **Clinical relevance**: Parameters reflect realistic therapeutic scenarios

## 🎯 Use Cases Covered

### Medical Applications
- Fever management with cooling pads
- Hypothermia prevention with heating blankets
- Targeted therapeutic cooling (TTM)
- Post-operative temperature management

### Research Applications  
- Thermal comfort studies with heated/cooled surfaces
- Occupational heat stress with cooling garments
- Sports performance with cooling devices
- Environmental exposure with protective equipment

## 📖 Documentation Structure

```
docs/
├── README.md                           # This file
├── CONDUCTIVE_HEAT_TRANSFER_GUIDE.md   # Complete technical guide
└── fever_cooling_template.py           # Validated example template
```

## 🔧 Parameter Quick Reference

| Parameter | Range | Units | Typical Values |
|-----------|-------|-------|----------------|
| `material_temp` | Any | °C | Cooling: 0-15°C, Heating: 35-45°C |
| `contact_area` | 0.0-1.0 | - | Small pad: 0.3-0.5, Large: 0.7-0.9 |
| `contact_resistance` | 0.001-1.0 | K⋅m²/W | Good contact: 0.01-0.02, Poor: 0.1+ |

## 🧪 Body Segment Reference

| Index | Segment | Index | Segment | Common Use |
|-------|---------|-------|---------|------------|
| 0 | Head | 9 | R_Forearm | Head cooling |
| 1 | Neck | 10 | L_Hand | - |
| 2 | Chest | 11 | R_Hand | Cooling vests |
| **3** | **Back** | 12 | R_Hand | **Primary target** |
| 4 | Pelvis | 13 | L_Thigh | Large muscle mass |
| 5 | L_Shoulder | 14 | R_Thigh | Large muscle mass |
| 6 | L_Arm | 15 | L_Leg | - |
| 7 | R_Arm | 16 | R_Leg | - |
| 8 | L_Forearm | | | |

## 💡 Tips for Success

1. **Start with the template** - Modify `fever_cooling_template.py` rather than starting from scratch
2. **Validate setup** - Use the built-in validation functions before running long simulations
3. **Allow sufficient time** - Core temperature changes require 2-4 hours of simulation
4. **Check energy balance** - Cooling power should be reasonable fraction of metabolic rate
5. **Use realistic parameters** - Refer to the parameter tables in the guide

## 🐛 Common Issues

| Issue | Cause | Solution |
|-------|-------|---------|
| No heat transfer | Wrong array assignment | Use proper contact_area assignment |
| Unrealistic values | Bad parameters | Check resistance and area ranges |
| No core response | Short simulation | Extend to 4+ hours |

## 📞 Support

For technical questions or issues:
1. Check the **Troubleshooting** section in the guide
2. Review the validation examples
3. Refer to the main project README.md

---

*Documentation Version: 1.0*  
*Last Updated: January 2025*  
*Enhanced JOS-3 Development Team*