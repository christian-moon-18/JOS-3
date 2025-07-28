# Technical Design Document
## JOS3 Heat Transfer Visualization Module

**Version:** 1.0  
**Date:** January 2025  
**Author:** [Your Name]  
**Status:** Draft

---

## 1. Executive Summary

This technical design document outlines the implementation strategy for a heat transfer visualization module for JOS3. The module will be developed as a standalone package that interfaces with JOS3 output data, providing 3D/2D visualizations of heat transfer mechanisms in a humanoid model.

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────┐
│   JOS3 Simulation   │
│      (Existing)     │
└──────────┬──────────┘
           │ Output Data
           │ (.csv/.pkl)
           ▼
┌─────────────────────┐
│   JOS3-Viz Module   │
├─────────────────────┤
│  • Data Parser      │
│  • Heat Calculator  │
│  • 3D Model Gen     │
│  • Renderer         │
│  • Exporter        │
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│   Output Formats    │
│  • Images (PNG/SVG) │
│  • Video (MP4)      │
│  • 3D Models (STL)  │
└─────────────────────┘
```

### 2.2 Module Structure

```
jos3_viz/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── data_parser.py      # Parse JOS3 output
│   ├── heat_calculator.py  # Calculate external heat/cooling
│   └── config_loader.py    # Load YAML/JSON configs
├── models/
│   ├── __init__.py
│   ├── body_segments.py    # Body segment definitions
│   ├── mannequin.py        # 3D model generation
│   └── scaling.py          # Anthropometric scaling
├── visualization/
│   ├── __init__.py
│   ├── renderer_2d.py      # 2D visualization
│   ├── renderer_3d.py      # 3D visualization
│   ├── color_mapping.py    # Heat-to-color mapping
│   └── annotations.py      # Labels, legends, etc.
├── export/
│   ├── __init__.py
│   ├── image_export.py     # PNG/SVG export
│   ├── video_export.py     # MP4 animation export
│   └── model_export.py     # STL/OBJ export
├── cli/
│   ├── __init__.py
│   └── main.py            # Command-line interface
├── config/
│   ├── default_config.yaml
│   └── body_geometry.json
└── examples/
    ├── basic_visualization.py
    └── sample_config.yaml
```

## 3. Core Components

### 3.1 Data Parser Module

**Purpose:** Extract and structure JOS3 simulation output data

```python
class JOS3DataParser:
    def __init__(self, data_source):
        """
        Initialize parser with JOS3 output
        data_source: Path to CSV or DataFrame
        """
        self.data = None
        self.time_points = None
        self.body_segments = BODY_SEGMENTS  # 17 segments
        
    def load_data(self):
        """Load JOS3 output from CSV or DataFrame"""
        
    def get_timestep_data(self, time_index):
        """Extract all data for specific time point"""
        
    def get_heat_transfer_data(self, time_index, mechanism):
        """
        Get specific heat transfer mechanism data
        mechanism: 'conduction', 'convection', 'radiation', 'evaporation'
        
        JOS3 Parameter Mappings:
        - Convection + Radiation: SHLsk_[segment] (Sensible Heat Loss)
        - Evaporation: LHLsk_[segment] (Latent Heat Loss)
        - Blood flow: BFcr_[segment], BFsk_[segment], BFms_[segment]
        - Temperatures: Tcr_[segment], Tsk_[segment]
        """
```

### 3.2 Heat Calculator Module

**Purpose:** Calculate external heating/cooling power requirements

```python
class ExternalHeatCalculator:
    def __init__(self, jos3_data):
        self.data = jos3_data
        
    def calculate_instantaneous_heat(self, time_index, body_segment):
        """
        Calculate instantaneous external heat/cooling power
        Returns: Power in Watts for specific segment
        """
        # Sum all heat transfer mechanisms
        # Q_external = Q_metabolic - (Q_conduction + Q_convection + 
        #                            Q_radiation + Q_evaporation + Q_storage)
        
    def calculate_time_averaged_heat(self, start_time, end_time, body_segment):
        """Calculate time-averaged external power"""
        
    def get_total_body_heat(self, time_index):
        """Sum external heat for all body segments"""
```

### 3.3 3D Model Generation

**Purpose:** Create scalable 3D humanoid model with 17 body segments

```python
class MannequinGenerator:
    def __init__(self, anthropometry):
        """
        anthropometry: dict with height, weight, body_fat
        """
        self.height = anthropometry['height']
        self.weight = anthropometry['weight']
        self.body_fat = anthropometry.get('body_fat', 15)
        
    def generate_body_segments(self):
        """
        Generate simple 3D primitives for each body segment
        Returns: Dict of meshes for each segment
        """
        segments = {}
        # Use cylinders for limbs, spheres for joints, 
        # boxes for torso segments
        
    def apply_scaling(self, base_segments):
        """Scale segments based on anthropometry"""
```

**Body Segment Geometry Approach:**
- Head: Sphere/Ellipsoid
- Neck: Cylinder
- Chest/Back/Pelvis: Rectangular prisms
- Arms/Legs: Tapered cylinders
- Hands/Feet: Simplified box shapes

### 3.4 Visualization Renderer

**Purpose:** Render heat transfer data on 3D/2D models

```python
class HeatTransferRenderer:
    def __init__(self, mannequin, heat_data):
        self.mannequin = mannequin
        self.heat_data = heat_data
        self.colormap = 'RdBu_r'  # Red-Blue reversed
        
    def render_3d(self, config):
        """
        Render 3D visualization
        config: Dict with show_conduction, show_convection, etc.
        """
        
    def render_2d(self, config):
        """Render 2D body map visualization"""
        
    def apply_heat_mapping(self, segment, heat_value):
        """Map heat value to color"""
```

## 4. Implementation Details

### 4.1 Technology Stack

**Core Dependencies:**
```python
# requirements.txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0      # 2D visualization
vtk>=9.0.0            # 3D visualization
pyyaml>=5.4.0         # Configuration files
click>=8.0.0          # CLI framework
imageio>=2.9.0        # Video export
trimesh>=3.9.0        # 3D geometry operations
```

### 4.2 Configuration File Format

```yaml
# config.yaml
visualization:
  mode: "3d"  # or "2d"
  time_points: [0, 30, 60, 120, 180]  # minutes
  
heat_transfer:
  show_conduction: true
  show_convection: true
  show_radiation: false
  show_evaporation: false
  show_blood_flow: false
  
appearance:
  colormap: "RdBu_r"
  temperature_range: "auto"  # or [min, max]
  opacity: 0.8
  
export:
  format: ["png", "mp4"]
  resolution: [1920, 1080]
  fps: 10  # for video
  
calculation:
  external_heat_segments: ["all"]  # or specific list
  time_averaging_window: 60  # minutes
```

### 4.3 Command-Line Interface

```bash
# Basic usage
jos3-viz simulate_output.csv --config config.yaml

# Specific time point
jos3-viz simulate_output.csv --time 30 --output heat_map_30min.png

# Generate video
jos3-viz simulate_output.csv --video --start 0 --end 180 --fps 10

# Calculate external heat
jos3-viz simulate_output.csv --calc-external-heat --segments "Chest,Back"
```

### 4.4 Data Processing Pipeline

```python
def visualization_pipeline(jos3_output, config):
    """Main visualization pipeline"""
    
    # 1. Parse JOS3 data
    parser = JOS3DataParser(jos3_output)
    data = parser.load_data()
    
    # 2. Extract anthropometry
    anthropometry = parser.get_anthropometry()
    
    # 3. Generate 3D model
    mannequin = MannequinGenerator(anthropometry)
    body_model = mannequin.generate_body_segments()
    
    # 4. Calculate heat transfer
    heat_calc = ExternalHeatCalculator(data)
    
    # 5. Render visualization
    renderer = HeatTransferRenderer(body_model, data)
    
    # 6. Process each time point
    for time_point in config['time_points']:
        heat_data = heat_calc.calculate_instantaneous_heat(time_point)
        
        if config['mode'] == '3d':
            scene = renderer.render_3d(heat_data, config)
        else:
            scene = renderer.render_2d(heat_data, config)
            
        # 7. Export
        export_frame(scene, time_point, config)
```

## 5. External Heat Calculation Methodology

### 5.1 Instantaneous Calculation

For each body segment at time t:

```
Q_external = Q_metabolic - (Q_stored + Q_conduction + Q_convection + 
                           Q_radiation + Q_evaporation + Q_respiration)

Where:
- Q_metabolic: Metabolic heat production
- Q_stored: Rate of heat storage (m*c*dT/dt)
- Q_conduction: Inter-segment conductive heat transfer
- Q_convection: Convective heat loss to environment
- Q_radiation: Radiative heat exchange
- Q_evaporation: Evaporative heat loss (sweating)
- Q_respiration: Respiratory heat loss
```

### 5.2 JOS3 Parameter Mapping

**Available JOS3 Output Parameters:**

For each body segment (suffix: _Head, _Neck, _Chest, etc.):
- `Tcr`: Core temperature [°C]
- `Tsk`: Skin temperature [°C]
- `Qcr`: Core heat production [W]
- `Qsk`: Skin heat production [W]
- `Qms`: Muscle heat production [W] (only for segments with muscle)
- `Qfat`: Fat heat production [W] (only for segments with fat)
- `SHLsk`: Sensible heat loss from skin (convection + radiation) [W]
- `LHLsk`: Latent heat loss from skin (evaporation) [W]
- `THLsk`: Total heat loss from skin [W]
- `BFcr`: Core blood flow [L/h]
- `BFsk`: Skin blood flow [L/h]
- `BFms`: Muscle blood flow [L/h]
- `Esk`: Evaporative heat loss at skin [W]
- `Esweat`: Evaporative heat loss by sweating only [W]

Whole body parameters:
- `RES`: Total respiratory heat loss [W]
- `Met`: Total metabolic heat production [W]

### 5.3 Implementation

```python
def calculate_external_heat_segment(segment_data, segment_name, time_index, dt=60):
    """
    Calculate external heat for one segment using JOS3 parameters
    
    Parameters:
    - segment_data: DataFrame with JOS3 output
    - segment_name: str, e.g., 'Head', 'Chest', etc.
    - time_index: int, current time step
    - dt: float, time step in seconds (default 60)
    
    Returns:
    - q_external: float, external heat/cooling power in Watts
    """
    
    # Get heat production (metabolic)
    q_met = 0
    q_met += segment_data[f'Qcr_{segment_name}'][time_index]  # Core
    q_met += segment_data[f'Qsk_{segment_name}'][time_index]  # Skin
    
    # Add muscle and fat if they exist for this segment
    if f'Qms_{segment_name}' in segment_data.columns:
        q_met += segment_data[f'Qms_{segment_name}'][time_index]
    if f'Qfat_{segment_name}' in segment_data.columns:
        q_met += segment_data[f'Qfat_{segment_name}'][time_index]
    
    # Get heat losses (already calculated by JOS3)
    q_sensible = segment_data[f'SHLsk_{segment_name}'][time_index]  # Convection + Radiation
    q_latent = segment_data[f'LHLsk_{segment_name}'][time_index]    # Evaporation
    
    # Calculate heat storage rate
    if time_index > 0:
        # Core temperature change
        dTcr = segment_data[f'Tcr_{segment_name}'][time_index] - \
               segment_data[f'Tcr_{segment_name}'][time_index-1]
        
        # Skin temperature change  
        dTsk = segment_data[f'Tsk_{segment_name}'][time_index] - \
               segment_data[f'Tsk_{segment_name}'][time_index-1]
        
        # Get segment mass and specific heat (would need anthropometric data)
        # For now, use approximate values
        segment_mass = get_segment_mass(segment_name)  # kg
        c_core = 3500  # J/(kg·K) - specific heat of body tissue
        c_skin = 3680  # J/(kg·K) - specific heat of skin
        
        # Assume mass distribution: 90% core, 10% skin
        q_stored = (0.9 * segment_mass * c_core * dTcr + 
                   0.1 * segment_mass * c_skin * dTsk) / dt
    else:
        q_stored = 0
    
    # Respiratory heat loss (distributed across segments)
    # Total RES is divided proportionally by segment metabolic rate
    total_met = segment_data['Met'][time_index]
    q_resp = segment_data['RES'][time_index] * (q_met / total_met)
    
    # External heat required (positive = heating, negative = cooling)
    q_external = q_met - (q_stored + q_sensible + q_latent + q_resp)
    
    return q_external


def calculate_total_external_heat(jos3_data, time_index):
    """
    Calculate total external heating/cooling for entire body
    
    Returns dict with:
    - segment_heat: dict of heat/cooling per segment
    - total_heat: total for whole body
    - heating_segments: list of segments needing heating
    - cooling_segments: list of segments needing cooling
    """
    segments = ['Head', 'Neck', 'Chest', 'Back', 'Pelvis', 
                'LShoulder', 'LArm', 'LHand', 'RShoulder', 'RArm', 'RHand',
                'LThigh', 'LLeg', 'LFoot', 'RThigh', 'RLeg', 'RFoot']
    
    results = {
        'segment_heat': {},
        'total_heat': 0,
        'heating_segments': [],
        'cooling_segments': []
    }
    
    for segment in segments:
        q_ext = calculate_external_heat_segment(jos3_data, segment, time_index)
        results['segment_heat'][segment] = q_ext
        results['total_heat'] += q_ext
        
        if q_ext > 0:
            results['heating_segments'].append(segment)
        elif q_ext < 0:
            results['cooling_segments'].append(segment)
    
    return results
```

## 6. Performance Optimization

### 6.1 Memory Management
- Process data in chunks for large simulations
- Use numpy arrays for efficient computation
- Implement lazy loading for time series data

### 6.2 Rendering Optimization
- Use VTK's built-in decimation for complex meshes
- Implement level-of-detail (LOD) for 3D models
- Cache rendered frames for video generation

### 6.3 Parallel Processing
- Use multiprocessing for batch frame generation
- Parallelize heat calculations across body segments

## 7. Detailed External Heat Calculation Specification

### 7.1 Calculation Overview

The external heating/cooling calculation determines how much additional heat must be added or removed from each body segment to maintain the temperatures predicted by JOS3. This is critical for designing therapeutic cooling/heating systems.

### 7.2 Heat Balance Equation

For each body segment, the heat balance is:

```
Heat Storage = Heat Production - Heat Loss + External Heat

Rearranging:
External Heat = Heat Storage - Heat Production + Heat Loss
```

### 7.3 Component Calculations

#### 7.3.1 Metabolic Heat Production
```python
Q_metabolic = Qcr + Qsk + Qms + Qfat
```
- All values directly from JOS3 output
- Note: Qms and Qfat only exist for certain segments

#### 7.3.2 Heat Losses
```python
Q_loss_total = SHLsk + LHLsk + Q_respiratory_distributed
```
- `SHLsk`: Sensible heat loss (convection + radiation combined)
- `LHLsk`: Latent heat loss (evaporation)
- `Q_respiratory_distributed`: Portion of total respiratory loss

#### 7.3.3 Heat Storage Rate
```python
Q_storage = m_segment * c_effective * (dT/dt)
```
Where:
- `m_segment`: Mass of the body segment [kg]
- `c_effective`: Weighted average specific heat
- `dT/dt`: Rate of temperature change

### 7.4 Segment Mass Estimation

Based on anthropometric data and standard body segment percentages:

```python
SEGMENT_MASS_PERCENTAGES = {
    'Head': 0.0694,
    'Neck': 0.0244,
    'Chest': 0.1580,
    'Back': 0.1580,
    'Pelvis': 0.1420,
    'LShoulder': 0.0260, 'RShoulder': 0.0260,
    'LArm': 0.0160, 'RArm': 0.0160,
    'LHand': 0.0060, 'RHand': 0.0060,
    'LThigh': 0.1000, 'RThigh': 0.1000,
    'LLeg': 0.0465, 'RLeg': 0.0465,
    'LFoot': 0.0145, 'RFoot': 0.0145
}

def get_segment_mass(segment_name, total_body_mass):
    return total_body_mass * SEGMENT_MASS_PERCENTAGES[segment_name]
```

### 7.5 Time-Averaged Calculations

For therapy planning, time-averaged values are often more useful:

```python
def calculate_time_averaged_external_heat(jos3_data, start_time, end_time, segment):
    """
    Calculate average external heat over a time period
    
    Returns:
    - average_power: Average power in Watts
    - total_energy: Total energy in Joules
    - peak_power: Maximum instantaneous power
    """
    powers = []
    for t in range(start_time, end_time):
        q_ext = calculate_external_heat_segment(jos3_data, segment, t)
        powers.append(q_ext)
    
    average_power = np.mean(powers)
    total_energy = np.sum(powers) * 60  # Assuming 60s time steps
    peak_power = np.max(np.abs(powers))
    
    return {
        'average_power': average_power,
        'total_energy': total_energy,
        'peak_power': peak_power
    }
```

### 7.6 Validation Considerations

1. **Energy Conservation**: Total body heat production should equal total heat loss + storage
2. **Steady State**: At steady state, storage rate should approach zero
3. **Physical Limits**: External cooling/heating should be within realistic bounds

### 7.7 Output Format

The external heat calculator will provide:
- Instantaneous power requirements per segment [W]
- Time-averaged power requirements [W]
- Total energy transfer over simulation period [J]
- Identification of critical segments requiring most cooling/heating
- Temporal profiles showing when cooling/heating is most needed

## 8. Development Roadmap

## 8. Development Roadmap

### Phase 1: Core Implementation (Weeks 1-4)
- [ ] Set up project structure and dependencies
- [ ] Implement data parser for JOS3 output
- [ ] Create basic 2D visualization with matplotlib
- [ ] Implement external heat calculation

### Phase 2: 3D Visualization (Weeks 5-8)
- [ ] Develop 3D mannequin generator
- [ ] Implement VTK-based 3D renderer
- [ ] Add heat mapping to 3D model
- [ ] Create export functionality

### Phase 3: Full Feature Set (Weeks 9-12)
- [ ] Implement all heat transfer visualization modes
- [ ] Add video generation capability
- [ ] Create comprehensive CLI
- [ ] Write documentation and examples

## 9. Testing Strategy

### 9.1 Validation Tests
- Compare calculated external heat with known scenarios
- Verify visualization accuracy against reference data
- Test anthropometric scaling accuracy

### 9.2 Integration Tests
- Test compatibility with various JOS3 output formats
- Verify export file compatibility (SolidWorks, etc.)
- Test configuration file parsing

### 9.3 Performance Tests
- Benchmark processing time for various simulation sizes
- Monitor memory usage during visualization
- Test rendering performance

## 10. Deployment

### 10.1 Package Structure
```bash
# Installation from source
git clone https://github.com/yourorg/jos3-viz.git
cd jos3-viz
pip install -e .
```

### 10.2 Distribution Options
1. **Source distribution:** Direct GitHub repository access
2. **Wheel package:** Pre-built for Windows systems
3. **Standalone executable:** Using PyInstaller for non-Python users

## 11. Future Enhancements

- Real-time visualization during JOS3 simulation
- Web-based visualization interface
- VR/AR support for immersive analysis
- Machine learning-based heat pattern analysis
- Integration with thermal imaging data

## Appendices

### A. Example Usage Scenarios

```python
# Scenario 1: Visualize cooling therapy
from jos3_viz import visualize_heat_transfer

# Load simulation of cooling blanket therapy
viz = visualize_heat_transfer(
    'cooling_therapy_72hr.csv',
    config='cooling_config.yaml'
)

# Generate snapshots at key time points
viz.export_images(time_points=[0, 60, 360, 720, 1440])  # minutes

# Calculate required cooling power
cooling_power = viz.calculate_external_heat(
    segments=['Chest', 'Back', 'Pelvis'],
    time_range=(0, 1440)
)
print(f"Average cooling power required: {cooling_power['average']} W")
```

### B. Configuration Templates

Available in `jos3_viz/config/templates/`
- `cooling_therapy.yaml`
- `heating_assessment.yaml`
- `exercise_recovery.yaml`
- `publication_figures.yaml`