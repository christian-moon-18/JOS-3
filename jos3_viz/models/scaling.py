"""
Anthropometric scaling system for body segments

Implementation of segment mass calculations
Source: TDD Section 7.4 - Segment Mass Estimation
User Story: From Agile Plan Sprint 1, Epic 1.3, Task 1.3.2

Based on anthropometric data and standard body segment percentages.
"""

from typing import Dict
from .body_segments import BODY_SEGMENTS, validate_segment_name


# Body segment mass percentages relative to total body mass
# Source: TDD Section 7.4 - Segment Mass Estimation
SEGMENT_MASS_PERCENTAGES: Dict[str, float] = {
    'Head': 0.0694,
    'Neck': 0.0244,
    'Chest': 0.1580,
    'Back': 0.1580,
    'Pelvis': 0.1420,
    'LShoulder': 0.0260, 
    'RShoulder': 0.0260,
    'LArm': 0.0160, 
    'RArm': 0.0160,
    'LHand': 0.0060, 
    'RHand': 0.0060,
    'LThigh': 0.1000, 
    'RThigh': 0.1000,
    'LLeg': 0.0465, 
    'RLeg': 0.0465,
    'LFoot': 0.0145, 
    'RFoot': 0.0145
}

# Specific heat capacities for body tissues [J/(kg·K)]
# Source: TDD Section 5.3 - Implementation
TISSUE_SPECIFIC_HEAT: Dict[str, float] = {
    'core': 3500,  # Body core tissue
    'skin': 3680,  # Skin tissue
    'muscle': 3421,  # Muscle tissue
    'fat': 2348    # Fat tissue
}

# Tissue composition percentages for heat storage calculations
# Approximate values for mass distribution within segments
TISSUE_COMPOSITION: Dict[str, Dict[str, float]] = {
    'default': {
        'core': 0.85,    # 85% core tissue
        'skin': 0.10,    # 10% skin
        'muscle': 0.05,  # 5% muscle (varies by segment)
        'fat': 0.0       # Varies by body fat percentage
    }
}


def get_segment_mass(segment_name: str, total_body_mass: float) -> float:
    """
    Calculate mass of a specific body segment
    
    Args:
        segment_name: Name of body segment (from BODY_SEGMENTS)
        total_body_mass: Total body mass in kg
        
    Returns:
        Segment mass in kg
        
    Raises:
        ValueError: If segment name is invalid
    """
    if not validate_segment_name(segment_name):
        raise ValueError(f"Invalid segment name: {segment_name}")
    
    if segment_name not in SEGMENT_MASS_PERCENTAGES:
        raise ValueError(f"No mass percentage defined for segment: {segment_name}")
    
    if total_body_mass <= 0:
        raise ValueError("Total body mass must be positive")
    
    return total_body_mass * SEGMENT_MASS_PERCENTAGES[segment_name]


def get_all_segment_masses(total_body_mass: float) -> Dict[str, float]:
    """
    Calculate masses for all body segments
    
    Args:
        total_body_mass: Total body mass in kg
        
    Returns:
        Dictionary mapping segment names to masses in kg
    """
    masses = {}
    for segment in BODY_SEGMENTS:
        masses[segment] = get_segment_mass(segment, total_body_mass)
    return masses


def validate_mass_percentages() -> bool:
    """
    Validate that segment mass percentages sum to approximately 1.0
    
    Returns:
        True if percentages are valid, False otherwise
    """
    total_percentage = sum(SEGMENT_MASS_PERCENTAGES.values())
    # Allow small tolerance for floating point arithmetic
    tolerance = 0.001
    return abs(total_percentage - 1.0) < tolerance


def get_effective_specific_heat(segment_name: str, body_fat_percentage: float = 15.0) -> float:
    """
    Calculate effective specific heat for a body segment based on tissue composition
    
    Args:
        segment_name: Name of body segment
        body_fat_percentage: Body fat percentage (default 15%)
        
    Returns:
        Effective specific heat in J/(kg·K)
    """
    if not validate_segment_name(segment_name):
        raise ValueError(f"Invalid segment name: {segment_name}")
    
    # Get tissue composition (simplified model)
    composition = TISSUE_COMPOSITION['default'].copy()
    
    # Adjust for body fat percentage
    fat_fraction = body_fat_percentage / 100.0
    composition['fat'] = min(fat_fraction, 0.3)  # Cap at 30% fat
    composition['core'] = max(0.5, composition['core'] - composition['fat'])
    
    # Calculate weighted average specific heat
    effective_c = 0.0
    for tissue, fraction in composition.items():
        effective_c += fraction * TISSUE_SPECIFIC_HEAT[tissue]
    
    return effective_c


def get_segment_dimensions(segment_name: str, height: float, weight: float) -> Dict[str, float]:
    """
    Estimate basic segment dimensions for 3D model scaling
    
    Args:
        segment_name: Name of body segment
        height: Body height in meters
        weight: Body weight in kg
        
    Returns:
        Dictionary with segment dimensions (length, width, depth in meters)
        
    Note: This is a simplified model for basic 3D visualization
    """
    if not validate_segment_name(segment_name):
        raise ValueError(f"Invalid segment name: {segment_name}")
    
    # Basic scaling relationships (simplified)
    # These are rough approximations for 3D model generation
    dimensions = {}
    
    if segment_name == 'Head':
        dimensions = {'length': 0.24 * height, 'width': 0.16 * height, 'depth': 0.20 * height}
    elif segment_name == 'Neck':
        dimensions = {'length': 0.05 * height, 'width': 0.12 * height, 'depth': 0.12 * height}
    elif segment_name == 'Chest':
        dimensions = {'length': 0.20 * height, 'width': 0.30 * height, 'depth': 0.18 * height}
    elif segment_name == 'Back':
        dimensions = {'length': 0.20 * height, 'width': 0.30 * height, 'depth': 0.15 * height}
    elif segment_name == 'Pelvis':
        dimensions = {'length': 0.15 * height, 'width': 0.28 * height, 'depth': 0.20 * height}
    elif 'Arm' in segment_name:
        dimensions = {'length': 0.186 * height, 'width': 0.08 * height, 'depth': 0.08 * height}
    elif 'Hand' in segment_name:
        dimensions = {'length': 0.108 * height, 'width': 0.08 * height, 'depth': 0.02 * height}
    elif 'Thigh' in segment_name:
        dimensions = {'length': 0.245 * height, 'width': 0.12 * height, 'depth': 0.12 * height}
    elif 'Leg' in segment_name:
        dimensions = {'length': 0.246 * height, 'width': 0.08 * height, 'depth': 0.08 * height}
    elif 'Foot' in segment_name:
        dimensions = {'length': 0.152 * height, 'width': 0.08 * height, 'depth': 0.05 * height}
    else:
        # Default dimensions for shoulders or other segments
        dimensions = {'length': 0.10 * height, 'width': 0.10 * height, 'depth': 0.10 * height}
    
    return dimensions