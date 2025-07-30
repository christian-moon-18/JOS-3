"""
Body segment definitions for JOS3 visualization

Implementation of segment mapping utilities
Source: PRD Appendix A - JOS3 Body Segments
User Story: From Agile Plan Sprint 1, Epic 1.2, Task 1.2.3

JOS3 uses 17 distinct body segments matching standard thermal models.
"""

from typing import List, Dict

# 17 body segments as defined in JOS3
BODY_SEGMENTS: List[str] = [
    "Head",
    "Neck", 
    "Chest",
    "Back",
    "Pelvis",
    "LShoulder",  # Left Shoulder
    "LArm",       # Left Arm
    "LHand",      # Left Hand
    "RShoulder",  # Right Shoulder
    "RArm",       # Right Arm
    "RHand",      # Right Hand
    "LThigh",     # Left Thigh
    "LLeg",       # Left Leg
    "LFoot",      # Left Foot
    "RThigh",     # Right Thigh
    "RLeg",       # Right Leg
    "RFoot"       # Right Foot
]

# Mapping from JOS3 segment names to standardized names
JOS3_SEGMENT_MAPPING: Dict[str, str] = {
    "Head": "Head",
    "Neck": "Neck",
    "Chest": "Chest", 
    "Back": "Back",
    "Pelvis": "Pelvis",
    "LShoulder": "LShoulder",
    "LArm": "LArm",
    "LHand": "LHand",
    "RShoulder": "RShoulder",
    "RArm": "RArm", 
    "RHand": "RHand",
    "LThigh": "LThigh",
    "LLeg": "LLeg",
    "LFoot": "LFoot",
    "RThigh": "RThigh",
    "RLeg": "RLeg",
    "RFoot": "RFoot"
}

# JOS3 parameter suffixes for different data types
JOS3_PARAMETER_TYPES: Dict[str, str] = {
    "core_temperature": "Tcr",
    "skin_temperature": "Tsk", 
    "core_heat_production": "Qcr",
    "skin_heat_production": "Qsk",
    "muscle_heat_production": "Qms",
    "fat_heat_production": "Qfat",
    "sensible_heat_loss": "SHLsk",
    "latent_heat_loss": "LHLsk",
    "total_heat_loss": "THLsk",
    "core_blood_flow": "BFcr",
    "skin_blood_flow": "BFsk",
    "muscle_blood_flow": "BFms",
    "evaporative_heat_loss": "Esk",
    "sweat_evaporation": "Esweat",
    # NEW: Conductive heat transfer parameters
    "conductive_heat": "Qcond",
    "material_temperature": "MaterialTemp",
    "contact_area": "ContactArea",
    "contact_resistance": "ContactResistance"
}

# Required JOS3 columns for visualization
REQUIRED_JOS3_COLUMNS: List[str] = [
    "Met",  # Total metabolic heat production
    "RES"   # Total respiratory heat loss
]

# Add required segment-specific columns (JOS3 format: ParameterSegment not Parameter_Segment)
for segment in BODY_SEGMENTS:
    REQUIRED_JOS3_COLUMNS.extend([
        f"Tcr{segment}",   # Core temperature
        f"Tsk{segment}",   # Skin temperature
        f"Qcr{segment}",   # Core heat production
        f"Qsk{segment}",   # Skin heat production
        f"SHLsk{segment}", # Sensible heat loss
        f"LHLsk{segment}"  # Latent heat loss
    ])

# Optional columns that may be present for some segments (JOS3 format: ParameterSegment)
OPTIONAL_JOS3_COLUMNS: List[str] = []
for segment in BODY_SEGMENTS:
    OPTIONAL_JOS3_COLUMNS.extend([
        f"Qms{segment}",   # Muscle heat production
        f"Qfat{segment}",  # Fat heat production
        f"BFcr{segment}",  # Core blood flow
        f"BFsk{segment}",  # Skin blood flow
        f"BFms{segment}",  # Muscle blood flow
        f"Esk{segment}",   # Evaporative loss
        f"Esweat{segment}", # Sweat evaporation
        # NEW: Conductive heat transfer parameters
        f"Qcond{segment}",        # Conductive heat transfer rate
        f"MaterialTemp{segment}", # Material temperature
        f"ContactArea{segment}",  # Contact area fraction
        f"ContactResistance{segment}" # Contact thermal resistance
    ])


def get_segment_column_name(segment: str, parameter_type: str) -> str:
    """
    Generate JOS3 column name for a specific segment and parameter type
    
    Args:
        segment: Body segment name from BODY_SEGMENTS
        parameter_type: Parameter type key from JOS3_PARAMETER_TYPES
    
    Returns:
        Column name in format: ParameterType_SegmentName
        
    Raises:
        ValueError: If segment or parameter_type not recognized
    """
    if segment not in BODY_SEGMENTS:
        raise ValueError(f"Unknown segment: {segment}. Must be one of {BODY_SEGMENTS}")
    
    if parameter_type not in JOS3_PARAMETER_TYPES:
        raise ValueError(f"Unknown parameter type: {parameter_type}. Must be one of {list(JOS3_PARAMETER_TYPES.keys())}")
    
    parameter_prefix = JOS3_PARAMETER_TYPES[parameter_type]
    return f"{parameter_prefix}_{segment}"


def validate_segment_name(segment: str) -> bool:
    """
    Check if segment name is valid
    
    Args:
        segment: Segment name to validate
        
    Returns:
        True if valid, False otherwise
    """
    return segment in BODY_SEGMENTS


def get_bilateral_segments() -> Dict[str, List[str]]:
    """
    Get pairs of bilateral body segments (left/right)
    
    Returns:
        Dictionary mapping bilateral group to segment pairs
    """
    return {
        "shoulders": ["LShoulder", "RShoulder"],
        "arms": ["LArm", "RArm"],
        "hands": ["LHand", "RHand"],
        "thighs": ["LThigh", "RThigh"], 
        "legs": ["LLeg", "RLeg"],
        "feet": ["LFoot", "RFoot"]
    }


def get_torso_segments() -> List[str]:
    """Get torso/core body segments"""
    return ["Head", "Neck", "Chest", "Back", "Pelvis"]


def get_limb_segments() -> List[str]:
    """Get all limb segments (arms and legs)"""
    bilateral = get_bilateral_segments()
    limbs = []
    for segment_pair in bilateral.values():
        limbs.extend(segment_pair)
    return limbs