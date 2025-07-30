"""
Configuration System for JOS3 Heat Transfer Visualization

Implementation of configuration loading and validation
Source: Agile Plan Sprint 4, Epic 4.3, Task 4.3.3
User Story: As a user, I need configuration files to automate complex workflows

Loads YAML/JSON configurations with validation and defaults.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import warnings

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    warnings.warn("PyYAML not available. YAML configuration files will not be supported.")

from ..core.logger import get_logger
from ..core.exceptions import VisualizationError

logger = get_logger(__name__)


class ConfigurationLoader:
    """
    Load and validate configuration files for JOS3 visualization workflows
    
    Implements: TDD Section 4.3 - Configuration Management
    Fulfills: PRD Section 3.4.2 - Configuration system requirements
    """
    
    # Default configuration values
    DEFAULT_CONFIG = {
        'visualization': {
            'default_mode': '2d',
            'default_colormap': 'RdYlBu_r',
            'figure_size': [12, 8],
            'dpi': 150,
            'show_colorbar': True,
            'show_labels': True,
            'background_color': 'white'
        },
        'export': {
            'default_format': 'png',
            'output_directory': './outputs',
            'filename_pattern': '{base_name}_{mode}_{time}.{format}',
            'include_metadata': True,
            'quality': 'high'
        },
        'video': {
            'default_fps': 10,
            'default_format': 'mp4',
            'preset': 'presentation',
            'compression': 'medium',
            'include_timestamp': True,
            'include_metrics': True
        },
        'model_export': {
            'default_format': 'stl',
            'include_heat_data': False,
            'include_contact_geometry': True,
            'units': 'meters',
            'scale_factor': 1.0
        },
        'thermal_analysis': {
            'temperature_units': 'celsius',
            'heat_flow_units': 'watts',
            'energy_balance_tolerance': 0.1,
            'min_significant_heat': 0.1
        },
        'therapy': {
            'device_detection': 'auto',
            'effectiveness_threshold': 0.7,
            'coverage_threshold': 0.5,
            'power_efficiency_target': 15.0  # W/m²
        },
        'logging': {
            'level': 'INFO',
            'console_output': True,
            'file_output': False,
            'log_file': 'jos3_viz.log'
        },
        'performance': {
            'enable_caching': True,
            'max_cache_size_mb': 500,
            'parallel_processing': True,
            'max_workers': None  # Auto-detect
        }
    }
    
    # Configuration validation schema
    VALIDATION_SCHEMA = {
        'visualization.default_mode': {'type': str, 'choices': ['2d', '3d', 'both']},
        'visualization.dpi': {'type': int, 'min': 50, 'max': 600},
        'visualization.figure_size': {'type': list, 'length': 2, 'element_type': (int, float)},
        'export.default_format': {'type': str, 'choices': ['png', 'svg', 'pdf', 'jpg']},
        'video.default_fps': {'type': int, 'min': 1, 'max': 60},
        'video.preset': {'type': str, 'choices': ['presentation', 'research', 'quick_preview', 'publication']},
        'model_export.default_format': {'type': str, 'choices': ['stl', 'obj', 'ply', 'vtk']},
        'model_export.scale_factor': {'type': (int, float), 'min': 0.1, 'max': 10.0},
        'thermal_analysis.energy_balance_tolerance': {'type': (int, float), 'min': 0.01, 'max': 1.0},
        'logging.level': {'type': str, 'choices': ['DEBUG', 'INFO', 'WARNING', 'ERROR']},
        'performance.max_cache_size_mb': {'type': int, 'min': 10, 'max': 5000}
    }
    
    def __init__(self):
        """Initialize configuration loader"""
        self.config = self.DEFAULT_CONFIG.copy()
        self.config_file_path = None
        self.validation_errors = []
        
        logger.info("Configuration loader initialized with defaults")
    
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from file
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
            
        Returns:
            Loaded and validated configuration dictionary
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise VisualizationError(f"Configuration file not found: {config_path}")
        
        self.config_file_path = config_path
        
        try:
            # Load file based on extension
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                if not YAML_AVAILABLE:
                    raise VisualizationError("PyYAML required for YAML config files. Install with: pip install pyyaml")
                
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    
            else:
                raise VisualizationError(f"Unsupported config file format: {config_path.suffix}")
            
            if loaded_config is None:
                loaded_config = {}
            
            # Merge with defaults
            self.config = self._merge_configs(self.DEFAULT_CONFIG, loaded_config)
            
            # Validate configuration
            self._validate_config()
            
            logger.info(f"Configuration loaded from: {config_path}")
            return self.config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise VisualizationError(f"Configuration loading failed: {str(e)}")
    
    def save_config(self, config_path: Union[str, Path], format: str = 'yaml') -> bool:
        """
        Save current configuration to file
        
        Args:
            config_path: Output file path
            format: File format ('yaml' or 'json')
            
        Returns:
            True if save successful
        """
        config_path = Path(config_path)
        
        try:
            # Ensure output directory exists
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'yaml':
                if not YAML_AVAILABLE:
                    raise VisualizationError("PyYAML required for YAML export")
                
                with open(config_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
                    
            elif format.lower() == 'json':
                with open(config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
                    
            else:
                raise VisualizationError(f"Unsupported save format: {format}")
            
            logger.info(f"Configuration saved to: {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            return False
    
    def get_config(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Configuration key path (e.g., 'visualization.dpi')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise VisualizationError(f"Configuration key not found: {key_path}")
    
    def set_config(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation
        
        Args:
            key_path: Configuration key path (e.g., 'visualization.dpi')
            value: Value to set
        """
        keys = key_path.split('.')
        config_ref = self.config
        
        # Navigate to parent dictionary
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        # Set the value
        config_ref[keys[-1]] = value
        
        # Validate the change
        self._validate_single_key(key_path, value)
        
        logger.debug(f"Configuration updated: {key_path} = {value}")
    
    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration from dictionary
        
        Args:
            updates: Dictionary with configuration updates
        """
        self.config = self._merge_configs(self.config, updates)
        self._validate_config()
        
        logger.debug(f"Configuration updated from dictionary with {len(updates)} changes")
    
    def override_from_cli_args(self, cli_args: Dict[str, Any]) -> None:
        """
        Override configuration with command-line arguments
        
        Args:
            cli_args: Dictionary of CLI arguments
        """
        # Map CLI args to config keys
        cli_mapping = {
            'mode': 'visualization.default_mode',
            'colormap': 'visualization.default_colormap', 
            'dpi': 'visualization.dpi',
            'format': 'export.default_format',
            'output_dir': 'export.output_directory',
            'fps': 'video.default_fps',
            'preset': 'video.preset',
            'verbose': 'logging.level'
        }
        
        updates_applied = 0
        
        for cli_key, config_key in cli_mapping.items():
            if cli_key in cli_args and cli_args[cli_key] is not None:
                # Special handling for verbose flag
                if cli_key == 'verbose' and cli_args[cli_key]:
                    self.set_config(config_key, 'DEBUG')
                else:
                    self.set_config(config_key, cli_args[cli_key])
                updates_applied += 1
        
        if updates_applied > 0:
            logger.info(f"Applied {updates_applied} CLI overrides to configuration")
    
    def create_example_config(self, output_path: Union[str, Path]) -> bool:
        """
        Create example configuration file with comments
        
        Args:
            output_path: Path for example config file
            
        Returns:
            True if creation successful
        """
        output_path = Path(output_path)
        
        try:
            # Generate commented configuration
            example_config = self._generate_example_config()
            
            if output_path.suffix.lower() in ['.yaml', '.yml']:
                if not YAML_AVAILABLE:
                    raise VisualizationError("PyYAML required for YAML example")
                
                with open(output_path, 'w') as f:
                    f.write(example_config['yaml'])
                    
            elif output_path.suffix.lower() == '.json':
                with open(output_path, 'w') as f:
                    f.write(example_config['json'])
                    
            else:
                # Default to YAML
                output_path = output_path.with_suffix('.yaml')
                with open(output_path, 'w') as f:
                    f.write(example_config['yaml'])
            
            logger.info(f"Example configuration created: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create example config: {str(e)}")
            return False
    
    def validate_therapy_config(self, therapy_config: Dict[str, Any]) -> bool:
        """
        NEW: Validate therapeutic device configuration
        
        Args:
            therapy_config: Therapeutic device configuration
            
        Returns:
            True if configuration is valid
        """
        required_fields = ['device_type', 'segments', 'temperature', 'contact_area']
        
        for field in required_fields:
            if field not in therapy_config:
                self.validation_errors.append(f"Missing required therapy field: {field}")
                return False
        
        # Validate device type
        valid_devices = ['cooling-vest', 'heating-pad', 'comprehensive', 'custom']
        if therapy_config['device_type'] not in valid_devices:
            self.validation_errors.append(f"Invalid device type: {therapy_config['device_type']}")
            return False
        
        # Validate temperature range
        temp = therapy_config['temperature']
        if not (5 <= temp <= 50):  # Reasonable range for therapeutic devices
            self.validation_errors.append(f"Temperature out of range (5-50°C): {temp}")
            return False
        
        # Validate contact area
        contact_area = therapy_config['contact_area']
        if not (0.0 <= contact_area <= 1.0):
            self.validation_errors.append(f"Contact area must be 0-1: {contact_area}")
            return False
        
        return True
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_config(self) -> bool:
        """Validate entire configuration against schema"""
        self.validation_errors = []
        
        for key_path, rules in self.VALIDATION_SCHEMA.items():
            try:
                value = self.get_config(key_path)
                self._validate_single_key(key_path, value, rules)
            except VisualizationError:
                # Key not found - use default
                continue
        
        if self.validation_errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(self.validation_errors)
            logger.error(error_msg)
            raise VisualizationError(error_msg)
        
        return True
    
    def _validate_single_key(self, key_path: str, value: Any, rules: Optional[Dict[str, Any]] = None) -> bool:
        """Validate single configuration key"""
        if rules is None:
            rules = self.VALIDATION_SCHEMA.get(key_path, {})
        
        if not rules:
            return True  # No validation rules defined
        
        # Type validation
        if 'type' in rules:
            expected_type = rules['type']
            if not isinstance(value, expected_type):
                self.validation_errors.append(f"{key_path}: Expected {expected_type}, got {type(value)}")
                return False
        
        # Choice validation
        if 'choices' in rules:
            if value not in rules['choices']:
                self.validation_errors.append(f"{key_path}: Must be one of {rules['choices']}, got {value}")
                return False
        
        # Range validation
        if 'min' in rules and value < rules['min']:
            self.validation_errors.append(f"{key_path}: Must be >= {rules['min']}, got {value}")
            return False
        
        if 'max' in rules and value > rules['max']:
            self.validation_errors.append(f"{key_path}: Must be <= {rules['max']}, got {value}")
            return False
        
        # List validation
        if 'length' in rules and len(value) != rules['length']:
            self.validation_errors.append(f"{key_path}: Must have length {rules['length']}, got {len(value)}")
            return False
        
        if 'element_type' in rules and isinstance(value, list):
            for i, element in enumerate(value):
                if not isinstance(element, rules['element_type']):
                    self.validation_errors.append(f"{key_path}[{i}]: Expected {rules['element_type']}, got {type(element)}")
                    return False
        
        return True
    
    def _generate_example_config(self) -> Dict[str, str]:
        """Generate example configuration with comments"""
        
        yaml_content = """# JOS3 Heat Transfer Visualization Configuration
# Complete configuration example with all available options

# Visualization settings
visualization:
  default_mode: '2d'          # Default visualization mode: 2d, 3d, both
  default_colormap: 'RdYlBu_r'  # Default color scheme
  figure_size: [12, 8]        # Figure dimensions in inches [width, height]
  dpi: 150                    # Resolution in dots per inch
  show_colorbar: true         # Display color scale bar
  show_labels: true           # Show segment labels
  background_color: 'white'   # Background color

# Export settings
export:
  default_format: 'png'       # Default export format: png, svg, pdf, jpg
  output_directory: './outputs'  # Directory for output files
  filename_pattern: '{base_name}_{mode}_{time}.{format}'  # Filename template
  include_metadata: true      # Include metadata in exports
  quality: 'high'             # Export quality: low, medium, high, highest

# Video animation settings
video:
  default_fps: 10             # Frames per second
  default_format: 'mp4'       # Video format: mp4, avi, gif
  preset: 'presentation'      # Animation preset: presentation, research, quick_preview, publication
  compression: 'medium'       # Compression level: low, medium, high
  include_timestamp: true     # Show timestamp overlay
  include_metrics: true       # Show thermal metrics

# 3D model export settings
model_export:
  default_format: 'stl'       # Export format: stl, obj, ply, vtk
  include_heat_data: false    # Embed thermal data in model
  include_contact_geometry: true  # Include therapeutic device geometry
  units: 'meters'             # Model units
  scale_factor: 1.0           # Scaling factor

# Thermal analysis settings
thermal_analysis:
  temperature_units: 'celsius'    # Temperature units: celsius, fahrenheit
  heat_flow_units: 'watts'        # Heat flow units: watts, btu_per_hour
  energy_balance_tolerance: 0.1   # Energy balance validation tolerance
  min_significant_heat: 0.1       # Minimum heat transfer to consider significant (W)

# Therapeutic device analysis
therapy:
  device_detection: 'auto'        # Device detection: auto, manual
  effectiveness_threshold: 0.7    # Minimum effectiveness ratio (0-1)
  coverage_threshold: 0.5         # Minimum coverage ratio (0-1)
  power_efficiency_target: 15.0   # Target power density (W/m²)

# Logging configuration
logging:
  level: 'INFO'               # Log level: DEBUG, INFO, WARNING, ERROR
  console_output: true        # Enable console logging
  file_output: false          # Enable file logging
  log_file: 'jos3_viz.log'    # Log file name

# Performance settings
performance:
  enable_caching: true        # Enable result caching
  max_cache_size_mb: 500      # Maximum cache size in MB
  parallel_processing: true   # Enable parallel processing
  max_workers: null           # Maximum worker threads (null = auto-detect)

# Therapeutic device configurations
therapeutic_devices:
  cooling_vest:
    device_type: 'cooling-vest'
    segments: ['Chest', 'Back']
    temperature: 18             # °C
    contact_area: 0.8           # 80% coverage
    contact_resistance: 0.01    # m²·K/W
    
  heating_pad:
    device_type: 'heating-pad'
    segments: ['LThigh', 'RThigh']
    temperature: 42             # °C
    contact_area: 0.6           # 60% coverage
    contact_resistance: 0.015   # m²·K/W
"""

        json_content = json.dumps(self.DEFAULT_CONFIG, indent=2)
        
        return {
            'yaml': yaml_content,
            'json': json_content
        }
    
    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors"""
        return self.validation_errors.copy()
    
    def is_valid(self) -> bool:
        """Check if current configuration is valid"""
        try:
            self._validate_config()
            return True
        except VisualizationError:
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration"""
        return {
            'config_file': str(self.config_file_path) if self.config_file_path else 'defaults',
            'validation_status': 'valid' if self.is_valid() else 'invalid',
            'validation_errors': len(self.validation_errors),
            'sections': list(self.config.keys()),
            'total_settings': self._count_config_keys(self.config)
        }
    
    def _count_config_keys(self, config_dict: Dict[str, Any]) -> int:
        """Recursively count configuration keys"""
        count = 0
        for key, value in config_dict.items():
            if isinstance(value, dict):
                count += self._count_config_keys(value)
            else:
                count += 1
        return count


# Convenience functions for common configuration operations
def load_config_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration file (convenience function)"""
    loader = ConfigurationLoader()
    return loader.load_config(config_path)


def create_default_config(output_path: Union[str, Path]) -> bool:
    """Create default configuration file (convenience function)"""
    loader = ConfigurationLoader()
    return loader.create_example_config(output_path)


def validate_config_dict(config: Dict[str, Any]) -> bool:
    """Validate configuration dictionary (convenience function)"""
    loader = ConfigurationLoader()
    loader.config = config
    return loader.is_valid()