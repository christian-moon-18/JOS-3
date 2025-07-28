"""
JOS3 Data Parser Module

Implementation of JOS3DataParser class
Source: TDD Section 3.1 - Data Parser Module
User Story: From Agile Plan Sprint 1, Epic 1.2, Task 1.2.1

Extracts and structures JOS3 simulation output data for visualization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, List, Optional, Any
import warnings

from .logger import get_logger
from .exceptions import DataParsingError, DataValidationError
from ..models.body_segments import (
    BODY_SEGMENTS, 
    REQUIRED_JOS3_COLUMNS, 
    OPTIONAL_JOS3_COLUMNS,
    get_segment_column_name,
    validate_segment_name
)

logger = get_logger(__name__)


class JOS3DataParser:
    """
    Parse and validate JOS3 simulation output data
    
    Implements: TDD Section 3.1 - Data Parser Module
    Fulfills: PRD Section 3.2.1 - Input Requirements
    """
    
    def __init__(self, data_source: Optional[Union[str, Path, pd.DataFrame]] = None):
        """
        Initialize parser with JOS3 output
        
        Args:
            data_source: Path to CSV file, DataFrame, or None to load later
        """
        self.data: Optional[pd.DataFrame] = None
        self.time_points: Optional[pd.Index] = None
        self.body_segments: List[str] = BODY_SEGMENTS.copy()
        self.anthropometry: Dict[str, Any] = {}
        self.data_source = data_source
        
        if data_source is not None:
            self.load_data(data_source)
    
    def load_data(self, data_source: Union[str, Path, pd.DataFrame]) -> None:
        """
        Load JOS3 output from CSV file or DataFrame
        
        Args:
            data_source: Path to CSV file or pandas DataFrame
            
        Raises:
            DataParsingError: If data cannot be loaded
        """
        logger.info(f"Loading JOS3 data from {type(data_source).__name__}")
        
        try:
            if isinstance(data_source, pd.DataFrame):
                self.data = data_source.copy()
                logger.info(f"Loaded DataFrame with shape {self.data.shape}")
                
            elif isinstance(data_source, (str, Path)):
                file_path = Path(data_source)
                if not file_path.exists():
                    raise DataParsingError(f"File not found: {file_path}")
                
                if file_path.suffix.lower() == '.csv':
                    self.data = pd.read_csv(file_path)
                    logger.info(f"Loaded CSV file with shape {self.data.shape}")
                elif file_path.suffix.lower() in ['.pkl', '.pickle']:
                    self.data = pd.read_pickle(file_path)
                    logger.info(f"Loaded pickle file with shape {self.data.shape}")
                else:
                    raise DataParsingError(f"Unsupported file format: {file_path.suffix}")
            else:
                raise DataParsingError(f"Unsupported data source type: {type(data_source)}")
            
            # Set time points (assume index is time or first column is time)
            if self.data.index.name in ['time', 'Time'] or isinstance(self.data.index, pd.DatetimeIndex):
                self.time_points = self.data.index
            elif 'time' in self.data.columns or 'Time' in self.data.columns:
                time_col = 'time' if 'time' in self.data.columns else 'Time'
                self.data = self.data.set_index(time_col)
                self.time_points = self.data.index
            else:
                # Use integer index as time points
                self.time_points = self.data.index
                logger.warning("No time column found, using integer index as time points")
            
            # Validate data after loading
            self.validate_data()
            
            # Extract anthropometry if available
            self._extract_anthropometry()
            
        except Exception as e:
            raise DataParsingError(f"Failed to load data: {str(e)}") from e
    
    def validate_data(self) -> bool:
        """
        Validate JOS3 data format and required columns
        
        Returns:
            True if data is valid
            
        Raises:
            DataValidationError: If validation fails
        """
        if self.data is None:
            raise DataValidationError("No data loaded")
        
        logger.info("Validating JOS3 data format")
        
        # Check for required columns
        missing_required = []
        for col in REQUIRED_JOS3_COLUMNS:
            if col not in self.data.columns:
                missing_required.append(col)
        
        if missing_required:
            raise DataValidationError(
                f"Missing required columns: {missing_required}\n"
                f"Available columns: {list(self.data.columns)}"
            )
        
        # Check for optional columns and warn if missing
        missing_optional = []
        for col in OPTIONAL_JOS3_COLUMNS:
            if col not in self.data.columns:
                missing_optional.append(col)
        
        if missing_optional:
            logger.warning(f"Missing optional columns: {len(missing_optional)} columns")
            logger.debug(f"Missing optional columns: {missing_optional}")
        
        # Validate data ranges
        self._validate_temperature_ranges()
        self._validate_heat_transfer_values()
        
        logger.info("Data validation completed successfully")
        return True
    
    def _validate_temperature_ranges(self) -> None:
        """Validate temperature values are within realistic ranges"""
        temp_columns = [col for col in self.data.columns if col.startswith(('Tcr_', 'Tsk_'))]
        
        for col in temp_columns:
            temp_values = self.data[col].dropna()
            if len(temp_values) == 0:
                continue
                
            min_temp, max_temp = temp_values.min(), temp_values.max()
            
            # Check for realistic human body temperature ranges (°C)
            if min_temp < 10 or max_temp > 50:
                warnings.warn(
                    f"Temperature values in {col} outside realistic range: "
                    f"{min_temp:.1f}°C to {max_temp:.1f}°C"
                )
    
    def _validate_heat_transfer_values(self) -> None:
        """Validate heat transfer values are reasonable"""
        heat_columns = [col for col in self.data.columns if any(
            col.startswith(prefix) for prefix in ['Q', 'SHL', 'LHL', 'BF']
        )]
        
        for col in heat_columns:
            values = self.data[col].dropna()
            if len(values) == 0:
                continue
                
            # Check for unrealistic heat transfer values
            if col.startswith('Q') and (values < 0).any():
                warnings.warn(f"Negative heat production values found in {col}")
            
            if col.startswith(('SHL', 'LHL')) and (values < 0).any():
                warnings.warn(f"Negative heat loss values found in {col}")
    
    def get_timestep_data(self, time_index: Union[int, float]) -> pd.Series:
        """
        Extract all data for specific time point
        
        Args:
            time_index: Time index or time value
            
        Returns:
            Series with all data at specified time point
            
        Raises:
            DataParsingError: If time index is invalid
        """
        if self.data is None:
            raise DataParsingError("No data loaded")
        
        try:
            if isinstance(time_index, int):
                # Integer index
                if time_index < 0 or time_index >= len(self.data):
                    raise IndexError(f"Time index {time_index} out of range")
                return self.data.iloc[time_index]
            else:
                # Time value
                return self.data.loc[time_index]
                
        except (KeyError, IndexError) as e:
            raise DataParsingError(f"Invalid time index {time_index}: {str(e)}") from e
    
    def get_heat_transfer_data(self, time_index: Union[int, float], 
                             mechanism: str) -> Dict[str, float]:
        """
        Get specific heat transfer mechanism data for all segments
        
        Args:
            time_index: Time index or time value
            mechanism: Heat transfer mechanism ('conduction', 'convection', 
                      'radiation', 'evaporation', 'total_loss', 'blood_flow')
        
        Returns:
            Dictionary mapping segment names to heat transfer values
            
        Raises:
            DataParsingError: If mechanism or time index is invalid
        """
        if mechanism not in ['conduction', 'convection', 'radiation', 'evaporation', 
                           'total_loss', 'blood_flow', 'sensible', 'latent']:
            raise DataParsingError(f"Unknown heat transfer mechanism: {mechanism}")
        
        timestep_data = self.get_timestep_data(time_index)
        heat_data = {}
        
        for segment in self.body_segments:
            try:
                if mechanism == 'sensible':
                    # Sensible heat loss (convection + radiation)
                    col_name = f"SHLsk_{segment}"
                elif mechanism == 'latent':
                    # Latent heat loss (evaporation)
                    col_name = f"LHLsk_{segment}"
                elif mechanism == 'evaporation':
                    # Evaporative heat loss
                    col_name = f"Esk_{segment}"
                    if col_name not in timestep_data:
                        col_name = f"LHLsk_{segment}"  # Fallback
                elif mechanism == 'total_loss':
                    # Total heat loss from skin
                    col_name = f"THLsk_{segment}"
                    if col_name not in timestep_data:
                        # Calculate from sensible + latent
                        shl = timestep_data.get(f"SHLsk_{segment}", 0)
                        lhl = timestep_data.get(f"LHLsk_{segment}", 0)
                        heat_data[segment] = shl + lhl
                        continue
                elif mechanism == 'blood_flow':
                    # Blood flow heat transfer (simplified)
                    col_name = f"BFsk_{segment}"
                    if col_name not in timestep_data:
                        heat_data[segment] = 0.0
                        continue
                else:
                    # For conduction, convection, radiation - use sensible heat loss as approximation
                    col_name = f"SHLsk_{segment}"
                
                if col_name in timestep_data:
                    heat_data[segment] = float(timestep_data[col_name])
                else:
                    logger.warning(f"Column {col_name} not found, using 0.0")
                    heat_data[segment] = 0.0
                    
            except Exception as e:
                logger.warning(f"Error getting {mechanism} data for {segment}: {str(e)}")
                heat_data[segment] = 0.0
        
        return heat_data
    
    def get_temperature_data(self, time_index: Union[int, float], 
                           temp_type: str = 'skin') -> Dict[str, float]:
        """
        Get temperature data for all segments
        
        Args:
            time_index: Time index or time value
            temp_type: Temperature type ('skin', 'core')
            
        Returns:
            Dictionary mapping segment names to temperatures
        """
        if temp_type not in ['skin', 'core']:
            raise DataParsingError(f"Unknown temperature type: {temp_type}")
        
        timestep_data = self.get_timestep_data(time_index)
        temp_data = {}
        
        prefix = 'Tsk' if temp_type == 'skin' else 'Tcr'
        
        for segment in self.body_segments:
            col_name = f"{prefix}_{segment}"
            if col_name in timestep_data:
                temp_data[segment] = float(timestep_data[col_name])
            else:
                logger.warning(f"Temperature column {col_name} not found")
                temp_data[segment] = 37.0  # Default body temperature
        
        return temp_data
    
    def get_body_segments(self) -> List[str]:
        """Return list of 17 body segments"""
        return self.body_segments.copy()
    
    def get_anthropometry(self) -> Dict[str, Any]:
        """
        Extract anthropometric data from simulation parameters
        
        Returns:
            Dictionary with height, weight, age, etc. if available
        """
        return self.anthropometry.copy()
    
    def _extract_anthropometry(self) -> None:
        """Extract anthropometric parameters from data if available"""
        # This is a placeholder - JOS3 output format may vary
        # Common anthropometric parameters to look for
        anthro_params = {
            'height': ['height', 'Height', 'H'],
            'weight': ['weight', 'Weight', 'W', 'mass'],
            'age': ['age', 'Age'],
            'body_fat': ['bf', 'body_fat', 'fat_percentage']
        }
        
        for param, possible_names in anthro_params.items():
            for name in possible_names:
                if name in self.data.columns:
                    # Take first non-null value (assuming constant across time)
                    value = self.data[name].dropna().iloc[0] if len(self.data[name].dropna()) > 0 else None
                    if value is not None:
                        self.anthropometry[param] = value
                        logger.debug(f"Found {param}: {value}")
                    break
        
        # Set defaults if not found
        if 'height' not in self.anthropometry:
            self.anthropometry['height'] = 1.75  # Default height in meters
            logger.warning("Height not found in data, using default: 1.75m")
        
        if 'weight' not in self.anthropometry:
            self.anthropometry['weight'] = 70.0  # Default weight in kg
            logger.warning("Weight not found in data, using default: 70kg")
    
    def get_time_range(self) -> tuple:
        """
        Get time range of simulation data
        
        Returns:
            Tuple of (start_time, end_time)
        """
        if self.time_points is None:
            return (0, 0)
        return (self.time_points.min(), self.time_points.max())
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary information about loaded data
        
        Returns:
            Dictionary with data characteristics
        """
        if self.data is None:
            return {}
        
        time_start, time_end = self.get_time_range()
        
        return {
            'shape': self.data.shape,
            'time_range': (time_start, time_end),
            'time_points_count': len(self.time_points),
            'segments_available': len(self.body_segments),
            'required_columns_present': all(col in self.data.columns for col in REQUIRED_JOS3_COLUMNS),
            'anthropometry': self.anthropometry,
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024 / 1024
        }