"""
External Heat Calculator Module

Implementation of ExternalHeatCalculator class
Source: TDD Section 5 - External Heat Calculation Methodology
User Story: From Agile Plan Sprint 1, Epic 1.3, Task 1.3.1

Calculate external heating/cooling power requirements for body segments.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import warnings

from .logger import get_logger
from .exceptions import HeatCalculationError, DataValidationError
from .data_parser import JOS3DataParser
from ..models.body_segments import BODY_SEGMENTS, validate_segment_name
from ..models.scaling import get_segment_mass, get_effective_specific_heat

logger = get_logger(__name__)


class ExternalHeatCalculator:
    """
    Calculate external heating/cooling power for body segments
    
    Implements: TDD Section 5.2 - External Heat Calculation Algorithm
    Fulfills: PRD Section 3.2.2 - Calculated Metrics requirement
    """
    
    def __init__(self, jos3_data: Union[JOS3DataParser, pd.DataFrame]):
        """
        Initialize calculator with JOS3 simulation data
        
        Args:
            jos3_data: JOS3DataParser instance or DataFrame with simulation data
        """
        if isinstance(jos3_data, JOS3DataParser):
            self.parser = jos3_data
            self.data = jos3_data.data
            self.anthropometry = jos3_data.get_anthropometry()
        elif isinstance(jos3_data, pd.DataFrame):
            self.parser = JOS3DataParser(jos3_data)
            self.data = jos3_data
            self.anthropometry = self.parser.get_anthropometry()
        else:
            raise HeatCalculationError(f"Invalid data type: {type(jos3_data)}")
        
        if self.data is None:
            raise HeatCalculationError("No data available for heat calculations")
        
        self.segments = BODY_SEGMENTS.copy()
        self.time_step_seconds = 60  # Default 60 seconds between time points
        
        logger.info(f"Initialized heat calculator with {len(self.data)} time points")
    
    def calculate_instantaneous_heat(self, time_index: Union[int, float], 
                                   body_segment: str) -> float:
        """
        Calculate instantaneous external heat/cooling power for specific segment
        
        Implements: TDD Section 5.3 - Implementation
        Algorithm follows exact specification in TDD Section 5.3
        
        Args:
            time_index: Time point index or value
            body_segment: Body segment name from BODY_SEGMENTS
            
        Returns:
            External heat power in Watts (positive = heating, negative = cooling)
            
        Raises:
            HeatCalculationError: If calculation fails
        """
        if not validate_segment_name(body_segment):
            raise HeatCalculationError(f"Invalid segment name: {body_segment}")
        
        try:
            # Get data for this time point
            timestep_data = self.parser.get_timestep_data(time_index)
            
            # 1. Calculate metabolic heat production
            q_met = self._get_metabolic_heat(timestep_data, body_segment)
            
            # 2. Calculate heat losses (already computed by JOS3)
            q_sensible = timestep_data.get(f'SHLsk{body_segment}', 0.0)  # Convection + Radiation
            q_latent = timestep_data.get(f'LHLsk{body_segment}', 0.0)    # Evaporation
            
            # 3. Calculate heat storage rate
            q_stored = self._calculate_heat_storage_rate(time_index, body_segment)
            
            # 4. Calculate respiratory heat loss (distributed across segments)
            q_resp = self._calculate_respiratory_loss_distributed(timestep_data, body_segment, q_met)
            
            # 5. Get conductive heat transfer (if available) - NEW
            q_conductive = timestep_data.get(f'Qcond{body_segment}', 0.0)
            
            # 6. External heat required (heat balance equation) - UPDATED
            # Q_external = Q_metabolic - (Q_stored + Q_sensible + Q_latent + Q_respiratory + Q_conductive)
            q_external = q_met - (q_stored + q_sensible + q_latent + q_resp + q_conductive)
            
            logger.debug(f"Heat calculation for {body_segment} at t={time_index}: "
                        f"Met={q_met:.2f}W, Stored={q_stored:.2f}W, "
                        f"Sensible={q_sensible:.2f}W, Latent={q_latent:.2f}W, "
                        f"Resp={q_resp:.2f}W, Conductive={q_conductive:.2f}W, External={q_external:.2f}W")
            
            return float(q_external)
            
        except Exception as e:
            raise HeatCalculationError(
                f"Failed to calculate heat for {body_segment} at time {time_index}: {str(e)}"
            ) from e
    
    def _get_metabolic_heat(self, timestep_data: pd.Series, segment: str) -> float:
        """Get total metabolic heat production for segment"""
        q_met = 0.0
        
        # Core heat production (always present)
        q_met += timestep_data.get(f'Qcr{segment}', 0.0)
        
        # Skin heat production (always present)
        q_met += timestep_data.get(f'Qsk{segment}', 0.0)
        
        # Muscle heat production (if present)
        if f'Qms{segment}' in timestep_data.index:
            q_met += timestep_data.get(f'Qms{segment}', 0.0)
        
        # Fat heat production (if present)
        if f'Qfat{segment}' in timestep_data.index:
            q_met += timestep_data.get(f'Qfat{segment}', 0.0)
        
        return q_met
    
    def _calculate_heat_storage_rate(self, time_index: Union[int, float], segment: str) -> float:
        """
        Calculate heat storage rate for segment
        
        Formula: Q_storage = m_segment * c_effective * (dT/dt)
        """
        if isinstance(time_index, int) and time_index == 0:
            return 0.0  # No storage rate at first time point
        
        try:
            # Get current and previous temperatures
            current_data = self.parser.get_timestep_data(time_index)
            
            if isinstance(time_index, int):
                prev_data = self.parser.get_timestep_data(time_index - 1)
            else:
                # Find previous time point
                time_points = list(self.data.index)
                current_idx = time_points.index(time_index)
                if current_idx == 0:
                    return 0.0
                prev_data = self.parser.get_timestep_data(time_points[current_idx - 1])
            
            # Temperature changes
            dTcr = current_data.get(f'Tcr{segment}', 37.0) - prev_data.get(f'Tcr{segment}', 37.0)
            dTsk = current_data.get(f'Tsk{segment}', 34.0) - prev_data.get(f'Tsk{segment}', 34.0)
            
            # Get segment mass and specific heat
            segment_mass = get_segment_mass(segment, self.anthropometry.get('weight', 70.0))
            c_effective = get_effective_specific_heat(segment, self.anthropometry.get('body_fat', 15.0))
            
            # Simplified model: assume uniform temperature change
            # More sophisticated models would separate core/skin masses
            dT_avg = (dTcr + dTsk) / 2.0
            
            # Heat storage rate
            q_stored = segment_mass * c_effective * dT_avg / self.time_step_seconds
            
            return q_stored
            
        except Exception as e:
            logger.warning(f"Could not calculate storage rate for {segment}: {str(e)}")
            return 0.0
    
    def _calculate_respiratory_loss_distributed(self, timestep_data: pd.Series, 
                                              segment: str, segment_met: float) -> float:
        """
        Distribute total respiratory heat loss across segments proportionally
        """
        try:
            total_res = timestep_data.get('RES', 0.0)
            total_met = timestep_data.get('Met', 1.0)  # Avoid division by zero
            
            if total_met <= 0:
                return 0.0
            
            # Distribute respiratory loss proportionally by metabolic rate
            segment_respiratory = total_res * (segment_met / total_met)
            
            return segment_respiratory
            
        except Exception as e:
            logger.warning(f"Could not calculate respiratory loss for {segment}: {str(e)}")
            return 0.0
    
    def calculate_time_averaged_heat(self, start_time: Union[int, float], 
                                   end_time: Union[int, float], 
                                   body_segment: str) -> Dict[str, float]:
        """
        Calculate time-averaged external power for segment over time period
        
        Args:
            start_time: Start time index/value
            end_time: End time index/value  
            body_segment: Body segment name
            
        Returns:
            Dictionary with average_power, total_energy, peak_power
        """
        if not validate_segment_name(body_segment):
            raise HeatCalculationError(f"Invalid segment name: {body_segment}")
        
        try:
            # Get time indices in range
            if isinstance(start_time, int) and isinstance(end_time, int):
                time_indices = range(start_time, end_time + 1)
            else:
                # Filter time points in range
                time_mask = (self.data.index >= start_time) & (self.data.index <= end_time)
                time_indices = self.data.index[time_mask]
            
            if len(time_indices) == 0:
                raise HeatCalculationError(f"No time points found in range {start_time} to {end_time}")
            
            # Calculate instantaneous power for each time point
            powers = []
            for t in time_indices:
                try:
                    q_ext = self.calculate_instantaneous_heat(t, body_segment)
                    powers.append(q_ext)
                except Exception as e:
                    logger.warning(f"Skipping time point {t}: {str(e)}")
                    continue
            
            if not powers:
                raise HeatCalculationError("No valid power calculations in time range")
            
            powers = np.array(powers)
            
            # Calculate statistics
            average_power = np.mean(powers)
            total_energy = np.sum(powers) * self.time_step_seconds  # Energy in Joules
            peak_power = np.max(np.abs(powers))
            
            return {
                'average_power': float(average_power),
                'total_energy': float(total_energy), 
                'peak_power': float(peak_power),
                'power_profile': powers.tolist()
            }
            
        except Exception as e:
            raise HeatCalculationError(
                f"Failed to calculate time-averaged heat for {body_segment}: {str(e)}"
            ) from e
    
    def get_total_body_heat(self, time_index: Union[int, float]) -> Dict[str, Union[float, Dict, List]]:
        """
        Calculate total external heating/cooling for entire body
        
        Returns:
            Dictionary with segment_heat, total_heat, heating_segments, cooling_segments
        """
        try:
            results = {
                'segment_heat': {},
                'total_heat': 0.0,
                'heating_segments': [],
                'cooling_segments': [],
                'time_index': time_index
            }
            
            for segment in self.segments:
                try:
                    q_ext = self.calculate_instantaneous_heat(time_index, segment)
                    results['segment_heat'][segment] = q_ext
                    results['total_heat'] += q_ext
                    
                    if q_ext > 0.1:  # Small threshold to avoid noise
                        results['heating_segments'].append(segment)
                    elif q_ext < -0.1:
                        results['cooling_segments'].append(segment)
                        
                except Exception as e:
                    logger.warning(f"Skipping segment {segment}: {str(e)}")
                    results['segment_heat'][segment] = 0.0
            
            return results
            
        except Exception as e:
            raise HeatCalculationError(f"Failed to calculate total body heat: {str(e)}") from e
    
    def validate_energy_balance(self, time_index: Union[int, float], 
                              tolerance: float = 5.0) -> Dict[str, Union[bool, float, str]]:
        """
        Validate energy conservation for entire body at specific time point
        
        Args:
            time_index: Time point to validate
            tolerance: Maximum allowed imbalance in Watts
            
        Returns:
            Dictionary with validation results
        """
        try:
            timestep_data = self.parser.get_timestep_data(time_index)
            
            # Total metabolic heat production
            total_metabolic = timestep_data.get('Met', 0.0)
            
            # Total heat losses
            total_sensible = sum(timestep_data.get(f'SHLsk{seg}', 0.0) for seg in self.segments)
            total_latent = sum(timestep_data.get(f'LHLsk{seg}', 0.0) for seg in self.segments)
            total_respiratory = timestep_data.get('RES', 0.0)
            
            # Calculate total heat storage rate
            total_storage = 0.0
            for segment in self.segments:
                try:
                    storage = self._calculate_heat_storage_rate(time_index, segment)
                    total_storage += storage
                except:
                    continue
            
            # Energy balance: Met = Losses + Storage + External
            total_losses = total_sensible + total_latent + total_respiratory
            calculated_external = total_metabolic - total_losses - total_storage
            
            # Get calculated external heat from our method
            body_heat = self.get_total_body_heat(time_index)
            measured_external = body_heat['total_heat']
            
            # Check balance
            imbalance = abs(calculated_external - measured_external)
            is_balanced = imbalance <= tolerance
            
            return {
                'is_balanced': is_balanced,
                'imbalance_watts': imbalance,
                'total_metabolic': total_metabolic,
                'total_losses': total_losses,
                'total_storage': total_storage,
                'calculated_external': calculated_external,
                'measured_external': measured_external,
                'message': f"Energy balance {'OK' if is_balanced else 'FAILED'}: "
                          f"imbalance = {imbalance:.2f}W"
            }
            
        except Exception as e:
            return {
                'is_balanced': False,
                'imbalance_watts': float('inf'),
                'message': f"Validation failed: {str(e)}"
            }
    
    def get_heating_cooling_summary(self, time_range: Optional[Tuple[Union[int, float], Union[int, float]]] = None) -> Dict:
        """
        Generate summary of heating/cooling requirements over time range
        
        Args:
            time_range: Tuple of (start_time, end_time), or None for entire simulation
            
        Returns:
            Dictionary with comprehensive heating/cooling analysis
        """
        if time_range is None:
            time_start, time_end = 0, len(self.data) - 1
        else:
            time_start, time_end = time_range
        
        summary = {
            'time_range': (time_start, time_end),
            'segment_analysis': {},
            'whole_body': {
                'peak_heating': 0.0,
                'peak_cooling': 0.0,
                'average_net': 0.0
            },
            'critical_segments': {
                'highest_heating': '',
                'highest_cooling': ''
            }
        }
        
        # Analyze each segment
        for segment in self.segments:
            try:
                segment_stats = self.calculate_time_averaged_heat(time_start, time_end, segment)
                
                summary['segment_analysis'][segment] = {
                    'average_power': segment_stats['average_power'],
                    'peak_power': segment_stats['peak_power'],
                    'total_energy': segment_stats['total_energy'],
                    'requires_heating': segment_stats['average_power'] > 0,
                    'requires_cooling': segment_stats['average_power'] < 0
                }
                
                # Track extremes
                avg_power = segment_stats['average_power']
                if avg_power > summary['whole_body']['peak_heating']:
                    summary['whole_body']['peak_heating'] = avg_power
                    summary['critical_segments']['highest_heating'] = segment
                
                if avg_power < summary['whole_body']['peak_cooling']:
                    summary['whole_body']['peak_cooling'] = avg_power
                    summary['critical_segments']['highest_cooling'] = segment
                    
            except Exception as e:
                logger.warning(f"Could not analyze segment {segment}: {str(e)}")
                continue
        
        # Calculate whole body average
        segment_averages = [stats['average_power'] for stats in summary['segment_analysis'].values()]
        if segment_averages:
            summary['whole_body']['average_net'] = sum(segment_averages)
        
        return summary
    
    def get_conductive_heat_summary(self, time_index: Union[int, float]) -> Dict[str, Union[float, Dict, List]]:
        """
        Get summary of conductive heat transfer for all segments (NEW)
        
        Args:
            time_index: Time point to analyze
            
        Returns:
            Dictionary with conductive heat transfer analysis
        """
        try:
            # Get conductive data from parser
            conductive_data = self.parser.get_conductive_data(time_index)
            
            summary = {
                'time_index': time_index,
                'segments_in_contact': [],
                'segments_cooling': [],
                'segments_heating': [],
                'total_conductive_heat': 0.0,
                'segment_details': {},
                'contact_statistics': {
                    'total_segments_with_contact': 0,
                    'average_contact_area': 0.0,
                    'temperature_range': {'min': float('inf'), 'max': float('-inf')}
                }
            }
            
            contact_areas = []
            material_temps = []
            
            for segment, data in conductive_data.items():
                heat_transfer = data['heat_transfer']
                material_temp = data['material_temperature']
                contact_area = data['contact_area']
                contact_resistance = data['contact_resistance']
                
                # Store segment details
                summary['segment_details'][segment] = {
                    'heat_transfer': heat_transfer,
                    'material_temperature': material_temp,
                    'contact_area': contact_area,
                    'contact_resistance': contact_resistance,
                    'has_contact': contact_area > 0 and not np.isnan(material_temp)
                }
                
                # Check for contact
                if contact_area > 0 and not np.isnan(material_temp):
                    summary['segments_in_contact'].append(segment)
                    contact_areas.append(contact_area)
                    material_temps.append(material_temp)
                    
                    # Classify heating vs cooling
                    if heat_transfer > 0.1:  # Heating
                        summary['segments_heating'].append(segment)
                    elif heat_transfer < -0.1:  # Cooling
                        summary['segments_cooling'].append(segment)
                
                # Add to total
                summary['total_conductive_heat'] += heat_transfer
            
            # Calculate contact statistics
            if contact_areas:
                summary['contact_statistics']['total_segments_with_contact'] = len(contact_areas)
                summary['contact_statistics']['average_contact_area'] = np.mean(contact_areas)
                summary['contact_statistics']['temperature_range']['min'] = np.min(material_temps)
                summary['contact_statistics']['temperature_range']['max'] = np.max(material_temps)
            else:
                summary['contact_statistics']['temperature_range'] = {'min': 0, 'max': 0}
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get conductive heat summary: {str(e)}")
            return {
                'error': str(e),
                'time_index': time_index,
                'segments_in_contact': [],
                'total_conductive_heat': 0.0
            }