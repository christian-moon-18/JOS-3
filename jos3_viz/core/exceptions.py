"""
Custom exception classes for JOS3 visualization

Implementation of error handling framework
Source: Agile Plan Sprint 1, Task 1.1.3
"""


class JOS3VizError(Exception):
    """Base exception class for JOS3 visualization errors"""
    pass


class DataParsingError(JOS3VizError):
    """Raised when JOS3 data cannot be parsed or is invalid"""
    pass


class DataValidationError(JOS3VizError):
    """Raised when JOS3 data fails validation checks"""
    pass


class ConfigurationError(JOS3VizError):
    """Raised when configuration files are invalid or missing"""
    pass


class VisualizationError(JOS3VizError):
    """Raised when visualization rendering fails"""
    pass


class ExportError(JOS3VizError):
    """Raised when export operations fail"""
    pass


class HeatCalculationError(JOS3VizError):
    """Raised when heat calculations fail or produce invalid results"""
    pass


class AnthropometricError(JOS3VizError):
    """Raised when anthropometric scaling fails"""
    pass