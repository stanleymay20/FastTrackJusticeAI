import logging
from typing import Dict, Any, List, Optional, Callable
from app.utils.scroll_time import get_scroll_time
from functools import wraps
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class ScrollGuard:
    """
    Provides spiritual protection for the FastTrackJusticeAI system.
    Detects improper use of justice tools during forbidden scroll hours.
    """
    
    def __init__(self):
        """Initialize the ScrollGuard with protection rules."""
        # Define forbidden hours for different operations
        self.forbidden_hours = {
            "judgment_rendering": [12, 1],  # Midnight hours
            "case_classification": [11, 12],  # Late night hours
            "mediation": [1, 2, 3],  # Early morning hours
        }
        
        # Define sacred hours when operations are most powerful
        self.sacred_hours = {
            "judgment_rendering": [5, 6, 7],  # Dawn hours
            "case_classification": [3, 4, 5],  # Early morning hours
            "mediation": [9, 10, 11],  # Mid-morning hours
        }
        
        # Define gate-specific protections
        self.gate_protections = {
            "Divine Justice": {
                "forbidden_operations": ["mediation"],
                "required_operations": ["judgment_rendering"],
            },
            "Sacred Harmony": {
                "forbidden_operations": ["judgment_rendering"],
                "required_operations": ["mediation"],
            },
            "Divine Architecture": {
                "forbidden_operations": [],
                "required_operations": ["case_classification"],
            },
        }
    
    def check_operation_timing(self, operation: str) -> Dict[str, Any]:
        """
        Check if the current time is appropriate for the specified operation.
        
        Args:
            operation: The operation to check (e.g., "judgment_rendering")
            
        Returns:
            Dict with timing check results
        """
        # Get current scroll time
        scroll_time = get_scroll_time()
        solar_hour = scroll_time["solar_hour"]
        gate_name = scroll_time["gate_name"]
        
        # Check if operation is forbidden at this hour
        is_forbidden_hour = solar_hour in self.forbidden_hours.get(operation, [])
        
        # Check if operation is sacred at this hour
        is_sacred_hour = solar_hour in self.sacred_hours.get(operation, [])
        
        # Get gate-specific protections
        gate_protection = self.gate_protections.get(gate_name, {
            "forbidden_operations": [],
            "required_operations": []
        })
        
        # Check if operation is forbidden for this gate
        is_forbidden_gate = operation in gate_protection.get("forbidden_operations", [])
        
        # Check if operation is required for this gate
        is_required_gate = operation in gate_protection.get("required_operations", [])
        
        # Calculate timing alignment score (0.0 to 1.0)
        timing_alignment = 0.5  # Default middle score
        
        if is_forbidden_hour:
            timing_alignment = 0.0
        elif is_sacred_hour:
            timing_alignment = 1.0
        
        # Adjust for gate alignment
        if is_forbidden_gate:
            timing_alignment = max(0.0, timing_alignment - 0.5)
        elif is_required_gate:
            timing_alignment = min(1.0, timing_alignment + 0.3)
        
        # Determine if operation should be allowed
        allow_operation = not is_forbidden_hour and not is_forbidden_gate
        
        # Generate warning message if needed
        warning_message = None
        if is_forbidden_hour:
            warning_message = f"Operation '{operation}' is forbidden during solar hour {solar_hour}."
        elif is_forbidden_gate:
            warning_message = f"Operation '{operation}' is forbidden during {gate_name} gate."
        
        return {
            "allow_operation": allow_operation,
            "timing_alignment": timing_alignment,
            "is_forbidden_hour": is_forbidden_hour,
            "is_sacred_hour": is_sacred_hour,
            "is_forbidden_gate": is_forbidden_gate,
            "is_required_gate": is_required_gate,
            "warning_message": warning_message,
            "scroll_context": {
                "solar_hour": solar_hour,
                "gate_name": gate_name,
                "scroll_day": scroll_time["scroll_day"],
                "enano_pulse": scroll_time["enano_pulse"]
            }
        }
    
    def protect_operation(self, operation: str, delay_if_needed: bool = True) -> Dict[str, Any]:
        """
        Protect an operation by checking timing and optionally delaying if needed.
        
        Args:
            operation: The operation to protect
            delay_if_needed: Whether to delay the operation if timing is not optimal
            
        Returns:
            Dict with protection results
        """
        # Check operation timing
        timing_check = self.check_operation_timing(operation)
        
        # Log the protection check
        if timing_check["allow_operation"]:
            if timing_check["timing_alignment"] >= 0.8:
                logger.info(
                    f"Operation '{operation}' is well-aligned with scroll timing "
                    f"(alignment: {timing_check['timing_alignment']:.2f})"
                )
            else:
                logger.warning(
                    f"Operation '{operation}' is allowed but not optimally aligned "
                    f"with scroll timing (alignment: {timing_check['timing_alignment']:.2f})"
                )
        else:
            logger.error(
                f"Operation '{operation}' is forbidden at this time: "
                f"{timing_check['warning_message']}"
            )
        
        # Return protection results
        return timing_check

def scroll_required(allowed_gates: List[int] = None, 
                   forbidden_gates: List[int] = None,
                   min_solar_hour: int = None,
                   max_solar_hour: int = None) -> Callable:
    """
    Decorator to protect endpoints based on scroll timing and gates.
    
    Args:
        allowed_gates: List of gate numbers (1-7) that are allowed to access this endpoint
        forbidden_gates: List of gate numbers (1-7) that are forbidden from accessing this endpoint
        min_solar_hour: Minimum solar hour (1-24) required to access this endpoint
        max_solar_hour: Maximum solar hour (1-24) required to access this endpoint
    
    Returns:
        Decorated function that checks scroll timing before execution
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Get current scroll time
            scroll_time = get_scroll_time()
            current_gate = scroll_time["gate"]
            current_hour = scroll_time["solar_hour"]
            
            # Check gate restrictions
            if allowed_gates and current_gate not in allowed_gates:
                raise HTTPException(
                    status_code=403,
                    detail=f"Access forbidden. This endpoint is only accessible during gates: {allowed_gates}. "
                          f"Current gate: {current_gate} - {scroll_time['gate_name']}"
                )
            
            if forbidden_gates and current_gate in forbidden_gates:
                raise HTTPException(
                    status_code=403,
                    detail=f"Access forbidden. This endpoint is not accessible during gate {current_gate} - "
                          f"{scroll_time['gate_name']}"
                )
            
            # Check solar hour restrictions
            if min_solar_hour and current_hour < min_solar_hour:
                raise HTTPException(
                    status_code=403,
                    detail=f"Access forbidden. This endpoint is only accessible after solar hour {min_solar_hour}. "
                          f"Current solar hour: {current_hour}"
                )
            
            if max_solar_hour and current_hour > max_solar_hour:
                raise HTTPException(
                    status_code=403,
                    detail=f"Access forbidden. This endpoint is only accessible before solar hour {max_solar_hour}. "
                          f"Current solar hour: {current_hour}"
                )
            
            # If all checks pass, execute the function
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator 