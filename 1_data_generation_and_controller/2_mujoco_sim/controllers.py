#!/usr/bin/env python3
"""
Control Systems for N-Link Robot Simulation

This module provides various control strategies for the n-link robot simulation:
- Random Controller: Applies random torques for exploration
- Constant Controller: Applies fixed torques
- PD Controller: Proportional-Derivative control for position tracking

Author: KAIST MLFR Project
"""

from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional
import numpy as np
import mujoco


class Controller(ABC):
    """
    Abstract base class for robot controllers.
    
    All controllers must implement the get_action method to provide
    control torques for the robot joints.
    """
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """
        Initialize the controller.
        
        Args:
            model: MuJoCo model object
            data: MuJoCo data object
        """
        self.model = model
        self.data = data
        self.num_actuators = model.nu
        
        if self.num_actuators <= 0:
            raise ValueError("Model must have at least one actuator")
    
    @abstractmethod
    def get_action(self) -> np.ndarray:
        """
        Get control action for the current state.
        
        Returns:
            Array of control torques for each actuator
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement get_action method")
    
    def clip_torques(self, torques: np.ndarray) -> np.ndarray:
        """
        Clip torques to actuator limits.
        
        Args:
            torques: Raw control torques
            
        Returns:
            Clipped torques within actuator limits
        """
        if len(self.model.actuator_ctrlrange) > 0:
            ctrl_range = self.model.actuator_ctrlrange
            torques = np.clip(torques, ctrl_range[:, 0], ctrl_range[:, 1])
        
        return torques


class RandomController(Controller):
    """
    Random controller that applies random torques within specified bounds.
    
    Useful for system identification, exploration, and generating diverse
    training data for machine learning applications.
    """
    
    def __init__(
        self, 
        model: mujoco.MjModel, 
        data: mujoco.MjData, 
        torque_range: Tuple[float, float] = (-5.0, 5.0),
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize random controller.
        
        Args:
            model: MuJoCo model object
            data: MuJoCo data object
            torque_range: Tuple of (min_torque, max_torque) in N⋅m
            seed: Random seed for reproducibility (optional)
        """
        super().__init__(model, data)
        
        if torque_range[0] >= torque_range[1]:
            raise ValueError("Invalid torque range: min must be less than max")
        
        self.torque_range = torque_range
        
        if seed is not None:
            np.random.seed(seed)
    
    def get_action(self) -> np.ndarray:
        """
        Generate random control action.
        
        Returns:
            Array of random torques within the specified range
        """
        torques = np.random.uniform(
            self.torque_range[0], 
            self.torque_range[1], 
            size=self.num_actuators
        )
        
        return self.clip_torques(torques)


class ConstantController(Controller):
    """
    Constant controller that applies fixed torques to all joints.
    
    Useful for testing steady-state behavior, gravity compensation,
    and simple open-loop control scenarios.
    """
    
    def __init__(
        self, 
        model: mujoco.MjModel, 
        data: mujoco.MjData, 
        constant_torque: Union[float, np.ndarray] = 0.5
    ) -> None:
        """
        Initialize constant controller.
        
        Args:
            model: MuJoCo model object
            data: MuJoCo data object
            constant_torque: Constant torque value(s) in N⋅m
                           Can be a scalar (applied to all joints) or array (per joint)
        """
        super().__init__(model, data)
        
        # Handle both scalar and array inputs
        if np.isscalar(constant_torque):
            self.constant_torque = np.full(self.num_actuators, constant_torque, dtype=float)
        else:
            constant_torque = np.asarray(constant_torque, dtype=float)
            if len(constant_torque) != self.num_actuators:
                raise ValueError(
                    f"Torque array length ({len(constant_torque)}) must match "
                    f"number of actuators ({self.num_actuators})"
                )
            self.constant_torque = constant_torque
    
    def get_action(self) -> np.ndarray:
        """
        Return constant control action.
        
        Returns:
            Array of constant torques
        """
        return self.clip_torques(self.constant_torque.copy())


class PDController(Controller):
    """
    PD (Proportional-Derivative) controller for position tracking.
    
    Implements the control law: τ = Kp * (θ_target - θ) + Kd * (ω_target - ω)
    where typically ω_target = 0 for position regulation.
    """
    
    def __init__(
        self, 
        model: mujoco.MjModel, 
        data: mujoco.MjData, 
        target_angle: Union[float, np.ndarray], 
        kp: Union[float, np.ndarray], 
        kd: Union[float, np.ndarray],
        target_velocity: Union[float, np.ndarray] = 0.0
    ) -> None:
        """
        Initialize PD controller.
        
        Args:
            model: MuJoCo model object
            data: MuJoCo data object
            target_angle: Target joint angle(s) in radians
            kp: Proportional gain(s)
            kd: Derivative gain(s)
            target_velocity: Target joint velocity(ies) in rad/s (default: 0.0)
        """
        super().__init__(model, data)
        
        # Convert inputs to arrays and validate dimensions
        self.target_angle = self._validate_and_convert_param(target_angle, "target_angle")
        self.kp = self._validate_and_convert_param(kp, "kp")
        self.kd = self._validate_and_convert_param(kd, "kd")
        self.target_velocity = self._validate_and_convert_param(target_velocity, "target_velocity")
        
        # Validate gains are positive
        if np.any(self.kp < 0) or np.any(self.kd < 0):
            raise ValueError("PD gains (kp, kd) must be non-negative")
    
    def _validate_and_convert_param(
        self, 
        param: Union[float, np.ndarray], 
        param_name: str
    ) -> np.ndarray:
        """
        Validate and convert parameter to appropriate array format.
        
        Args:
            param: Parameter value(s)
            param_name: Name of parameter for error messages
            
        Returns:
            Parameter as numpy array with correct dimensions
            
        Raises:
            ValueError: If parameter dimensions don't match number of actuators
        """
        if np.isscalar(param):
            return np.full(self.num_actuators, param, dtype=float)
        else:
            param = np.asarray(param, dtype=float)
            if len(param) != self.num_actuators:
                raise ValueError(
                    f"{param_name} array length ({len(param)}) must match "
                    f"number of actuators ({self.num_actuators})"
                )
            return param
    
    def get_action(self) -> np.ndarray:
        """
        Calculate PD control action.
        
        Returns:
            Array of PD control torques
        """
        # Get current joint positions and velocities
        current_pos = self.data.qpos[:self.num_actuators]
        current_vel = self.data.qvel[:self.num_actuators]
        
        # Calculate position and velocity errors
        pos_error = self.target_angle - current_pos
        vel_error = self.target_velocity - current_vel
        
        # Calculate control torques using PD control law
        torques = self.kp * pos_error + self.kd * vel_error
        
        return self.clip_torques(torques)
    
    def set_target(
        self, 
        target_angle: Union[float, np.ndarray], 
        target_velocity: Union[float, np.ndarray] = 0.0
    ) -> None:
        """
        Update target position and velocity.
        
        Args:
            target_angle: New target joint angle(s) in radians
            target_velocity: New target joint velocity(ies) in rad/s
        """
        self.target_angle = self._validate_and_convert_param(target_angle, "target_angle")
        self.target_velocity = self._validate_and_convert_param(target_velocity, "target_velocity")


def create_controller(
    controller_type: str, 
    model: mujoco.MjModel, 
    data: mujoco.MjData, 
    **kwargs
) -> Controller:
    """
    Factory function to create controllers.
    
    Args:
        controller_type: Type of controller ("random", "constant", "pd")
        model: MuJoCo model object
        data: MuJoCo data object
        **kwargs: Controller-specific parameters
        
    Returns:
        Initialized controller instance
        
    Raises:
        ValueError: If controller_type is unknown
        
    Example:
        >>> controller = create_controller("pd", model, data, 
        ...                               target_angle=1.0, kp=10.0, kd=0.1)
    """
    controller_type = controller_type.lower()
    
    if controller_type == "random":
        torque_range = kwargs.get("torque_range", (-5.0, 5.0))
        seed = kwargs.get("seed")
        return RandomController(model, data, torque_range=torque_range, seed=seed)
        
    elif controller_type == "constant":
        constant_torque = kwargs.get("constant_torque", 0.5)
        return ConstantController(model, data, constant_torque=constant_torque)
        
    elif controller_type == "pd":
        target_angle = kwargs.get("target_angle", 0.0)
        kp = kwargs.get("kp", 3.0)
        kd = kwargs.get("kd", 0.01)
        target_velocity = kwargs.get("target_velocity", 0.0)
        return PDController(model, data, target_angle, kp, kd, target_velocity)
        
    else:
        raise ValueError(
            f"Unknown controller type: {controller_type}. "
            f"Available types: 'random', 'constant', 'pd'"
        )