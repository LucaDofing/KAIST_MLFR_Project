import numpy as np
import mujoco

class Controller:
    """Base controller class."""
    def __init__(self, model, data):
        self.model = model
        self.data = data
    
    def get_action(self):
        """Get control action. To be implemented by subclasses."""
        raise NotImplementedError

class RandomController(Controller):
    """Random controller that applies random torques."""
    def __init__(self, model, data):
        super().__init__(model, data)
        self.torque_range = (-5.0, 5.0)  # Range of random torques
    
    def get_action(self):
        """Generate random control action."""
        return np.random.uniform(self.torque_range[0], self.torque_range[1], size=self.model.nu)

class ConstantController(Controller):
    """Constant controller that applies fixed torques."""
    def __init__(self, model, data, constant_torque=0.5):
        super().__init__(model, data)
        self.constant_torque = constant_torque
    
    def get_action(self):
        """Return constant control action."""
        return np.full(self.model.nu, self.constant_torque)

class PDController(Controller):
    """PD (Proportional-Derivative) controller."""
    def __init__(self, model, data, target_angle, kp, kd):
        super().__init__(model, data)
        self.target_angle = target_angle
        self.kp = kp  # Position gain
        self.kd = kd  # Velocity gain
        
    def get_action(self):
        """Calculate PD control action."""
        # Get current joint positions and velocities
        current_pos = self.data.qpos[:self.model.nu]
        current_vel = self.data.qvel[:self.model.nu]
        
        # Calculate position and velocity errors
        pos_error = self.target_angle - current_pos
        vel_error = 0.0 - current_vel  # Target velocity is 0
        
        # Calculate control torques using PD control law
        torques = self.kp * pos_error + self.kd * vel_error
        
        return torques

def create_controller(controller_type, model, data, **kwargs):
    """Factory function to create controllers."""
    if controller_type == "random":
        return RandomController(model, data)
    elif controller_type == "constant":
        return ConstantController(model, data, kwargs.get("constant_torque", 0.5))
    elif controller_type == "pd":
        return PDController(model, data,
                          kwargs.get("target_angle", 0.0),
                          kwargs.get("kp", 3.0),
                          kwargs.get("kd", 0.01))
    else:
        raise ValueError(f"Unknown controller type: {controller_type}") 