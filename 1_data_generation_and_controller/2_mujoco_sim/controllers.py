import numpy as np

class Controller:
    """Base controller class."""
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
    def get_action(self):
        """Get control action. To be implemented by subclasses."""
        raise NotImplementedError

class RandomController(Controller):
    """Controller that generates random actions."""
    def __init__(self, model, data, action_range):
        super().__init__(model, data)
        self.action_range = action_range
        
    def get_action(self):
        """Generate random torques within the specified range."""
        return np.random.uniform(
            low=self.action_range[0],
            high=self.action_range[1],
            size=self.model.nu
        )

class ConstantController(Controller):
    """Controller that applies constant torques."""
    def __init__(self, model, data, constant_torque):
        super().__init__(model, data)
        self.constant_torque = constant_torque
        
    def get_action(self):
        """Return constant torque for all joints."""
        return np.full(self.model.nu, self.constant_torque)

class PDController(Controller): # Currently this only supports regulation
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
    """Factory function to create controllers.
    
    Args:
        controller_type (str): Type of controller ("random", "constant", or "pd")
        model: MuJoCo model
        data: MuJoCo data
        **kwargs: Additional arguments for specific controllers
        
    Returns:
        Controller: Instance of the specified controller
    """
    controllers = {
        "random": RandomController,
        "constant": ConstantController,
        "pd": PDController
    }
    
    if controller_type not in controllers:
        raise ValueError(f"Unknown controller type: {controller_type}")
        
    return controllers[controller_type](model, data, **kwargs) 