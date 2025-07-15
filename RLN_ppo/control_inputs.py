import numpy as np


def apply_derivatives(steering_angle, velocity, steering_rate, acceleration, time_delta):
    """
    Apply derivatives to obtain new state values.
    
    Args:
        steering_angle (float): Current steering angle in radians
        velocity (float): Current velocity in m/s
        steering_rate (float): Desired steering rate in rad/s
        acceleration (float): Desired acceleration in m/s²
        time_delta (float): Time step in seconds
        
    Returns:
        tuple: (new_steering_angle, new_velocity)
    """
    new_steering_angle = steering_angle + steering_rate * time_delta
    new_velocity = velocity + acceleration * time_delta
    
    return new_steering_angle, new_velocity


def compute_derivatives(prev_steering, current_steering, prev_velocity, current_velocity, time_delta):
    """
    Compute derivatives from current and previous states.
    
    Args:
        prev_steering (float): Previous steering angle in radians
        current_steering (float): Current steering angle in radians
        prev_velocity (float): Previous velocity in m/s
        current_velocity (float): Current velocity in m/s
        time_delta (float): Time step in seconds
        
    Returns:
        tuple: (steering_rate, acceleration)
    """
    if time_delta <= 0:
        return 0.0, 0.0
        
    steering_rate = (current_steering - prev_steering) / time_delta
    acceleration = (current_velocity - prev_velocity) / time_delta
    
    return steering_rate, acceleration


def limit_steering_angle(steering_angle, max_angle=0.4):
    """
    Limit steering angle to maximum value.
    
    Args:
        steering_angle (float): Steering angle in radians
        max_angle (float): Maximum allowed steering angle in radians
        
    Returns:
        float: Limited steering angle
    """
    return np.clip(steering_angle, -max_angle, max_angle)


def limit_acceleration(acceleration, max_accel=3.0, max_decel=-3.0):
    """
    Limit acceleration to maximum values.
    
    Args:
        acceleration (float): Acceleration in m/s²
        max_accel (float): Maximum allowed positive acceleration in m/s²
        max_decel (float): Maximum allowed negative acceleration in m/s²
        
    Returns:
        float: Limited acceleration
    """
    return np.clip(acceleration, max_decel, max_accel)


def limit_steering_rate(steering_rate, max_rate=1.0):
    """
    Limit steering rate to maximum value.
    
    Args:
        steering_rate (float): Steering rate in rad/s
        max_rate (float): Maximum allowed steering rate in rad/s
        
    Returns:
        float: Limited steering rate
    """
    return np.clip(steering_rate, -max_rate, max_rate)


def smooth_controls(target_steering, target_velocity, current_steering, current_velocity, 
                   max_steering_rate=0.5, max_accel=1.0, max_decel=-2.0, time_delta=0.01):
    """
    Smooth control inputs based on rate limits.
    
    Args:
        target_steering (float): Target steering angle in radians
        target_velocity (float): Target velocity in m/s
        current_steering (float): Current steering angle in radians
        current_velocity (float): Current velocity in m/s
        max_steering_rate (float): Maximum steering rate in rad/s
        max_accel (float): Maximum acceleration in m/s²
        max_decel (float): Maximum deceleration in m/s²
        time_delta (float): Time step in seconds
        
    Returns:
        tuple: (new_steering, new_velocity)
    """
    # Calculate required rates
    desired_steering_rate = (target_steering - current_steering) / time_delta
    desired_accel = (target_velocity - current_velocity) / time_delta
    
    # Limit rates
    limited_steering_rate = limit_steering_rate(desired_steering_rate, max_steering_rate)
    limited_accel = np.clip(desired_accel, max_decel, max_accel)
    
    # Apply limited changes
    new_steering = current_steering + limited_steering_rate * time_delta
    new_velocity = current_velocity + limited_accel * time_delta
    
    return new_steering, new_velocity