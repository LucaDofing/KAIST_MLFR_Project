import matplotlib.pyplot as plt
import numpy as np
import os
def plot_prediction_scatter(pred_next, true_next):
    """
    Creates scatter plots comparing predicted vs. true next states.
    
    Args:
        pred_next (np.array): Array of shape [N, 2] with predicted [theta, omega].
        true_next (np.array): Array of shape [N, 2] with true [theta, omega].
    """
     # --- Start of Added Code ---
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist
    save_path = os.path.join(output_dir, 'prediction_scatter.png')
    # --- End of Added Code ---

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot for theta_next
    ax1.scatter(true_next[:, 0], pred_next[:, 0], alpha=0.3, label='Predictions')
    min_val_th = min(ax1.get_xlim()[0], ax1.get_ylim()[0])
    max_val_th = max(ax1.get_xlim()[1], ax1.get_ylim()[1])
    ax1.plot([min_val_th, max_val_th], [min_val_th, max_val_th], 'r--', label='Perfect Prediction (y=x)')
    ax1.set_xlabel('True Next Theta (rad)')
    ax1.set_ylabel('Predicted Next Theta (rad)')
    ax1.set_title('Next Angle (θ_t+1) Prediction')
    ax1.grid(True)
    ax1.legend()
    ax1.set_aspect('equal', 'box')

    # Plot for omega_next
    ax2.scatter(true_next[:, 1], pred_next[:, 1], alpha=0.3, label='Predictions')
    min_val_om = min(ax2.get_xlim()[0], ax2.get_ylim()[0])
    max_val_om = max(ax2.get_xlim()[1], ax2.get_ylim()[1])
    ax2.plot([min_val_om, max_val_om], [min_val_om, max_val_om], 'r--', label='Perfect Prediction (y=x)')
    ax2.set_xlabel('True Next Omega (rad/s)')
    ax2.set_ylabel('Predicted Next Omega (rad/s)')
    ax2.set_title('Next Angular Velocity (ω_t+1) Prediction')
    ax2.grid(True)
    ax2.legend()
    ax2.set_aspect('equal', 'box')

    fig.tight_layout()
    
    print(f"Saving prediction scatter plot to: {save_path}") # <-- Added print statement
    plt.savefig(save_path) # Use the full path
    plt.show()

# --- Example Usage ---
# After getting the data in main.py
#
# # Example data (replace with your actual data)
# N_SAMPLES = 1000
# true_data = np.random.randn(N_SAMPLES, 2) * np.array([np.pi, 10])
# # Add some noise and bias to simulate an imperfect model
# pred_data = true_data * 0.95 + np.random.randn(N_SAMPLES, 2) * np.array([0.1, 0.5])
#
# plot_prediction_scatter(pred_data, true_data)