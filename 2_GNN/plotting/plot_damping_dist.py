import matplotlib.pyplot as plt
import numpy as np
import os

def plot_damping_distribution(estimated_b, true_b):
    """
    Plots a histogram of the estimated damping coefficients and compares
    it to the true value.
    
    Args:
        estimated_b (np.array): Array of damping coefficients predicted by the GNN.
        true_b (float): The single true physical damping coefficient.
    """
 # --- Start of Added Code ---
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist
    save_path = os.path.join(output_dir, 'damping_distribution.png')
    # --- End of Added Code ---

    mean_b = np.mean(estimated_b)
    std_b = np.std(estimated_b)
    
    plt.figure(figsize=(10, 6))
    plt.hist(estimated_b, bins=30, density=True, alpha=0.7, label='GNN Estimated "b"')
    
    plt.axvline(true_b, color='red', linestyle='--', linewidth=2, label=f'True b = {true_b:.2f}')
    plt.axvline(mean_b, color='green', linestyle=':', linewidth=2, label=f'Mean Estimated b = {mean_b:.2f}')
    
    plt.title('Distribution of Estimated Damping Coefficient "b"')
    plt.xlabel('Damping Coefficient (Nms/rad)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    text_str = f'Mean: {mean_b:.3f}\nStd Dev: {std_b:.3f}'
    plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
             
    plt.tight_layout()

    print(f"Saving damping distribution plot to: {save_path}") # <-- Added print statement
    plt.savefig(save_path) # Use the full path
    plt.show()

# --- Example Usage ---
# After getting the data in main.py:
#
# # Example data (replace with your actual data)
# example_true_b = 0.7
# # Let's pretend the model learned an "effective" parameter around 0.9
# example_estimated_b = np.random.normal(loc=0.9, scale=0.15, size=500)
#
# plot_damping_distribution(example_estimated_b, example_true_b)