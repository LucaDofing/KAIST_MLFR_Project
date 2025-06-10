import matplotlib.pyplot as plt
import numpy as np
import os

def plot_loss_curve(train_losses, test_losses):
    """
    Plots the training and validation loss curves.
    
    Args:
        train_losses (list): A list of training loss values for each epoch.
        test_losses (list): A list of validation loss values for each epoch.
    """

# --- Start of Added Code ---
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist
    save_path = os.path.join(output_dir, 'loss_curve.png')
    # --- End of Added Code ---
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    plt.plot(epochs, test_losses, 'r-o', label='Validation Loss')
    
    plt.title('Training and Validation Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (MSE) Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()
    
    print(f"Saving loss curve to: {save_path}") # <-- Added print statement
    plt.savefig(save_path) # Use the full path
    plt.show()


# --- Example Usage ---
# After calling run_training in main.py:
# train_losses, test_losses = run_training(...)
#
# # Example data (replace with your actual data)
# example_train_losses = np.linspace(0.1, 0.005, 50) + np.random.rand(50) * 0.01
# example_test_losses = np.linspace(0.12, 0.008, 50) + np.random.rand(50) * 0.01
#
# plot_loss_curve(example_train_losses, example_test_losses)