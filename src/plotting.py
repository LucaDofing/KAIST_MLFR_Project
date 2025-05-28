# src/plotting.py
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

def plot_losses(train_losses, test_losses, save_path="loss_curve.png"):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    
    test_eval_indices = [i for i, (current, prev) in enumerate(zip(test_losses, [None] + test_losses[:-1])) if current != prev or i == 0 or i == len(test_losses)-1]
    actual_test_epochs = [epochs[i] for i in test_eval_indices if i < len(epochs)] # Boundary check
    actual_test_losses = [test_losses[i] for i in test_eval_indices if i < len(epochs)]

    if actual_test_epochs and actual_test_losses : # Ensure not empty before plotting
        plt.plot(actual_test_epochs, actual_test_losses, label='Test Loss', marker='o', linestyle='--')
    
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Test Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Loss curve saved to {save_path}")
    plt.close()

def plot_trajectory_comparison(model, dataset, device, simulate_step_fn, num_trajectories=3, steps_to_plot=50, save_dir="plots/"):
    if not dataset or len(dataset) == 0: # Check if dataset is None or empty
        print("Dataset is empty or None, cannot plot trajectories.")
        return
    
    if len(dataset) < steps_to_plot :
        print(f"Warning: Dataset smaller ({len(dataset)} samples) than steps_to_plot ({steps_to_plot}). Plotting {max(0, len(dataset)-1)} steps.")
        steps_to_plot = max(1, len(dataset) -1) 
        if steps_to_plot <= 0:
            print("Not enough data for any trajectory plot.")
            return

    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_trajectories):
        if len(dataset) < i + steps_to_plot : 
            print(f"Skipping trajectory {i+1}, not enough data points remaining.")
            break
        
        start_idx = np.random.randint(0, max(1, len(dataset) - steps_to_plot + 1))

        # Accessing data correctly whether it's a full InMemoryDataset or a Subset
        if hasattr(dataset, 'dataset') and isinstance(dataset.indices, (list, np.ndarray)) :
            original_start_index = dataset.indices[start_idx]
            initial_data_obj = dataset.dataset.get(original_start_index)
        else:
            initial_data_obj = dataset.get(start_idx)
        
        file_origin = getattr(initial_data_obj, 'file_origin', f"UnknownFile_traj{i+1}")
        start_step_in_file = getattr(initial_data_obj, 'step_index_in_file', torch.tensor(0)).item()
        
        current_x_true_for_sim = initial_data_obj.x.clone().to(device)
        dt = initial_data_obj.dt_step.item()

        true_thetas, true_omegas = [current_x_true_for_sim[0, 0].item()], [current_x_true_for_sim[0, 1].item()]
        pred_thetas, pred_omegas = [current_x_true_for_sim[0, 0].item()], [current_x_true_for_sim[0, 1].item()]
        
        actual_steps_collected = 1
        for step_offset in range(1, steps_to_plot):
            next_idx_in_dataset = start_idx + step_offset
            if next_idx_in_dataset >= len(dataset): break

            next_data_item_wrapper = dataset[next_idx_in_dataset]
            if hasattr(dataset, 'dataset') and isinstance(dataset.indices, (list, np.ndarray)):
                original_next_index = dataset.indices[next_idx_in_dataset]
                next_data_obj = dataset.dataset.get(original_next_index)
            else:
                next_data_obj = dataset.get(next_idx_in_dataset)

            if getattr(next_data_obj, 'file_origin', None) == file_origin and \
               getattr(next_data_obj, 'step_index_in_file', torch.tensor(-1)).item() == (start_step_in_file + step_offset):
                true_thetas.append(next_data_obj.x[0, 0].item())
                true_omegas.append(next_data_obj.x[0, 1].item())
                actual_steps_collected += 1
            else: break
        
        current_x_pred = initial_data_obj.x.clone().to(device)
        with torch.no_grad():
            for _ in range(actual_steps_collected - 1):
                temp_data_for_model = initial_data_obj.clone().to(device) # For edge_index etc.
                temp_data_for_model.x = current_x_pred
                estimated_param = model(temp_data_for_model)
                current_x_pred = simulate_step_fn(x=current_x_pred, estimated_damping=estimated_param, dt=dt)
                pred_thetas.append(current_x_pred[0, 0].item())
                pred_omegas.append(current_x_pred[0, 1].item())

        time_axis = np.arange(actual_steps_collected) * dt
        plt.figure(figsize=(14, 7))
        plt.subplot(1, 2, 1)
        plt.plot(time_axis, true_thetas, 'b-', marker='.', linewidth=2, label='True Theta')
        plt.plot(time_axis, pred_thetas, 'r--', marker='x', linewidth=2, label='Predicted Theta (Effective Damping Model)')
        plt.xlabel('Time (s)'); plt.ylabel('Theta (rad)')
        plt.title(f'Theta - Traj {i+1} ({file_origin}, Start: {start_step_in_file})')
        plt.legend(); plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(time_axis, true_omegas, 'b-', marker='.', linewidth=2, label='True Omega')
        plt.plot(time_axis, pred_omegas, 'r--', marker='x', linewidth=2, label='Predicted Omega (Effective Damping Model)')
        plt.xlabel('Time (s)'); plt.ylabel('Omega (rad/s)')
        plt.title(f'Omega - Traj {i+1}')
        plt.legend(); plt.grid(True)
        plt.tight_layout()
        safe_file_origin = "".join(c if c.isalnum() else "_ " for c in os.path.splitext(file_origin)[0])
        plot_filename = os.path.join(save_dir, f"traj_eff_damp_{i+1}_{safe_file_origin}_step{start_step_in_file}.png")
        plt.savefig(plot_filename)
        print(f"Trajectory plot saved: {plot_filename}")
        plt.close()