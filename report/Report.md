# GNNSys Id: Learning Physical Parameters from Motion Using Graph Neural Networks for Robot Control
## 1. Introduction 

Robots frequently encounter unpredictable internal mechanics like friction and micro slip, which are hard to model and harm performance. Instead of manual calibration or external sensors, this project uses simulation based Graph Neural Networks to identify internal dynamics from motion data in a self supervised manner, efficiently addressing limited computational resources.



## 2. MuJoCo Simulation System

MuJoCo was chosen for its computational efficiency, broad use in the reinforcement learning community, and smooth integration with the JAX framework. Its performance and flexibility make it ideal for robotic simulation and trajectory generation, as noted in recent reviews ([Kaup et al., 2024](https://arxiv.org/pdf/2407.08590.pdf)).


### 2.1 Modular N-Link Architecture

The simulation framework employs a **parametric design approach** that automatically constructs n-link robotic manipulators based on configuration parameters. This modular system allows to:

- **Scalable Design**: Automatically generate 1-link to n-link configurations
- **Parameter Studies**: Systematically analyze how damping, friction link count etc. affects the trajectory

<div align="center">
<table>
<tr>
<td align="center"><img src="Mujocosim_1_links.png" width="400"><br><b>Single-Link Manipulator (1-DOF)</b></td>
<td align="center"><img src="Mujocosim_2_links.png" width="400"><br><b>Two-Link Manipulator (2-DOF)</b></td>
<td align="center"><img src="Mujocosim_3_links.png" width="400"><br><b>Three-Link Manipulator (3-DOF)</b></td>
</tr>
</table>
</div>


### 2.2 Parametric Design Apporach
With the model setup several parameters such as inertia, mass, damping, friciton, gear ratios, maximum applied torque can be defined.

### 2.3 Damping Effect Analysis
In the ater GNN our Goal was it to estimate the damping of the generated robot setup. To validate that a proper damping estimation is crucial for predicting the system response an identical physical 1-link simulaton was performed under varying damping parameters.

#### 2.3.1 Experimental Setup
The following specifications were used for the damping analysis:
**Physical Properties:**
- **Mass**: 3.0 kg (main link body)
- **Fingertip Mass**: 0.5 kg (end effector)
- **Link Length**: 0.3 m
- **Link Radius**: 0.075 m
- **Moment of Inertia**: 0.159 kg⋅m²

**Dynamic Parameters:**
- **Friction Coefficient**: 0.0 (frictionless joints)
- **Torque Limit**: 20.0 N⋅m (maximum actuator output)

**Simulation Configuration:**
- **Time Step (dt)**: 0.01 s
- **Gravity**: [0.0, 0.0, -9.81] m/s²
- **Solver**: Implicit integration method
- **Simulation Duration**: 200 steps (2.0 seconds)

**Control Parameters:**
- **Proportional Gain (Kp)**: 40.0
- **Derivative Gain (Kd)**: 0.1
- **Control Type**: PD Controller

#### 2.3.2 Results

The following figure illustrates how different damping values significantly influence the system's response and settling behavior:
<div align="center">
<table>
<tr>
<td align="center"><img src="damping_comparison.png" width="1500"><br><b>Damping Effect on 1-DOF Manipulator Trajectory</b></td>
</tr>
</table>
</div>
The trajectory comparison clearly demonstrates that:
- **Higher damping** results in more conservative, slower convergence with reduced overshoot
- **Lower damping** produces faster response but with increased oscillatory behavior
- **Intermediate damping** values provide balanced performance between speed and stability

#### 2.3.3 Data Generation Strategy

The Data Generation Pipeline used for this project can be split down into the following:

1. **Systematic Parameter Variation**: Generate varying trajectories the same physical setup (inertia, damping, etc.) with different initial conditions and PD controller values
2. **State Trajectory Recording**: Capture complete system state evolution (position, velocity, acceleration, torque)

### 2.4 Automatic Generation Pipeline

The system features an **automated XML generation pipeline** that:

1. **Parameter Input**: unique physical robot specification, varying controller and initial condition setup  
2. **XML Construction**: Automatically builds MuJoCo model description
3. **Mujoco Simulation**: Executes Simulations based on XML file and varying controller and initial condition setup 
4. **Data Logger**: Save the Trajectories and specifications as JSON files


**Code Architecture**:
```
Parameter Space → XML Generation → MuJoCo Engine → PD Control → Data Export
      ↓               ↓              ↓              ↓           ↓
   • n-links       • Dynamic      • Physics      • Torque    • JSON
   • Masses        • Geometry     • Integration  • Limits    • Metadata  
   • Damping       • Joints       • Constraints  • Tracking  • Trajectories
   • Controllers   • Actuators    • Rendering    • Response  • Parameters
```


### 2.5 Debugging

- **Flexible Render Modes**: Execute simulation in visual mode for trajectory inspection and physical plausibility validation, or headless mode for rapid automated testing and data generation
- **Accelerated Validation**: Speed up simulation playback to quickly assess trajectory behavior and validate parameter configurations across longer time horizons


### 2.6.Feature Work

#### 2.6.1 Intelligent Data Collection
- **Trajectory Quality Assessment**: Automatically evaluate data richness and truncate trajectories at steady-state
- **Richness Metrics**: Implement measures for state space coverage, frequency content, and control effort diversity

#### 2.6.2 Realistic Signal Modeling
- **Sensor Noise**: Add realistic jitter to position, velocity, and torque measurements
- **Communication Effects**: Include delays, sampling limitations, and actuator response characteristics
- **Signal Processing**: Model discrete-time effects and filtering artifacts

#### 2.6.3 Advanced Physical Models
- **Enhanced Damping**: Non-linear damping models
- **Comprehensive Friction**: Stiction, Stribeck Model
- **Gear Train Effects**: Backlash

#### 2.6.4 Data Pipeline Improvements
- **Active Learning**: Intelligent parameter space exploration
- **Real-time Monitoring**: Online trajectory quality assessment


### 3. GNN Design

This section details our approach to identifying physical parameters of robotic systems using Graph Neural Networks (GNNs). Our goal is to predict latent parameters, such as joint damping coefficients, by learning to predict the system's dynamic response in an unsupervised, end-to-end manner.

### 3.1 Introduction to GNNs for Robotic Systems
GNNs are a class of neural networks specifically designed to operate on data with graph structures. This makes them inherently suitable for robotic systems, particularly articulated robots, where joints and links form a natural graph.

#### 3.1.1 Rationale for Using GNNs
Robotic systems like manipulators or legged robots can be represented as graphs:
-   **Nodes:** Represent joints or links of the robot.
-   **Edges:** Represent the physical connections or kinematic constraints between these components.

The advantages of GNNs for modeling such systems include:
-   **Permutation Invariance/Equivariance:** GNNs can process graph nodes in any order, which is suitable for robotic systems where the labeling of components might be arbitrary.
-   **Variable Structure:** They can adapt to graphs of varying sizes and topologies, making them applicable to robots with different numbers of links or configurations.
-   **Parameter Sharing:** GNNs typically share weights across nodes and/or edges, promoting generalization and reducing the total number of learnable parameters, which is beneficial for data efficiency.
-   **Local Information Aggregation:** GNN layers iteratively aggregate information from neighboring nodes. This allows the model to learn how local physical interactions (e.g., forces and torques at a joint) propagate and influence the overall system dynamics.

These properties make GNNs a promising tool for system identification, where the goal is to understand the interplay of forces and motions across interconnected components based on observed data.

#### 3.1.2 Graph Convolutional Networks (GCNs)
In this project, we utilize Graph Convolutional Networks (GCNs) [^1], a popular and effective type of GNN. A GCN layer updates the feature vector of each node by aggregating feature vectors from its neighbors, followed by a linear transformation and a non-linear activation function. For an $L$-layer GCN, each node's final representation effectively incorporates information from its $L$-hop neighborhood, allowing it to learn complex relationships based on local and extended connectivity.


### 3.2 Methodology: GNN for Damping Coefficient Estimation

The primary objective of this GNN module is to estimate the physical parameters of a robotic system. For the scope of this project, we focused on estimating the viscous joint damping coefficient ($b$) of a 1-link pendulum system.

**Unsupervised Learning Paradigm**

We employ an unsupervised learning strategy. The GNN is not directly provided with the true damping values during training. Instead, it learns to infer these parameters implicitly by minimizing the error in predicting the system's next state. The process is as follows:

1.  **Input:** The GNN receives the current kinematic state of the system for each joint $i$: its angle $\theta_{t,i}$ and angular velocity $\omega_{t,i}$.
2.  **GNN Output:** The GNN processes these inputs and outputs an estimate of the latent physical parameter for each joint, in this case, the estimated damping coefficient $\hat{b}_i$.
3.  **Differentiable Simulation:** This estimated parameter $\hat{b}_i$ is then fed into a differentiable physics simulation step along with the current state $(\theta_{t,i}, \omega_{t,i})$ and other known physical properties (mass, inertia, gravity, etc.) to predict the next state $(\hat{\theta}_{t+1,i}, \hat{\omega}_{t+1,i})$.
4.  **Objective Function:** The GNN is trained by minimizing the Mean Squared Error (MSE) between this predicted next state and the true next state $(\theta_{t+1,i}, \omega_{t+1,i})$ observed from the dataset:
    ```math
    \mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \left\| [\hat{\theta}_{t+1,i}, \hat{\omega}_{t+1,i}] - [\theta_{t+1,i}, \omega_{t+1,i}] \right\|_2^2
    ```
    where the loss is computed over all samples in a batch.

This end-to-end differentiable pipeline enables the GNN to learn parameter values that best explain the observed system dynamics.

#### 3.2.2 Datasets

Realistic datasets were generated from MuJoCo simulations as described previously.

**MuJoCo Simulation Datasets (`MuJoCoPendulumDataset` in `src/datasets.py`)**

These datasets provide realistic dynamics for a 1-link pendulum.
-   **JSON Parsing:** The `MuJoCoPendulumDataset` class parses JSON files containing metadata ($\Delta t$, gravity), static properties (mass $m$, link length $L$, physical damping $b_{true}$, inertia $I_{yy}$), and time-series data for $\theta, \omega, \alpha$ (angular acceleration), and applied torque $\tau_{app}$.
-   **Data Object Construction:** Each time step transition $t \rightarrow t+1$ from a simulation trajectory forms a `torch_geometric.data.Data` object with the following key attributes:
    -   `data.x = [\theta_t, \omega_t]`: The current state, which serves as the direct input to the GNN model.
    -   `data.x_next = [\theta_{t+1}, \omega_{t+1}]`: The ground-truth next state, used as the target for the loss function.
    -   `data.y_true_damping`: The true physical damping coefficient $b_{true}$ from the JSON file, stored for evaluation purposes only.
    -   **Physics Parameters:** Other parameters required for the differentiable simulation are also stored in the `Data` object, including `data.true_torque_t`, `data.true_mass`, `data.true_length`, `data.inertia_yy`, `data.gravity_accel`, and `data.dt_step`. Note that `data.length_com_for_gravity` is set equal to `data.true_length` for this setup.

#### 3.2.3 GNN Model Architecture (`DampingGCN` in `src/models.py`)

The GNN model, `DampingGCN`, is designed to predict the damping coefficient $\hat{b}$ for each joint.

-   **Input Features:** The model uses the raw kinematic state as direct input. The input features for each node are simply its angle and angular velocity: $[\theta_t, \omega_t]$. Therefore, the model's `INPUT_DIM` is **2**, as defined in `src/config.py`.
-   **Graph Convolutional Layers:** The model uses `NUM_LAYERS = 3` GCN layers (`torch_geometric.nn.GCNConv`) with `HIDDEN_DIM = 64` hidden units.
-   **Activation Functions:** ReLU activation (`torch.relu`) is applied after the intermediate GCN layers.
-   **Output Layer:** A linear layer (`torch.nn.Linear(HIDDEN_DIM, 1)`) maps the final hidden representation to a single scalar, the estimated damping coefficient $\hat{b}$. A final ReLU activation is applied to this output to ensure the physical constraint $\hat{b} \ge 0$.
-   **Operation for 1-Link System:** For the 1-link pendulum, each graph has only one node and no edges. In this case, the GCN layers effectively operate as a standard Multi-Layer Perceptron (MLP) on the node features.

The model architecture can be summarized as:
$h_0 = [\theta_t, \omega_t]$
$h_1 = \text{ReLU}(\text{MLP}_1(h_0))$
$h_2 = \text{ReLU}(\text{MLP}_2(h_1))$
$h_3 = \text{MLP}_3(h_2)$
$\hat{b} = \text{ReLU}(\text{Linear}(h_3))$

#### 3.2.4 Training Pipeline (`src/train.py`)

**A. Differentiable Physics Simulation (`simulate_step_physical`)**

The core of the unsupervised learning is the differentiable physics simulation. For the 1-link pendulum, the equation of motion is implemented as:
```math
I_{yy} \alpha_t = \tau_{app,t} + \tau_{g,t} - \hat{b} \omega_t
```
where:
-   $I_{yy}$ is the moment of inertia about the pivot axis.
-   $\tau_{app,t}$ is the applied torque.
-   $\tau_{g,t} = -m g L_{com} \sin(\theta_t)$ is the gravitational torque, assuming $\theta_t$ is the angle from the vertical.
-   $\hat{b}$ is the GNN's estimated damping coefficient.

The angular acceleration $\alpha_t$ is calculated from this equation. The next state is then predicted using a **semi-implicit Euler integration** scheme, which is fully differentiable in PyTorch:
```math
\begin{align*}
\omega_{t+1} &= \omega_t + \alpha_t \Delta t \\
\theta_{t+1} &= \theta_t + \omega_{t+1} \Delta t \quad \text{(uses the updated velocity)}
\end{align*}
```

**B. Loss Function and Optimization**
-   The loss function is a direct Mean Squared Error between the predicted next state vector and the true next state vector: `torch.nn.functional.mse_loss(predicted_next_state, true_next_state)`. It does not apply separate weights to the angle and velocity components.
-   An Adam optimizer is used with `LEARNING_RATE = 1e-3` and `WEIGHT_DECAY = 1e-5`. The training loop is enhanced with:
    -   `torch.optim.lr_scheduler.ReduceLROnPlateau`: Dynamically reduces the learning rate if the validation loss plateaus.
    -   **Early Stopping:** Halts training if the validation loss does not improve for `patience = 20` epochs, preventing overfitting.
    -   **Gradient Clipping:** Clips gradients to a maximum norm of 1.0 to ensure training stability.

All hyperparameters are managed in `src/config.py`.

### 3.3 Experiments and Results

*(This section should be filled with your latest results. The structure below is based on your final code.)*

The GNN model was trained and evaluated on datasets derived from MuJoCo simulations of a 1-link pendulum.

#### 3.3.1 Experimental Setup
-   **Datasets:** Trajectories were generated using various conditions. The final training used JSON files located in `data/mujoco/`.
-   **Data Split:** The combined dataset was split into training and testing sets with an 80/20 ratio.
-   **Training:** The model was trained using the parameters specified in Section 3.2.4.

#### 3.3.2 Next-State Prediction Performance
The primary metric for training is the MSE loss for next-state prediction.

**(Please insert your MSE loss curves here - e.g., a plot showing training and testing MSE vs. epochs.)**

*Figure 3.1: Training and Testing MSE Loss for Next-State Prediction.*

The MSE loss on the training set decreased to approximately **[Your Final Training MSE]** and on the test set to approximately **[Your Final Test MSE]**. The decreasing loss indicates that the model is successfully learning to predict the system's evolution.

#### 3.3.3 Damping Coefficient Estimation
The core goal was to see if the GNN could infer the true physical damping coefficient $b_{true}$ by minimizing the next-state prediction error.

*Example Output from `main.py` after training:*
```
Sample prediction (unsupervised):
Current state θ, ω:
[[...]]
Predicted next state θ_next, ω_next:
[[...]]
True next state θ_next, ω_next (from MuJoCo):
[[...]]
Estimated damping per joint (from GNN):
[Your Estimated b_coeff]
True physical damping per joint (from JSON):
[Value of b_true from your JSON file]
  (using mass: ..., length_com_for_gravity: ..., inertia_yy: ..., dt: ..., gravity_accel: ...)
```

**Observations on Damping Estimation:**
- The GNN-estimated damping coefficient $\hat{b}$ **[Describe your final observation: e.g., "converged to a value of X, which shows a discrepancy from the true physical damping $b_{true}$", or "was consistently around Y"].**
- This discrepancy suggests that the GNN is learning an "effective" damping coefficient that compensates for systemic differences between our semi-implicit Euler integrator and MuJoCo's more complex implicit integrator.

### 3.4 Discussion of GNN Module

#### 3.4.1 Effectiveness for System Identification
The GNN-based approach demonstrates the potential for learning system dynamics in an unsupervised manner. By tasking the model with predicting the next state, it is forced to learn an internal representation that captures some aspects of the underlying physics. However, the results indicate that when the differentiable physics model is an approximation of the true system, the GNN learns "effective" parameters optimal for its own internal model, which may not match the true physical values.

#### 3.4.2 Impact of Data and Model Fidelity
-   **Physics Model Accuracy:** The accuracy of the `simulate_step_physical` function is paramount. While it includes key physical terms (mass, inertia, gravity), the discrepancy due to the choice of integrator (semi-implicit Euler vs. MuJoCo's implicit solver) remains a significant factor, likely forcing the GNN to learn compensatory parameter values.
-   **Input Feature Representation:** The model uses the raw angle $\theta$ as an input. This is challenging for neural networks due to the angle's periodicity (e.g., $0$ and $2\pi$ are the same state but have different numerical values). This could hinder learning performance compared to using `sin(θ)` and `cos(θ)`.
-   **Loss Scaling:** The MSE loss is unweighted. Since angular velocities ($\omega$) often have a much larger numerical range than angles ($\theta$), the loss value and the resulting gradients may be dominated by errors in predicting $\omega$. This could lead to a model that is very good at predicting velocity but less accurate at predicting the next angular position.

#### 3.4.3 Challenges and Limitations
-   **Parameter Identifiability:** Estimating a single parameter like damping ($b$) is difficult when it interacts with many other known (but potentially imperfectly specified) parameters and unmodeled dynamics (e.g., friction, integrator differences).
-   **Integrator Discrepancy:** The difference between our simple Euler integrator and MuJoCo's implicit solver is a significant source of systematic error that the GNN attempts to compensate for by adjusting $\hat{b}$.

### 3.5 Future Work and Potential Improvements for GNN Module

-   **More Sophisticated Differentiable Simulator:** Implementing a more accurate differentiable integrator (e.g., Runge-Kutta 4) could reduce the model mismatch and lead to more accurate physical parameter estimation.
-   **Input Feature Engineering:** A key improvement would be to provide the GNN with engineered features that respect the physics, such as using `sin(θ)` and `cos(θ)` instead of raw `θ` to handle periodicity. Additionally, normalizing input features (especially `ω`) could improve training stability.
-   **Weighted Loss Function:** Introduce separate weights for the MSE loss components of angle and angular velocity to balance their contributions, preventing the loss from being dominated by velocity errors.
-   **Handling Multi-Link Systems:** Extending this to N-link robots where GNNs can truly leverage graph convolutions across multiple connected joints is a key next step. This would involve correctly defining edge features and ensuring the differentiable physics model can handle multi-body dynamics.
-   **Multi-Parameter Estimation:** Extend the GNN to estimate multiple parameters simultaneously (e.g., $m, I_{yy}, b$), which would require careful consideration of their identifiability.

### References
[^1]: Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. In International conference on learning representations (ICLR).