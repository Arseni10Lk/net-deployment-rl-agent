# High-Order Control Barrier Function-Augmented Guidance Architecture for Decoupled Autonomous Drone Interception

## 1. Executive Summary & Problem Formulation

In physical aerial capture operations, deploying an autonomous interceptor drone to net-capture a highly agile target drone requires a strict functional decoupling between the high-level capture policy and low-level guidance dynamics. The primary operational architecture involves a Reinforcement Learning (RL) agent tasked exclusively with deciding the discrete execution moment of firing a net, while the interceptor's spatial maneuvering is governed by an underlying continuous guidance algorithm. Standard pursuit methodologies, such as Proportional Navigation Guidance (PNG) or Pure Pursuit (PP), optimize lateral acceleration to nullify the line-of-sight (LOS) angular rate, driving the pursuer directly toward an intercept point. However, because classical PNG commands only lateral acceleration and lacks axial speed control or dynamic braking authority, a continuous pursuit trajectory culminates in a physical mid-air collision if the net deployment action is withheld by the RL policy. Attempting to rectify this operational challenge through naive structural modifications introduces severe system pathologies:

### 1.1 Severe Collision Penalization Pathologies

 Heavy penalties in the RL reward function for physical collisions induce catastrophic premature policy convergence. The RL agent learns to panic-fire the net at long ranges simply to terminate the episode safely and avoid the massive penalty, destroying capture effectiveness.

### 1.2 Virtual Target Point Offset Instability

 Aiming the PNG law at a virtual target point offset behind the intruder induces severe guidance instability. When an agile target executes evasive maneuvers, rapid rotations of its velocity vector cause the spatial coordinates of the offset virtual target to experience high-frequency positional jumps. This forces the interceptor guidance system into control saturation, extreme phase lag, and destructive lateral oscillations.

### 1.3 Hard Standoff Boundary Discontinuities

 Implementing hard-coded state-switching boundaries (such as transitioning from PNG to a braking regime at a fixed 5-meter boundary) introduces non-differentiable step-discontinuities into the system's differential equations of motion. This structural boundary acts as an artificial "bouncy wall," causing high-frequency chatter in orientation and velocity right at the critical engagement envelope, destroying the Markov Decision Process (MDP) smoothness required for policy gradient convergence. To resolve these coupled constraints, this report formulates a unified, mathematically continuous guidance architecture: High-Order Control Barrier Function-Augmented Proportional Navigation (HOCBF-PN). By mapping physical separation and relative closing kinematics into forward-invariant safety manifolds, the HOCBF-PN framework enables aggressive, optimal closing maneuvers against maneuvering targets while guaranteeing a smooth, non-discontinuous transition to safe standoff station-keeping if the RL agent refrains from firing.

## 2. Kinematic Modeling & Failure Analysis of Baseline Strategies

### 2.1 System Kinematics and Engagement Geometry

Consider a three-dimensional engagement space containing an interceptor drone ($I$) and a maneuvering target drone ($T$). Let $\vec{p}_i, \vec{p}_t \in \mathbb{R}^3$ denote their inertial positions, and $\vec{v}_i, \vec{v}_t \in \mathbb{R}^3$ denote their inertial velocity vectors. The relative position vector $\vec{r}$ and relative velocity vector $\vec{v}$ are defined as:

$$\vec{r} = \vec{p}_t - \vec{p}_i, \quad \vec{v} = \vec{v}_t - \vec{v}_i$$

The relative distance is $R = \Vert{}\vec{r}\Vert{}$, and the Line-of-Sight (LOS) unit vector is $\hat{u}_r = \vec{r} / R$. The LOS angular velocity vector $\vec{\omega}_{LOS}$ describes the rotation of the relative range vector in 3D space:

$$\vec{\omega}_{LOS} = \frac{\vec{r} \times \vec{v}}{R^2}$$

The range rate (axial closing velocity) is given by $\dot{R} = \frac{\vec{r} \cdot \vec{v}}{R} = -v_c$, where $v_c$ represents the closing speed along the LOS vector.

### 2.2 Mathematical Breakdown of Baseline Guidance Failures

Standard Proportional Navigation Guidance commands an acceleration vector $\vec{a}_{PNG}$ perpendicular to the LOS vector to drive $\vec{\omega}_{LOS} \to 0$:

$$\vec{a}_{PNG} = N \left( \vec{v}_{rel} \times \vec{\omega}_{LOS} \right)$$

where $N \ge 3$ is the navigation constant and $\vec{v}_{rel} = -\vec{v}$. Because $\vec{a}_{PNG} \perp \vec{v}_{rel}$, standard PNG exerts zero longitudinal braking force along the range vector $\vec{r}$. Consequently, as $R \to 0$, the range rate $\dot{R}$ remains strictly negative, guaranteeing a physical collision at time $t_{go} = R / v_c$ unless an external force intervenes.

### 2.3 Kinematic Instability of Unfiltered Virtual Target Offsets

When attempting to maintain a standoff distance $d$ behind an evasive target, the virtual target position $\vec{p}_{vt}$ is traditionally defined by shifting the target position backwards along its normalized velocity vector $\hat{v}_t = \vec{v}_t / \Vert{}\vec{v}_t\Vert{}$:

$$\vec{p}_{vt} = \vec{p}_t - d \cdot \frac{\vec{v}_t}{\Vert{}\vec{v}_t\Vert{}}$$

Differentiating $\vec{p}_{vt}$ with respect to time yields the virtual target velocity $\vec{v}_{vt}$:

$$\vec{v}_{vt} = \vec{v}_t - d \cdot \left( \frac{\vec{a}_t}{\Vert{}\vec{v}_t\Vert{}} - \frac{\vec{v}_t (\vec{v}_t \cdot \vec{a}_t)}{\Vert{}\vec{v}_t\Vert{}^3} \right)$$

When an agile target drone executes rapid evasive maneuvers (high lateral acceleration $\vec{a}_t \perp \vec{v}_t$), the term $\frac{\vec{a}_t}{\Vert{}\vec{v}_t\Vert{}}$ introduces large magnitude variations into $\vec{v}_{vt}$. Differentiating once more to obtain virtual target acceleration $\vec{a}_{vt}$ reveals that the acceleration required to track $\vec{p}_{vt}$ scales with target jerk $\dot{\vec{a}}_t$ and the inverse square of target speed $\Vert{}\vec{v}_t\Vert{}^2$. For light, highly maneuverable target drones executing rapid direction changes, this formulation causes $\vec{p}_{vt}$ to jump wildly across space. This spatial teleportation exceeds the physical acceleration authority of the interceptor, inducing control loop saturation, extreme phase lag, and destructive lateral oscillations.

### 2.4 Phase-Space Discontinuity of Hard Boundaries

A hard-coded boundary algorithm splits the control law into piecewise dynamic regimes based on a distance threshold $R_{safe}$:

$$\vec{a}_{cmd} = \begin{cases} \vec{a}_{PNG}(\vec{r}, \vec{v}) & \text{if } R > R_{safe} \\ \vec{a}_{brake}(\vec{r}, \vec{v}) & \text{if } R \le R_{safe} \end{cases}$$

At the boundary hypersurface $\mathcal{S} = \{\vec{x} \mid R - R_{safe} = 0\}$, the system state matrix encounters a non-differentiable $C^{-1}$ step change. In reinforcement learning, policy gradient estimators rely on the smooth evolution of state transition probabilities $P(\vec{s}_{t+1} \mid \vec{s}_t, \vec{a}_t)$. The step discontinuity at $\mathcal{S}$ creates an artificial gradient barrier. When the drone reaches $R_{safe}$, the instantaneous force reversal causes high-frequency velocity and orientation vector flipping ("bouncy wall" effect), destabilizing state estimation and corrupting the RL policy's observation vector at the exact moment net deployment feasibility must be evaluated.

### 2.5 Synthesis of Guidance Strategy Performance

The following table summarizes the baseline guidance laws vs the proposed unified solution:

| Guidance Strategy | Mathematical Formulation & Kinematic Mechanism | Operational Failure Mode | Impact on RL Policy Training |
| :--- | :--- | :--- | :--- |
| **Standard Proportional Navigation (PNG)** | $\vec{a} = N(\vec{v}_{rel} \times \vec{\omega}_{LOS})$<br><br>Commanded acceleration strictly perpendicular to relative velocity. | Zero axial braking force; pure pursuit trajectory guarantees physical mid-air crash if net is withheld. | Forces agent to "panic-fire" net prematurely to avoid large crash penalties, degrading terminal capture rate. |
| **Rigid Virtual Target Offset (VTO)** | $\vec{p}_{vt} = \vec{p}_t - d \cdot (\vec{v}_t / \Vert{}\vec{v}_t\Vert{})$<br><br>Aims pursuit law at a point offset behind the target. | Target velocity vector rotation causes offset point to teleport wildly; induces actuator saturation. | Introduces extreme high-frequency noise into observation states; policy fails to converge on agile targets. |
| **Hard Boundary Switching** | $\vec{a} = \vec{a}_{PNG}$ for $R > R_{safe}$;<br>$\vec{a} = \vec{a}_{brake}$ for $R \le R_{safe}$<br><br>Piecewise state-space switching. | Creates non-differentiable $C^{-1}$ step change at boundary; generates high-frequency chatter ("bouncy wall"). | Impairs Markov Decision Process smoothness; destabilizes attitude/velocity inputs at firing moment. |
| **HOCBF-PN Safety Filter (Proposed)** | $\vec{a}_{cmd} = \arg\min \Vert{}\vec{a} - \vec{a}_{nom}\Vert{}^2$<br>subject to $\mathbf{A}_{cbf}\vec{a} \le b_{cbf}$<br><br>Dynamic convex optimization filter. | None; guarantees continuous forward invariance, smooth axial braking, and safe standoff fallback. | Maintains $C^1/C^2$ continuous state transitions; allows agent to hold fire safely and optimize net capture timing. |

## 3. Unified Guidance Solution: HOCBF-Augmented Proportional Navigation (HOCBF-PN)

To achieve continuous, collision-free engagement without introducing state-space discontinuities or tracking instabilities, pursuit and safety must be unified into a single optimization problem. The recommended architecture employs Blended Generalized Proportional Navigation (B-GPN) to generate nominal terminal engagement commands, wrapped within a High-Order Control Barrier Function (HOCBF) solved continuously via an online Quadratic Program (QP).

### 3.1 Nominal Guidance Law: Blended Generalized Proportional Navigation (B-GPN)

The unconstrained nominal guidance command $\vec{a}_{nom}$ is synthesized using Blended Generalized Proportional Navigation (B-GPN). B-GPN augments classic proportional navigation by incorporating target acceleration estimates along the LOS plane, maintaining robust closing geometry against maneuvering evaders:

$$\vec{a}_{nom} = N \left( \vec{v}_{rel} \times \vec{\omega}_{LOS} \right) + \frac{N}{2} \vec{a}_{t,\perp}$$

where $N \in [3.0, 5.0]$ is the non-dimensional navigation constant, $\vec{v}_{rel} = -\vec{v}$, and $\vec{a}_{t,\perp}$ is the component of estimated target acceleration normal to the Line-of-Sight vector.

### 3.2 High-Order Control Barrier Function (HOCBF) Design

To prevent collisions continuously, a safe state space $\mathcal{C}$ is defined as the superlevel set of a continuously differentiable scalar function $h(\vec{x}): \mathbb{R}^n \to \mathbb{R}$:

$$\mathcal{C} = \{\vec{x} \in \mathbb{R}^n \mid h(\vec{x}) \ge 0\}$$

$$\partial \mathcal{C} = \{\vec{x} \in \mathbb{R}^n \mid h(\vec{x}) = 0\}$$

the safe state set is defined strictly in physical 3D position space as the distance boundary relative to the target:

$$h(\vec{x}) = \Vert{}\vec{r}\Vert{}^2 - d_{min}^2$$

where $\vec{r} = \vec{p}_t - \vec{p}_i$ is the relative position vector and $d_{min}$ is the minimum safe physical separation distance.

Because the control input (interceptor acceleration $\vec{a}_i$) does not appear in $h(\vec{x})$ or its first time derivative $\dot{h}(\vec{x}) = 2\vec{r}\cdot\vec{v}$, this barrier function has a true Relative Degree $r = 2$. Differentiating a second time yields:

$$\ddot{h}(\vec{x}, \vec{a}_i) = 2\Vert{}\vec{v}\Vert{}^2 + 2\vec{r}\cdot(\vec{a}_t - \vec{a}_i)$$

Using the High-Order Control Barrier Function formulation with linear gain parameters $k_1, k_2 > 0$:

$$\psi_0(\vec{x}) = \Vert{}\vec{r}\Vert{}^2 - d_{min}^2$$

$$\psi_1(\vec{x}) = \dot{\psi}_0(\vec{x}) + k_1 \psi_0(\vec{x}) = 2\vec{r}\cdot\vec{v} + k_1 (\Vert{}\vec{r}\Vert{}^2 - d_{min}^2)$$

$$\psi_2(\vec{x}, \vec{a}_i) = \ddot{\psi}_0(\vec{x}, \vec{a}_i) + (k_1 + k_2)\dot{\psi}_0(\vec{x}) + k_1 k_2 \psi_0(\vec{x}) \ge 0$$

### 3.3 Quadratic Program (QP) Safety Filter Formulation

The actual executable acceleration command $\vec{a}_{cmd}$ sent to the low-level flight controller is computed in real-time ($100\text{ Hz} - 500\text{ Hz}$) by solving a convex Quadratic Program:

$$\vec{a}_{cmd} = \arg\min_{\vec{a} \in \mathbb{R}^3} \frac{1}{2} \Vert{}\vec{a} - \vec{a}_{nom}\Vert{}^2$$

$$\text{subject to } \mathbf{A}_{cbf} \vec{a} \le b_{cbf}$$

$$\mathbf{A}_{act} \vec{a} \le \vec{b}_{act}$$

Expanding and rearranging $\psi_2 \ge 0$ into standard linear inequality form $\mathbf{A}_{cbf} \vec{a}_i \le b_{cbf}$:

$$\mathbf{A}_{cbf} = 2\vec{r}^T$$

$$b_{cbf} = 2\Vert{}\vec{v}\Vert{}^2 + 2\vec{r}\cdot\vec{a}_t + 2(k_1 + k_2)(\vec{r}\cdot\vec{v}) + k_1 k_2 (\Vert{}\vec{r}\Vert{}^2 - d_{min}^2)$$

#### 3.3.1 Key Operational Features

**Automatic Velocity Dampening**: The term $2(k_1 + k_2)(\vec{r}\cdot\vec{v})$ naturally acts as a range-rate brake. When closing rapidly ($\vec{r}\cdot\vec{v} < 0$), $b_{cbf}$ decreases, forcing the QP solver to command deceleration along the approach vector.

**Zero Division / Zero Square Roots**: All terms are computed using simple vector dot products and additions. This eliminates potential floating-point overflow or numerical instabilities near $\Vert{}\vec{r}\Vert{} \to 0$.

The inequality $\mathbf{A}_{act} \vec{a} \le \vec{b}_{act}$ explicitly enforces physical actuator limits and thrust vector saturation constraints ($\vert{}\vec{a}_{i,z}\vert{} \le a_{max,z}$, $\Vert{}\vec{a}_{i,xy}\Vert{} \le a_{max,xy}$).

### 3.4 Mathematical Continuity and Operational Dynamics

Because $\mathbf{A}_{cbf}$ and $b_{cbf}$ are continuously differentiable ($C^1/C^2$) functions over the state manifold $\mathbb{R}^3 \setminus \{0\}$, the solution map $\vec{a}_{cmd}(\vec{x})$ of the convex QP is Lipschitz continuous. This mathematical smoothness governs the interceptor's behavior across all operational phases:

#### 3.4.1 Far-Field Pursuit

 ($R \gg d_{min}$): When the interceptor is far from the target, $h(\vec{x}) \gg 0$ and $b_{cbf}$ is large. The safety constraint remains inactive ($\mathbf{A}_{cbf}\vec{a}_{nom} < b_{cbf}$). The system executes pure B-GPN, aggressively closing distance on maneuvering targets.

#### 3.4.2 Near-Field Capture Zone

 ($R \to d_{min}$): As the interceptor enters the net capture envelope, $h(\vec{x}) \to 0$ and the constraint becomes active. The QP smoothly alters $\vec{a}_{cmd}$ away from $\vec{a}_{nom}$, applying continuous axial braking and lateral deflection without state steps.

#### 3.4.3 Hold-Fire Standoff Regime

: If the RL policy refrains from firing the net, the interceptor smoothly asymptotic-approaches a dynamic standoff orbit at distance $R \approx d_{min} + \epsilon$, matching target velocity ($\dot{R} \to 0$) without experiencing step discontinuities, control chatter, or physical collisions.

### 3.5 Multi-Platform Adaptation & Parameter Scaling

The physical dynamics of interceptor platforms vary based on vehicle scale, mass properties, and thrust-to-weight ratios. The HOCBF-PN architecture accommodates these variations by tuning specific control hyperparameters. Recommended hyperparameter guidelines across platform scales:

| Operational Parameter | Light Interceptor Drones (< 2.0 kg) | Medium Interceptor Drones (2.0 kg - 15.0 kg) | Kinematic & Dynamic Rationale |
| :--- | :--- | :--- | :--- |
| **Thrust-to-Weight Ratio ($T/W$)** | High ($T/W \ge 4.0$) | Moderate ($1.5 \le T/W \le 2.5$) | Light drones possess high angular acceleration; medium drones are thrust-limited. |
| **Max Acceleration ($a_{max}$)** | $30.0\text{ m/s}^2 - 50.0\text{ m/s}^2$ | $12.0\text{ m/s}^2 - 20.0\text{ m/s}^2$ | Governs the feasible set boundary $\mathbf{A}_{act}\vec{a} \le \vec{b}_{act}$ in the QP solver. |
| **HOCBF Linear Gains ($k_1, k_2$)** | $k_1 = 4.0, \quad k_2 = 8.0$ | $k_1 = 1.5, \quad k_2 = 3.0$ | Higher gains exploit high control bandwidth; lower gains prevent actuator saturation. |
| **Navigation Constant ($N$)** | $N = 4.0 - 5.0$ | $N = 3.0 - 3.5$ | Higher values increase responsiveness; lower values prevent command oscillations. |
| **Target Acceleration Gain** | Full compensation ($\frac{N}{2}\vec{a}_{t,\perp}$) | Filtered compensation with lag recovery | Prevents sensor noise propagation into heavy motor controllers. |

## 4. Reinforcement Learning Integration & Policy Co-Design

### 4.1 Observation State Space Formulation

Because the underlying HOCBF-PN guidance algorithm deterministically guarantees physical safety and collision avoidance, the discrete RL agent can focus exclusively on maximizing net capture probability. The continuous observation state vector $\mathbf{S}_t \in \mathbb{R}^{14}$ provided to the RL policy at time step $t$ comprises:

$$\mathbf{S}_t = \Big[ R, \quad \dot{R}, \quad \theta_{LOS}, \quad \phi_{LOS}, \quad \vec{v}_{rel,body}, \quad \vec{a}_{t,est}, \quad \tau_{mod}, \quad P_{net\_hit} \Big]$$

where $\theta_{LOS}, \phi_{LOS}$ are the elevation and azimuth angles of the LOS vector, $\vec{v}_{rel,body}$ is relative velocity in the body frame, $\vec{a}_{t,est}$ is the EKF-estimated target acceleration, $\tau_{mod}$ is the modified time-to-closest-approach metric:

$$\tau_{mod} = \frac{R^2 - d_{min}^2}{R \cdot \vert{}\dot{R}\vert{}}$$

and $P_{net\_hit} \in [0, 1]$ is a deterministic geometric projection of net expansion geometry evaluated at the target's relative coordinates.

### 4.2 Operational Control Flow and Action Execution

The control execution follows a strict multi-rate operational structure:

#### 4.2.1 High-Rate Guidance Loop

 ($100\text{ Hz} - 500\text{ Hz}$): The onboard flight controller continuously executes the HOCBF-PN algorithm. It reads target state updates, evaluates the HOCBF linear safety constraint $\mathbf{A}_{cbf}\vec{a} \le b_{cbf}$, solves the convex QP, and commands motor thrusts.

#### 4.2.2 Low-Rate Policy Loop

 ($10\text{ Hz} - 20\text{ Hz}$): The RL agent evaluates the observation vector $\mathbf{S}_t$ and outputs a discrete action $a_t \in \{0, 1\}$ (0: Hold Fire, 1: Fire Net).

#### 4.2.3 Action Execution Logic

If $a_t = 0$ (Hold Fire): The interceptor continues flying under HOCBF-PN guidance. If the drone reaches $R \approx d_{min}$, the HOCBF automatically overrides nominal PNG, smoothly braking the drone into a safe standoff orbit behind the target. If $a_t = 1$ (Fire Net): The net deployment mechanism triggers instantly, launching the expanded net toward the target. The episode terminates, and terminal rewards are assigned based on capture overlap.

### 4.3 Reward Function Structure

 (Eliminating Panic-Firing)To ensure the agent learns optimal firing timing without premature panic-firing, the reward function $r_t$ isolates capture performance from safety enforcement:

$$
r_t = \begin{cases} 
+100 \cdot \text{OverlapRatio}(\text{Net}, \text{Target}) & \text{if Action = Fire (Terminal)} \\ 
-50 & \text{if Action = Fire AND Target Missed (Terminal)} \\ 
r_{approach}(t) & \text{if Action = Hold Fire (Non-Terminal)} 
\end{cases}
$$

The non-terminal step reward $r_{approach}(t)$ encourages optimal positioning within the effective net firing envelope ($R_{net\_min} \le R \le R_{net\_max}$) without penalizing hold-fire decisions:

$$r_{approach}(t) = c_1 \cdot \exp\left( -\frac{(R_t - R_{optimal})^2}{2\sigma_R^2} \right) - c_2 \cdot \Vert{}\vec{\omega}_{LOS}\Vert{}^2$$

where $c_1, c_2 > 0$ are shaping constants.

### 4.4 Training Dynamics Analysis

#### 4.4.1 Elimination of Panic Firing

 Because the underlying HOCBF-PN guidance continuously prevents collisions ($\lim_{R \to d_{min}} \dot{R} = 0$), holding fire does not lead to catastrophic episode-terminating crashes. The agent is never forced to panic-fire to avoid a crash penalty.

#### 4.4.2 Smooth Policy Optimization

 The mathematical smoothness ($C^1/C^2$) of the underlying state space allows policy gradient updates (e.g., PPO advantage estimates $A^{\pi}(s,a)$) to remain stable, accelerating training convergence and establishing high true-interception rates.

## 5. Operational Implementation & Verification Roadmap

Deploying the HOCBF-PN architecture into production requires a structured hardware and software verification pipeline.

### 5.1 Phase 1: Onboard Guidance Engine Integration

The primary requirement is embedding a fast convex optimization solver into the flight management unit (FMU).

#### 5.1.1 Algorithm Structuring

 Implement the B-GPN nominal guidance law alongside an Extended Kalman Filter (EKF) running at $100\text{ Hz}$ to estimate target states $\vec{p}_t, \vec{v}_t, \vec{a}_t$.

#### 5.1.2 QP Solver Setup

 Deploy an embedded C++ quadratic program solver (such as OSQP or qpOASES) on the companion computer (e.g., NVIDIA Jetson or Odroid XU4).

#### 5.1.3 Constraint Compilation

 Formulate the HOCBF matrices $\mathbf{A}\_{cbf}$ and $b\_{cbf}$ using real-time state estimates, enforcing maximum physical acceleration boundaries $\mathbf{A}\_{act}\vec{a} \le \vec{b}\_{act}$ to ensure motor limits are respected.

### 5.2 Phase 2: Simulation Training Environment

The reinforcement learning policy is trained in a physics-accurate 3D simulator (e.g., Gazebo, AirSim, or Isaac Gym).

#### 5.2.1 Environment Wrapping

 Enclose the multi-rate framework in an OpenAI Gymnasium interface, exposing the $14$-dimensional observation state $\mathbf{S}_t$ to the discrete RL policy.

#### 5.2.2 Target Evasion Modeling

 Train the discrete action policy against diverse target behaviors, ranging from passive straight-line flight to aggressive, non-cooperative evasive maneuvers (e.g., high-$g$ weaves, dynamic braking, sharp turns).

#### 5.2.3 Hyperparameter Optimization

 Train the policy using Proximal Policy Optimization (PPO) with clipped surrogate objectives to maintain policy update stability.

### 5.3 Phase 3: Hardware Verification Testing

Before full flight authorization, the combined system must undergo staged validation:

#### 5.3.1 Hold-Fire Verification

 Lock the RL agent action to $a_t = 0$ (Hold Fire) and fly aggressive pursuit intercepts against an agile evader. Confirm that the HOCBF-PN layer smoothly decelerates the interceptor into a standoff orbit at $R \approx d_{min} + \epsilon$ without boundary chatter, instability, or physical contact.

#### 5.3.2 Full Engagement Testing

 Enable the discrete RL policy to execute net firing decisions. Verify that the agent learns to delay deployment until the target is centered within the optimal capture cone ($P_{net\_hit} > 0.85$), achieving high physical capture rates without experiencing premature panic-firing or mid-air collisions.