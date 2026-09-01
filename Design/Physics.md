# 6-DOF Interceptor Physics Model

## 1. Coordinate Systems & Inertial Frames
Before defining the state, it is critical to explicitly define the inertial and body frames, especially to ensure compatibility with our datasets (MidAir and NeuroBEM).

### World Frame (Inertial)
We adopt the **ENU (East-North-Up)** coordinate system, which is standard in ROS (REP 103) and highly prevalent in computer vision / MoCap datasets.
* **X-axis:** East
* **Y-axis:** North
* **Z-axis:** Up
* **Gravity ($`g`$):** Acts in the $`-Z`$ direction ($`[0, 0, -9.81]^T`$).

*Dataset Alignment:* Both the MidAir simulation dataset and the NeuroBEM MoCap dataset report altitude as positive $`+Z`$, confirming they also utilize an Up-positive world frame (ENU or NWU).

### Body Frame
We adopt the **FLU (Forward-Left-Up)** coordinate system for the drone's body.
* **X-axis:** Forward (out the nose)
* **Y-axis:** Left
* **Z-axis:** Up (out the top of the drone)

Classical aerospace literature often uses NED (North-East-Down) and FRD (Forward-Right-Down). By explicitly choosing ENU/FLU, we avoid negative-altitude confusion and align perfectly with our pre-processed dataset loaders.

## 2. State Representation
Because we model first-order motor lag (to capture the sim-to-real gap), the individual rotor speeds must be integrated over time rather than applied instantly. Therefore, the drone is defined by a **17-variable state**:
* **Position (World Frame):** $`\vec{p} = [x, y, z]^T`$ (3 states)
* **Velocity (World Frame):** $`\vec{v} = [v_x, v_y, v_z]^T`$ (3 states)
* **Orientation (Quaternions):** $`\mathbf{q} = [q_w, q_x, q_y, q_z]^T`$ (4 states)
* **Angular Velocity (Body Frame):** $`\vec{\omega} = [p, q, r]^T`$ (3 states)
* **Motor Speeds:** $`\vec{\Omega} = [\Omega_1, \Omega_2, \Omega_3, \Omega_4]^T`$ (4 states)

## 3. Translational Dynamics (Newton's Second Law)
The acceleration of the drone in the world frame is dictated by gravity, aerodynamic drag, and the thrust produced by the rotors (which points strictly "up" relative to the drone's body).

$$ m \dot{\vec{v}} = m \vec{g} + \mathbf{R}(\mathbf{q}) \vec{T}_{body} - \vec{F}_{drag} $$

* $`m`$: Mass of the interceptor (kg).
* $`\vec{g}`$: Gravity vector $`[0, 0, -9.81]^T`$.
* $`\mathbf{R}(\mathbf{q})`$: Rotation matrix converting Body Frame to World Frame.
* $`\vec{T}_{body}`$: Total thrust vector in the body frame $`[0, 0, \sum F_i]^T`$.
* $`\vec{F}_{drag}`$: Aerodynamic drag force. Because a quadrotor is not a perfect sphere, its drag profile is highly **anisotropic** (e.g., falling flat encounters massive air resistance, while diving nose-down is highly aerodynamic). To capture this, drag must be computed in the **Body Frame** using diagonal coefficient matrices ($`D_x \neq D_z`$) and then rotated back to the World Frame:
  1. Transform velocity to Body Frame: $`\vec{v}_{body} = \mathbf{R}^T \vec{v}`$
  2. Compute Body-Frame Drag: $`\vec{F}_{drag, body} = \mathbf{D}_{lin} \vec{v}_{body} + \mathbf{D}_{quad} |\vec{v}_{body}| \circ \vec{v}_{body}`$
  3. Transform back to World Frame: $`\vec{F}_{drag} = \mathbf{R} \vec{F}_{drag, body}`$

## 4. Rotational Dynamics (Euler's Equations)
The angular acceleration is determined by the torques applied by the rotors and the gyroscopic effects of the drone's own spinning mass.

$$ \mathbf{J} \dot{\vec{\omega}} = \vec{\tau} - \vec{\omega} \times (\mathbf{J} \vec{\omega}) $$

* $`\mathbf{J}`$: $`3 \times 3`$ Inertia matrix of the drone (diagonal for symmetric drones).
* $`\vec{\tau}`$: Torque vector $`[\tau_\phi, \tau_\theta, \tau_\psi]^T`$ (Roll, Pitch, Yaw torques).
* $`\dot{\vec{\omega}}`$: Angular acceleration.

The quaternion kinematics update equation is:
$$ \dot{\mathbf{q}} = \frac{1}{2} \mathbf{q} \otimes [0, p, q, r]^T $$

## 5. Control Allocation (Motor Mixing)
The drone is driven by 4 rotors. Each rotor $`i`$ spins at speed $`\Omega_i`$, producing an upward force $`F_i = k_f \Omega_i^2`$ and a yaw drag torque $`M_i = k_m \Omega_i^2`$. 

The drone utilizes an **"X" frame configuration** (standard for agile interceptors and racing drones), meaning the motors are mounted at 45-degree angles to the forward axis, rather than on the axes directly ("+" frame). 

Let $`l = L \sin(45^\circ) = L \frac{\sqrt{2}}{2}`$ be the perpendicular moment arm to the roll and pitch axes. The total thrust $`T`$ and torques $`\vec{\tau}`$ are calculated via the X-configuration mixing matrix:

$$ \begin{bmatrix} T \\ \tau_\phi \\ \tau_\theta \\ \tau_\psi \end{bmatrix} = \begin{bmatrix} k_f & k_f & k_f & k_f \\ -k_f l & -k_f l & k_f l & k_f l \\ k_f l & -k_f l & k_f l & -k_f l \\ k_m & -k_m & -k_m & k_m \end{bmatrix} \begin{bmatrix} \Omega_1^2 \\ \Omega_2^2 \\ \Omega_3^2 \\ \Omega_4^2 \end{bmatrix} $$
*(Derived strictly for the **FLU** body frame. Roll torque $`\tau_\phi = \sum y_i F_i`$, Pitch torque $`\tau_\theta = \sum -x_i F_i`$. Motors: 1=Back-Right (CW), 2=Front-Right (CCW), 3=Back-Left (CCW), 4=Front-Left (CW)).*

### First-Order Motor Lag (Sim-to-Real Gap)
In reality, brushless motors cannot change their RPM instantly. If a drone commands maximum thrust to dodge, it takes tens of milliseconds for the propellers to physically spool up. To ensure our RL agent learns a policy that transfers to the real world without crashing, we must model this mechanical delay using a first-order low-pass filter:

$$ \dot{\Omega}_i = \frac{1}{\tau_m} (\Omega_{cmd, i} - \Omega_i) $$

* $`\Omega_{cmd, i}`$: The commanded rotor speed from the flight controller.
* $`\Omega_i`$: The actual physical rotor speed.
* $`\tau_m`$: The motor time constant (typically 0.02 to 0.05 seconds for high-performance quadrotors).

Without this lag, the RL agent will learn to perfectly "time" the net deployment based on impossibly fast, instant corrective maneuvers that a real drone cannot execute.

## 6. The "Inner-Loop" Geometric Flight Controller on $`\text{SO}(3)`$
Because our HOCBF-PN algorithm outputs a desired 3D acceleration vector ($`\vec{a}_{cmd}`$), we must convert this into physical motor torques. To avoid Gimbal Lock and the Quaternion Double-Cover problem, we use a Geometric Tracking Controller built directly on the Special Orthogonal group $`\text{SO}(3)`$.

**1. Desired Force, Thrust, & Drag Feedforward:**
The drone must achieve the commanded acceleration while fighting gravity **and** aerodynamic drag. If we do not actively compensate for drag, the drone will under-accelerate at high speeds and the HOCBF-PN safety boundary will fail. We solve this by adding Drag Feedforward:
$$ \vec{F}_{des} = m (\vec{a}_{cmd} - \vec{g}) + \vec{F}_{drag} $$
The total required thrust is the projection of this force onto the drone's current body axis:
$$ T = \vec{F}_{des} \cdot \vec{z}_{body} $$

**2. Desired Attitude ($`\mathbf{R}_d`$):**
The optimal orientation requires the drone's Z-axis to align perfectly with the desired force vector:
$$ \vec{z}_d = \frac{\vec{F}_{des}}{\|\vec{F}_{des}\|} $$
Given a desired yaw heading (e.g., pointing the camera at the target), we construct the orthogonal $`\vec{x}_d`$ and $`\vec{y}_d`$ axes to form the desired rotation matrix $`\mathbf{R}_d \in \text{SO}(3)`$.

**3. Attitude Error on $`\mathfrak{so}(3)`$:**
Instead of error-prone Euler angles or Quaternions, we extract the minimal-path rotation error vector directly using the Lie Algebra "vee" map $`(\cdot)^\vee`$:
$$ \vec{e}_R = \frac{1}{2} (\mathbf{R}_d^T \mathbf{R} - \mathbf{R}^T \mathbf{R}_d)^\vee $$
$$ \vec{e}_\omega = \vec{\omega} - \mathbf{R}^T \mathbf{R}_d \vec{\omega}_d $$

**4. Torque Command:**
The final commanded torques are generated via a PD control law over the $`\mathfrak{so}(3)`$ error manifold. It includes gyroscopic cancellation and **Angular Acceleration Feedforward** ($`\dot{\vec{\omega}}_d`$) to ensure zero tracking lag during aggressive maneuvers:
$$ \vec{\tau} = -k_R \vec{e}_R - k_\omega \vec{e}_\omega + \vec{\omega} \times (\mathbf{J} \vec{\omega}) - \mathbf{J} (\vec{\omega} \times \mathbf{R}^T \mathbf{R}_d \vec{\omega}_d - \mathbf{R}^T \mathbf{R}_d \dot{\vec{\omega}}_d) $$

This completely mathematically bypasses singularities and unwinding issues, guaranteeing that the interceptor takes the absolute shortest rotational path to aim its thrust vector and deploy the net.

---

## 7. Sanity Checks (Unit Testing Roadmap)
Before deploying this physics engine for RL training, the following mathematical invariants must be verified via unit tests to ensure the math is rock-solid:

**1. The Hover Invariant**
* **Condition:** $`\vec{a}_{cmd} = [0, 0, 0]^T`$
* **Expected:** The geometric controller computes $`\vec{F}_{des} = [0, 0, 9.81m]^T`$. The attitude stabilizes at pitch/roll = 0. All four motors settle perfectly at $`\Omega = \sqrt{\frac{mg}{4 k_f}}`$. Acceleration $`\dot{\vec{v}}`$ is identically zero.

**2. The Freefall & Terminal Velocity Check**
* **Condition:** Motors turned off ($`\Omega_{cmd} = 0`$).
* **Expected:** Initial acceleration is perfectly $`[0, 0, -9.81]^T`$. As velocity increases in the $`-Z`$ direction, anisotropic drag increases until $`\vec{F}_{drag, Z} = mg`$, at which point acceleration becomes $`0`$ (terminal velocity).

**3. FLU Coordinate Mixing Matrix Verification**
* **Condition:** Command a pure positive Roll torque ($`\tau_\phi > 0`$).
* **Expected:** By the Right-Hand Rule in FLU (X=Forward, Y=Left, Z=Up), positive roll dips the right side and lifts the left side. Motors 3 (BL) and 4 (FL) must increase RPM, while Motors 1 (BR) and 2 (FR) must decrease RPM.

**4. First-Order Motor Lag Time-Constant**
* **Condition:** A step input commands motors from $`0 \to 1000`$ RPM.
* **Expected:** The simulated physical $`\Omega`$ state must reach exactly $`\approx 63.2\%`$ of the commanded step ($`632`$ RPM) at exactly time $`t = \tau_m`$ seconds.

**5. SO(3) Attitude Shortest-Path Tracking**
* **Condition:** Command the drone to track a target orientation that is 179 degrees pitched up, then perturb it to 181 degrees.
* **Expected:** A quaternion-based PID controller might violently spin 358 degrees to unwind the double-cover. Our $`\text{SO}(3)`$ geometric controller must simply push 2 degrees in the opposite direction, always taking the mathematical shortest path on the rotation manifold.

**6. Quaternion Normalization Conservation**
* **Condition:** Continuous numerical integration of $`\dot{\mathbf{q}} = \frac{1}{2} \mathbf{q} \otimes \vec{\omega}`$ over 10,000 steps.
* **Expected:** The norm of the quaternion state $`\|\mathbf{q}\|`$ must strictly remain $`1.0`$ (requiring a normalization step in the RK4/Euler integrator to prevent drift).
