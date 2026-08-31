# 🦅 Autonomous Drone Interception & Net Deployment

This repository contains the simulation environment and training pipeline for a Reinforcement Learning (RL) agent designed to autonomously intercept and capture highly agile, evasive drones using a deployable net.

### The Challenge
Capturing a fast-moving drone in mid-air is incredibly difficult. If an interceptor gets too close, it risks a catastrophic mid-air collision. If it stays too far, the net will miss. The interceptor must chase the evader through complex maneuvers and deploy the net at the mathematically perfect millisecond.

### 🧠 The Architecture

This project splits the problem into two distinct layers:

#### 1. The Guidance Backend (HOCBF-PN)
Instead of forcing the RL agent to learn basic quadrotor flight dynamics from scratch, the interceptor is flown by a classical control backend:
* **Proportional Navigation (B-GPN):** A missile guidance law that aggressively calculates the optimal intercept vector to catch the evading target.
* **High-Order Control Barrier Functions (HOCBF):** A safety filter powered by a real-time Quadratic Program (QP) solver. The CBF monitors the Proportional Navigation commands and safely overrides them if the interceptor is about to breach a physical safety bubble of a certain radius around the target, preventing a crash.

#### 2. The RL Net Deployment Agent
The RL agent observes the relative kinematics of the engagement (Line-of-Sight angles, closing velocity, distance, target acceleration estimates) and learns the optimal policy for exactly when to trigger the net deployment mechanism. 

### 📊 Real-World Training Data
To ensure the RL agent learns to counter highly aggressive evasive maneuvers, the simulation is powered by two of the world's leading drone tracking datasets: **MidAir** and **NeuroBEM**

### 🎮 The Gymnasium Environment
The entire physics engine, guidance controller, and dataset loader are wrapped into a standard, vectorized `Gymnasium` environment (`DroneNet-3D`). This allows seamless integration with modern Reinforcement Learning libraries like Stable-Baselines3 (SB3) to train the deployment policy.

### 📚 Dataset Citations
The target drone trajectories used in this environment are derived from the following datasets. If you use this work, please cite them accordingly:

```
@article{bauersfeld2021neurobem,
  title={NeuroBEM: Hybrid Aerodynamic Quadrotor Model},
  author={Bauersfeld, Leonard and Kaufmann, Elia and Foehn, Philipp and Sun, Sihao and Scaramuzza, Davide},
  journal={RSS: Robotics, Science, and Systems},
  year={2021},
  publisher={IEEE}
}

@INPROCEEDINGS{Fonder2019MidAir,
  author = {Michael Fonder and Marc Van Droogenbroeck},
  title = {Mid-Air: A multi-modal dataset for extremely low altitude drone flights},
  booktitle = {Conference on Computer Vision and Pattern Recognition Workshop (CVPRW)},
  year = {2019},
  month = {June}
}
```
