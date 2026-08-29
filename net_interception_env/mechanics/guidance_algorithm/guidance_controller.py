import numpy as np
import qpsolvers

from net_interception_env.mechanics.guidance_algorithm.kinematics import compute_relative_kinematics
from net_interception_env.mechanics.guidance_algorithm.nominal_guidance import compute_bgpn_acceleration
from net_interception_env.mechanics.guidance_algorithm.cbf_filter import compute_cbf_matrices

class HOCBFGuidanceController:
    """
    Fully assembles the High-Order Control Barrier Function-Augmented Guidance.
    Connects the Kinematics, Nominal Guidance, and CBF Filters into a Quadratic Program.
    """
    def __init__(self, 
                 d_min: float = 1.5, 
                 k1: float = 1.5, 
                 k2: float = 3.0, 
                 N: float = 4.0,
                 a_max_xy: float = 20.0,
                 a_max_z: float = 10.0,
                 solver: str = 'clarabel'):
        """
        Initializes the guidance controller with physical and mathematical parameters.
        
        Args:
            d_min: Minimum allowable physical standoff distance.
            k1: First HOCBF linear gain (braking horizon).
            k2: Second HOCBF linear gain.
            N: Navigation constant for B-GPN.
            a_max_xy: Maximum lateral acceleration (m/s^2).
            a_max_z: Maximum vertical acceleration (m/s^2).
            solver: The QP solver backend to use (e.g., 'clarabel', 'osqp', 'quadprog').
        """
        self.d_min = d_min
        self.k1 = k1
        self.k2 = k2
        self.N = N
        self.a_max_xy = a_max_xy
        self.a_max_z = a_max_z
        self.solver = solver
        
        # P matrix for objective function: 1/2 * x^T P x + q^T x
        # Minimizing ||a - a_nom||^2 means P is the identity matrix.
        self.P = np.eye(3)
        
        # Actuator limits (box constraints for linear QP)
        self.lb = np.array([-self.a_max_xy, -self.a_max_xy, -self.a_max_z])
        self.ub = np.array([ self.a_max_xy,  self.a_max_xy,  self.a_max_z])

    def compute_command(self, p_i: np.ndarray, v_i: np.ndarray, p_t: np.ndarray, v_t: np.ndarray, a_t: np.ndarray) -> np.ndarray:
        """
        Computes the safe acceleration command for the interceptor drone.
        """
        # 1. Compute Base Kinematics
        kin = compute_relative_kinematics(p_i, v_i, p_t, v_t)
        
        # 2. Compute Nominal Objective (Aggressive Intercept)
        a_nom = compute_bgpn_acceleration(
            v_rel=kin['v_rel'], 
            omega_los=kin['omega_los'], 
            a_t=a_t, 
            r=kin['r'], 
            N=self.N
        )
        
        # 3. Compute Safety Constraints (HOCBF)
        A_cbf, b_cbf = compute_cbf_matrices(
            r=kin['r'], 
            v=kin['v'], 
            a_t=a_t, 
            d_min=self.d_min, 
            k1=self.k1, 
            k2=self.k2
        )
        
        # 4. Formulate the Quadratic Program
        # q = -a_nom (derived from expanding ||a - a_nom||^2)
        q = -a_nom
        
        # Gx <= h (The safety barrier constraint)
        G = A_cbf
        h = np.array([b_cbf])
        
        # 5. Solve the QP
        try:
            a_cmd = qpsolvers.solve_qp(
                P=self.P, 
                q=q, 
                G=G, 
                h=h, 
                lb=self.lb, 
                ub=self.ub, 
                solver=self.solver
            )
            
            # If the solver finds the constraints physically impossible (infeasible)
            if a_cmd is None:
                return self._fallback_braking(kin['r'], kin['distance'][0])
                
            return a_cmd
            
        except Exception as e:
            print(f"QP Solver Exception: {e}")
            return self._fallback_braking(kin['r'], kin['distance'][0])
            
    def _fallback_braking(self, r: np.ndarray, distance: float) -> np.ndarray:
        """Fallback maneuver if the QP solver fails to find a feasible solution."""
        if distance > 1e-6:
            # Max braking away from target
            return -self.a_max_xy * (r / distance)
        return np.zeros(3)
