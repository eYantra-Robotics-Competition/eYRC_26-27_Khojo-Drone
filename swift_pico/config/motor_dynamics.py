"""
Motor dynamics for Swift Pico in MuJoCo.

Physical constants calibrated to match the real Swift Pico hardware:

  Motor physics:
    Kt  = 8.54858e-06  N/(rad/s)^2   motor constant
    Km  = 0.016                       moment constant
    TAU_UP   = 0.0125  s              spin-up lag time constant
    TAU_DOWN = 0.025   s              spin-down lag time constant
    OMEGA_MAX = 800.0  rad/s          max rotor speed
    C_DRAG    = 8.06428e-05           rotor drag coefficient
    C_ROLL    = 1e-06                 rolling moment coefficient

  Body:
    mass = 1.5 kg,  Ixx=0.0347563, Iyy=0.07, Izz=0.0977 kg·m²

  Motor positions (body frame):
    m1 (FR, ccw): ( 0.19, -0.19, -0.12)
    m2 (BL, ccw): (-0.19,  0.19, -0.12)
    m3 (FL, cw):  ( 0.19,  0.19, -0.12)
    m4 (BR, cw):  (-0.19, -0.19, -0.12)
"""

import numpy as np

# ── Physical constants (calibrated to the real Swift Pico hardware) ────────
MASS      = 1.5           # kg
GRAVITY   = 9.81          # m/s²
WEIGHT    = MASS * GRAVITY # 14.715 N

KT        = 8.54858e-06   # N/(rad/s)²  — motorConstant
KM        = 0.016         # —            momentConstant
OMEGA_MAX = 800.0         # rad/s        maxRotVelocity
F_MAX     = KT * OMEGA_MAX**2  # ≈ 5.47 N per motor
F_HOVER   = WEIGHT / 4    # ≈ 3.68 N per motor at hover

# Motor time constants (first-order lag model)
TAU_UP   = 0.0125   # s — timeConstantUp   (spin-up)
TAU_DOWN = 0.025    # s — timeConstantDown (spin-down)

# Aerodynamic disturbance coefficients
C_DRAG = 8.06428e-05  # rotorDragCoefficient  — drag force opposing translation
C_ROLL = 1e-06        # rollingMomentCoefficient — gyroscopic rolling moment


# ── Motor dynamics (first-order lag + rotor drag/gyroscopic model) ──────────
class MotorDynamics:
    """
    Models the full motor dynamics of the real Swift Pico:

    1. First-order lag filter on motor omega (timeConstantUp / timeConstantDown)
       dω/dt = (ω_cmd - ω) / τ,  τ = TAU_UP if speeding up else TAU_DOWN

    2. Omega clamped to [0, OMEGA_MAX]  (maxRotVelocity = 800 rad/s)

    3. Thrust force:   F_i = Kt * ω_i²   (applied via actuator gear in XML)

    4. Rotor drag force (rotorDragCoefficient):
       Opposes horizontal body translation, proportional to rotor speed.
       F_drag = -C_DRAG * Σ|ω_i| * v_horizontal_world
       Applied as world-frame force on drone body via xfrc_applied.

    5. Rolling moment (rollingMomentCoefficient):
       Gyroscopic precession — spinning rotor moving through air creates a
       pitching/rolling moment perpendicular to its velocity.
       M_roll = C_ROLL * Σ(dir_i * ω_i) * cross(ẑ_world, v_world)
       Applied as world-frame torque on drone body via xfrc_applied.

    Motor spin directions: m1,m2 ccw (+1),  m3,m4 cw (-1)
    """

    # Spin directions per motor [m1, m2, m3, m4]
    DIRS = np.array([+1.0, +1.0, -1.0, -1.0])

    def __init__(self):
        self.omega = np.zeros(4)   # actual motor angular velocities [rad/s]

    def reset(self):
        self.omega[:] = 0.0

    def step(self, forces_cmd: np.ndarray, dt: float) -> np.ndarray:
        """
        Advance motor dynamics by one physics timestep.

        Args:
            forces_cmd: desired thrust forces [N] from controller (4,)
            dt:         physics timestep [s]

        Returns:
            forces_actual: actual thrust forces [N] after lag filter (4,)
        """
        # Commanded omegas from desired forces
        omega_cmd = np.sqrt(np.maximum(0.0, forces_cmd) / KT)

        # First-order lag: different time constants for spin-up vs spin-down
        for i in range(4):
            tau = TAU_UP if omega_cmd[i] > self.omega[i] else TAU_DOWN
            self.omega[i] += (omega_cmd[i] - self.omega[i]) / tau * dt
            self.omega[i]  = np.clip(self.omega[i], 0.0, OMEGA_MAX)

        return KT * self.omega ** 2

    def apply_disturbances(self, model, data, body_id: int):
        """
        Apply rotor drag and rolling moment to the drone body via xfrc_applied.
        Must be called each step after self.step().

        Args:
            body_id: MuJoCo body index of the drone body
        """
        v_world = data.qvel[0:3]   # world-frame linear velocity

        # ── Rotor drag ────────────────────────────────────────────────────
        # Each spinning rotor creates a drag force on the body opposing
        # horizontal translation.  Force ∝ rotor_speed × horizontal_velocity.
        # Sum over all 4 motors (drag adds regardless of spin direction).
        total_omega = np.sum(self.omega)
        F_drag = -C_DRAG * total_omega * np.array([v_world[0], v_world[1], 0.0])

        # ── Rolling / gyroscopic moment ───────────────────────────────────
        # Net angular momentum about body Z from all rotors.
        # When drone translates, this angular momentum creates a precession
        # torque perpendicular to the velocity: M = C_ROLL * L_z × v
        # L_z (signed) = Σ dir_i * ω_i
        L_z = float(np.dot(self.DIRS, self.omega))
        # cross(ẑ, v_world) = [v_y, -v_x, 0]
        M_gyro = C_ROLL * L_z * np.array([v_world[1], -v_world[0], 0.0])

        # Write to xfrc_applied [fx, fy, fz, tx, ty, tz] in world frame at COM
        data.xfrc_applied[body_id, 0:3] = F_drag
        data.xfrc_applied[body_id, 3:6] = M_gyro
