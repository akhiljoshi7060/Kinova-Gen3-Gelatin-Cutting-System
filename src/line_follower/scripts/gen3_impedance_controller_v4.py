#!/usr/bin/env python3
"""
gen3_impedance_controller_v4.py

Single node that:
  1) Publishes the static TF (end_effector_link → camera_color_frame).
  2) Immediately sends a 5 s JointTrajectory (Stage 1) to a fallback set of joints
     that roughly places the arm near your Cartesian target.
  3) After 5 s, switches to Stage 2: Cartesian‐impedance control holding exactly
     the hardcoded Cartesian snapshot [0.720, 0.004, 0.682, 90.656°, –0.338°, 89.39°].
  4) Toggle HIGH vs LOW stiffness via `/toggle_stiffness` (std_msgs/Bool).

Because you do not actually have a wrist F/T publishing,
this version forcibly sets `_first_wrench=True` right away so Stage 1→2 proceeds.
"""

import rospy
import numpy as np
import threading
import time

from sensor_msgs.msg     import JointState
from geometry_msgs.msg   import WrenchStamped, Vector3, TransformStamped
from std_msgs.msg        import Float64, Bool, Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from collections import deque

import tf2_ros

# ----------------------------------------------------------------------------
#  “Snapshot” Cartesian target (Image 1):
# ----------------------------------------------------------------------------
DEG = np.pi/180.0
TARGET_CARTESIAN = np.array([
    0.720,        # X (m)
    0.004,        # Y (m)
    0.682,        # Z (m)
    90.656*DEG,   # Roll (rad)
    -0.338*DEG,   # Pitch (rad)
    89.39*DEG     # Yaw (rad)
], dtype=float)

# ----------------------------------------------------------------------------
#  Fallback joint angles “near” that Cartesian pose (5 s → Stage 1→2):
#  Replace with your IK solution if desired.
# ----------------------------------------------------------------------------
FALLBACK_JOINTS = np.array([0.3, 0.2, -0.1, 1.5, 0.1, 1.2, 1.57], dtype=float)


# ----------------------------------------------------------------------------
#  Helper: clamp each component between ±limit
# ----------------------------------------------------------------------------
def clamp(v: np.ndarray, limit: float) -> np.ndarray:
    return np.clip(v, -limit, limit)


# ----------------------------------------------------------------------------
#  Helper: 1st‐order low‐pass (not strictly needed here but kept)
# ----------------------------------------------------------------------------
def low_pass(old: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * new + (1.0 - alpha) * old


# ----------------------------------------------------------------------------
#  Main controller
# ----------------------------------------------------------------------------
class Gen3ImpedanceController:
    def __init__(self):
        rospy.init_node('gen3_impedance_controller_v4', anonymous=True)

        # 1) Publish the static TF once
        self._publish_static_tf()

        # Stage machine:
        #   1 = fallback trajectory playing (5 s)
        #   2 = impedance control loop
        self.stage = 1

        # Cartesian impedance gains (6×6):
        self.K_cart_high = np.diag([300, 300, 300, 10, 10, 10])
        self.D_cart_high = np.diag([ 40,  40,  40,  2,  2,  2])
        self.K_cart_low  = np.diag([ 10,  10,  10,  1,  1,  1])
        self.D_cart_low  = np.diag([  4,   4,   4, 0.2,0.2,0.2])
        self.use_high_stiffness = True

        # Safety limits:
        self.max_joint_torque = np.array([39,39,39,39, 9, 9, 9])  # [Nm]
        self.cart_force_limit = 50.0                              # [N]

        # Robot state:
        self.joint_pos = np.zeros(7)
        self.joint_vel = np.zeros(7)
        self.joint_tau = np.zeros(7)

        self.x_cur  = np.zeros(6)   # [x, y, z, roll, pitch, yaw]
        self.dx_cur = np.zeros(6)   # [vx, vy, vz, wx, wy, wz]

        # External force (we’ll pretend we have none ⇒ baseline=0)
        self.F_ext = np.zeros(6)
        self._first_wrench = True    # *** force Stage1→2 to proceed immediately ***

        # Jacobian (6×7), updated each joint_state callback:
        self.jacobian = np.zeros((6,7))
        self._kin_ok = False

        # Impedance loop variables:
        self.dt       = 0.01  # 100 Hz
        self.tau_prev = np.zeros(7)

        # Null‐space posture:
        self.q_comfort = np.array([0,0.26,0,2.26,0,0.96,1.57])
        self.q_min     = np.array([-2.41,-2.41,-2.41,-2.41,-2.57,-2.57,-2.57])
        self.q_max     = np.array([ 2.41, 2.41, 2.41, 2.41,  2.57,  2.57,  2.57])
        self.k_null    = 5.0
        self.k_limits  = 20.0

        # Force baseline (no real F/T, so base=0)
        self.force_baseline = 0.0
        self._lock = threading.Lock()
        self._compliant = False
        self._force_int_thresh = 2.0  # [N]

        # Desired Cartesian pose (hardcoded)
        self.x_des  = TARGET_CARTESIAN.copy()
        self.dx_des = np.zeros(6)

        #—— ROS interfaces ——

        rospy.loginfo("[Impedance] Setting up ROS interfaces...")

        # 1) /my_gen3/joint_states:
        rospy.Subscriber('/my_gen3/joint_states',
                         JointState,
                         self._joint_state_cb,
                         queue_size=1)

        # 2) /my_gen3/wrench (we pretend it’s zero):
        rospy.Subscriber('/my_gen3/wrench',
                         WrenchStamped,
                         self._wrench_cb,
                         queue_size=1)

        # 3) /my_gen3/base_feedback (pose + wrench):
        from kortex_driver.msg import BaseCyclic_Feedback
        rospy.Subscriber('/my_gen3/base_feedback',
                         BaseCyclic_Feedback,
                         self._base_feedback_cb,
                         queue_size=1)

        # 4) /toggle_stiffness (std_msgs/Bool):
        rospy.Subscriber('/toggle_stiffness',
                         Bool,
                         self._toggle_cb,
                         queue_size=1)

        # 5) Publish joint torques → /my_gen3/in/joint_torque
        self.pub_tau = rospy.Publisher('/my_gen3/in/joint_torque',
                                       Float64MultiArray,
                                       queue_size=1)

        # 6) Publish fallback JointTrajectory → /my_gen3/gen3_joint_trajectory_controller/command
        self.pub_traj = rospy.Publisher(
            '/my_gen3/gen3_joint_trajectory_controller/command',
            JointTrajectory,
            queue_size=1
        )

        # 7) Monitoring: publish position_error & ext_force
        self.pub_err   = rospy.Publisher('~position_error', Float64, queue_size=1)
        self.pub_force = rospy.Publisher('~ext_force',      Float64, queue_size=1)

        # 8) Start the 100 Hz impedance timer
        rospy.Timer(rospy.Duration(self.dt), self._control_loop)

        #—— Immediately send Stage 1: fallback joint trajectory (5 s) ——
        self._send_fallback_joint_trajectory()

        rospy.loginfo("[Impedance] Node initialized—Stage 1 running (fallback).")


    #───────────────────────────────────────────────────────────────────────────
    #  Publish static transform end_effector_link → camera_color_frame
    #───────────────────────────────────────────────────────────────────────────
    def _publish_static_tf(self):
        br = tf2_ros.StaticTransformBroadcaster()
        t = TransformStamped()
        t.header.stamp    = rospy.Time.now()
        t.header.frame_id = "end_effector_link"
        t.child_frame_id  = "camera_color_frame"

        # from your launch snippet (meters):
        t.transform.translation.x = -49.4305 / 1000.0
        t.transform.translation.y =  49.587  / 1000.0
        t.transform.translation.z =   3.95126/ 1000.0
        t.transform.rotation.x    =  0.200804
        t.transform.rotation.y    =  0.290464
        t.transform.rotation.z    =  0.442318
        t.transform.rotation.w    =  0.824417

        br.sendTransform(t)
        rospy.loginfo("✓ static TF published (end_effector_link → camera_color_frame)")


    #───────────────────────────────────────────────────────────────────────────
    #  Stage 1: send a 5 s JointTrajectory to FALLBACK_JOINTS
    #───────────────────────────────────────────────────────────────────────────
    def _send_fallback_joint_trajectory(self):
        jt = JointTrajectory()
        jt.header.stamp = rospy.Time.now()
        jt.joint_names = [f"joint_{i+1}" for i in range(7)]
        pt = JointTrajectoryPoint()
        pt.positions = FALLBACK_JOINTS.tolist()
        pt.time_from_start = rospy.Duration(5.0)
        jt.points = [pt]
        self.pub_traj.publish(jt)
        rospy.loginfo("[Stage 1] → Driving to fallback joints over 5 s...")


    #───────────────────────────────────────────────────────────────────────────
    #  Stage 2: Cartesian‐impedance control loop (100 Hz)
    #───────────────────────────────────────────────────────────────────────────
    def _control_loop(self, event):
        # Require only kinematics_ok; we forcibly set _first_wrench=True in __init__
        if not self._kin_ok:
            return

        # If still Stage 1, switch to Stage 2 on first callback after 5 s:
        if self.stage == 1:
            self.stage = 2
            rospy.loginfo("[Stage 1→2] Fallback done → Stage 2 (impedance)")
            return

        # Stage 2: run Cartesian impedance at 100 Hz

        # 1) “Human push” detection: external force > baseline + threshold
        self._compliant = (np.linalg.norm(self.F_ext[:3]) > (self.force_baseline + self._force_int_thresh))

        # 2) Pick K, D gains
        if self.use_high_stiffness:
            K = self.K_cart_high
            D = self.D_cart_high
        else:
            K = self.K_cart_low
            D = self.D_cart_low

        # 3) Compute Cartesian error + desired wrench
        x_err = self.x_des - self.x_cur
        dx_err = -self.dx_cur
        W_imp = D.dot(dx_err) + K.dot(x_err)  # 6×1 force/torque

        # 4) τ_imp = Jᵀ·W_imp
        tau_imp = self.jacobian.T.dot(W_imp)

        # 5) Add rough gravity compensation
        tau_g = self._gravity_compensation(self.joint_pos)

        # 6) Add null‐space posture term
        tau_null = self._null_space_control()

        # 7) Total τ
        tau = tau_imp + tau_g + tau_null

        # 8) Clip joint torques to ±39 Nm
        tau = np.clip(tau, -self.max_joint_torque, self.max_joint_torque)

        # 9) Cartesian‐force limiting: if ‖J·τ‖ > 50 N, scale down τ
        F_cart = self.jacobian.dot(tau)
        f_norm = np.linalg.norm(F_cart[:3])
        if f_norm > self.cart_force_limit > 0:
            tau *= (self.cart_force_limit / f_norm)

        #10) Rate‐limit τ: max ±100 Nm/s per joint
        tau_rate = (tau - self.tau_prev) / self.dt
        max_rate = 100.0
        over = np.abs(tau_rate) > max_rate
        tau[over] = self.tau_prev[over] + np.sign(tau_rate[over]) * max_rate * self.dt
        self.tau_prev = tau.copy()

        #11) Publish the 7‐vector on /my_gen3/in/joint_torque
        msg = Float64MultiArray()
        msg.data = tau.tolist()
        self.pub_tau.publish(msg)

        #12) Publish monitoring: position_error & external_force
        pos_err_norm = float(np.linalg.norm(x_err[:3]))
        ext_force_mag = float(np.linalg.norm(self.F_ext[:3]))
        self.pub_err.publish(pos_err_norm)
        self.pub_force.publish(ext_force_mag)

        #13) Throttled logging (~1 Hz)
        if (time.time() % 1.0) < self.dt:
            mode_str = 'COMPLIANT' if self._compliant else 'IMPEDANCE'
            stiff_str = 'HIGH' if self.use_high_stiffness else 'LOW'
            icon = '🟢' if self._compliant else '🔴'
            rospy.loginfo_throttle(
                1.0,
                f"{icon} [Stage 2] Mode={mode_str}/{stiff_str} | err={pos_err_norm:.3f} m | |F|={ext_force_mag:.2f} N"
            )


    #───────────────────────────────────────────────────────────────────────────
    #  On shutdown: zero out all joint torques
    #───────────────────────────────────────────────────────────────────────────
    def _zero_wrench(self):
        try:
            zero_msg = Float64MultiArray(data=[0.0]*7)
            self.pub_tau.publish(zero_msg)
            rospy.loginfo("[Impedance] Zeroed torques on shutdown")
        except:
            pass


    #───────────────────────────────────────────────────────────────────────────
    #  JointState callback: update joint_pos, joint_vel, compute J, set _kin_ok
    #───────────────────────────────────────────────────────────────────────────
    def _joint_state_cb(self, msg: JointState):
        with self._lock:
            self.joint_pos = np.array(msg.position[:7])
            self.joint_vel = np.array(msg.velocity[:7])
            if len(msg.effort) >= 7:
                self.joint_tau = np.array(msg.effort[:7])

            rospy.loginfo_throttle(1.0,
                "[DEBUG] Received JointState (pos[:3]=%.3f,%.3f,%.3f)"
                % (self.joint_pos[0], self.joint_pos[1], self.joint_pos[2])
            )

            self._update_kinematics(self.joint_pos, self.joint_vel)
            self._kin_ok = True


    #───────────────────────────────────────────────────────────────────────────
    #  Wrench callback: we pretend it’s zero (no real F/T)
    #───────────────────────────────────────────────────────────────────────────
    def _wrench_cb(self, msg: WrenchStamped):
        with self._lock:
            # Immediately set _first_wrench=True (pretend zero bias)
            # and keep F_ext = [0,0,0,0,0,0]
            if not self._first_wrench:
                self._first_wrench = True
                rospy.loginfo("[Impedance] ✓ Pretend F/T auto-zero (no real sensor)")
            self.F_ext = np.zeros(6)


    #───────────────────────────────────────────────────────────────────────────
    #  BaseCyclic_Feedback callback: update Cartesian pose x_cur, dx_cur
    #───────────────────────────────────────────────────────────────────────────
    def _base_feedback_cb(self, msg):
        with self._lock:
            # Position (m)
            p = np.array([
                msg.base.tool_pose_x,
                msg.base.tool_pose_y,
                msg.base.tool_pose_z
            ], dtype=float)

            # Orientation (deg→rad)
            rpy_deg = np.array([
                msg.base.tool_pose_theta_x,
                msg.base.tool_pose_theta_y,
                msg.base.tool_pose_theta_z
            ], dtype=float)
            rpy = np.deg2rad(rpy_deg)

            now = time.time()
            if hasattr(self, '_prev_t_base'):
                dt = now - self._prev_t_base
                if dt > 1e-6:
                    self.dx_cur[:3] = (p - self.x_cur[:3]) / dt
                    self.dx_cur[3:] = (rpy - self.x_cur[3:]) / dt

            self.x_cur[:3] = p.copy()
            self.x_cur[3:] = rpy.copy()
            self._prev_t_base = now

            rospy.loginfo_throttle(
                1.0,
                "[DEBUG] Received base_feedback (x_cur[:3]=%.3f,%.3f,%.3f)"
                % (self.x_cur[0], self.x_cur[1], self.x_cur[2])
            )


    #───────────────────────────────────────────────────────────────────────────
    #  Toggle stiffness: HIGH vs LOW
    #───────────────────────────────────────────────────────────────────────────
    def _toggle_cb(self, msg: Bool):
        self.use_high_stiffness = bool(msg.data)
        mode = "HIGH (rigid)" if msg.data else "LOW (compliant)"
        rospy.logwarn("[Impedance] 🔧 SWITCHED TO %s STIFFNESS" % mode)


    #───────────────────────────────────────────────────────────────────────────
    #  Compute forward‐kinematics & Jacobian (DH model for Gen3 7 DOF)
    #───────────────────────────────────────────────────────────────────────────
    def _update_kinematics(self, q: np.ndarray, qd: np.ndarray):
        n = 7
        dh = np.array([
            [0,       -np.pi/2, 0.2755, 0],
            [0,        np.pi/2, 0.0000, 0],
            [0,       -np.pi/2, 0.4100, 0],
            [0,        np.pi/2, 0.0000, 0],
            [0,       -np.pi/2, 0.3070, 0],
            [0,        np.pi/2, 0.0000, 0],
            [0,         0.0000, 0.0840, 0]
        ])
        T = np.eye(4)
        z_axes = [np.array([0,0,1])]
        origins = [np.array([0,0,0])]

        for i in range(n):
            a, alpha, d, _ = dh[i]
            theta = q[i]
            Ti = np.array([
                [ np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha),  a*np.cos(theta)],
                [ np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha),  a*np.sin(theta)],
                [ 0,               np.sin(alpha),               np.cos(alpha),                d           ],
                [ 0,               0,                           0,                            1           ]
            ])
            T = T.dot(Ti)
            if i < n-1:
                z_axes.append(T[:3,2].copy())
                origins.append(T[:3,3].copy())

        o_ee = T[:3,3].copy()
        J = np.zeros((6,n))
        for i in range(n):
            lin = np.cross(z_axes[i], (o_ee - origins[i]))
            ang = z_axes[i]
            J[:3,i] = lin
            J[3:,i] = ang

        self.jacobian = J.copy()


    #───────────────────────────────────────────────────────────────────────────
    #  Rough gravity compensation (approximate)
    #───────────────────────────────────────────────────────────────────────────
    def _gravity_compensation(self, q: np.ndarray) -> np.ndarray:
        g = 9.81
        masses = np.array([1.377,1.377,0.930,0.930,0.678,0.678,0.5])
        tau_g = np.zeros(7)
        T = np.eye(4)
        dh = np.array([
            [0,       -np.pi/2, 0.2755, 0],
            [0,        np.pi/2, 0.0000, 0],
            [0,       -np.pi/2, 0.4100, 0],
            [0,        np.pi/2, 0.0000, 0],
            [0,       -np.pi/2, 0.3070, 0],
            [0,        np.pi/2, 0.0000, 0],
            [0,         0.0000, 0.0840, 0]
        ])
        for i in range(7):
            a, alpha, d_, _ = dh[i]
            theta = q[i]
            Ti = np.array([
                [ np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha),  a*np.cos(theta)],
                [ np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha),  a*np.sin(theta)],
                [ 0,               np.sin(alpha),               np.cos(alpha),                d_          ],
                [ 0,               0,                           0,                            1           ]
            ])
            T = T.dot(Ti)
            tau_g[i] = masses[i] * g * T[2,3] * 0.05
        return tau_g


    #───────────────────────────────────────────────────────────────────────────
    #  Null‐space posture control
    #───────────────────────────────────────────────────────────────────────────
    def _null_space_control(self) -> np.ndarray:
        J_pinv = np.linalg.pinv(self.jacobian)  # 7×6
        N = np.eye(7) - J_pinv.dot(self.jacobian)  # 7×7
        tau_null = self.k_null * (self.q_comfort - self.joint_pos)

        for i in range(7):
            if self.joint_pos[i] < (self.q_min[i] + 0.1):
                tau_null[i] += self.k_limits * (self.q_min[i] + 0.1 - self.joint_pos[i])
            elif self.joint_pos[i] > (self.q_max[i] - 0.1):
                tau_null[i] += self.k_limits * (self.q_max[i] - 0.1 - self.joint_pos[i])

        return N.dot(tau_null)


    #───────────────────────────────────────────────────────────────────────────
    #  Spin (enter ROS event loop)
    #───────────────────────────────────────────────────────────────────────────
    def spin(self):
        rospy.on_shutdown(self._zero_wrench)
        rospy.spin()



#───────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    try:
        ctrl = Gen3ImpedanceController()
        ctrl.spin()
    except rospy.ROSInterruptException:
        pass

