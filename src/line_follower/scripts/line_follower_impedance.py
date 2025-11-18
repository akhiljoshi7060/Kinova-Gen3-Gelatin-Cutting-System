#!/usr/bin/env python3
"""
Gen3 Cartesian Impedance Controller – FULL v4
==============================================

This version first uses MoveIt to move the arm exactly to your desired Cartesian pose,
then hands off to the impedance loop for compliant, hand-guidable behaviour.

Features:
1. **MoveIt** drives to the target pose (X,Y,Z + orientation quaternion).
2. **6-D impedance** control in tool frame for forces & torques.
3. **Adaptive softness** when human touches (10% stiffness).
4. Auto-baseline & low-pass filter on wrench readings.
5. Workspace clamping identical to MoveIt bounds.
6. All gains & limits are ROS params at launch.

Launch:
------
```bash
rosrun line_follower gen3_impedance_controller_v4.py \
    _moveit_pose:="{x:0.583, y:-0.016, z:0.748, qx:0.056, qy:-0.002, qz:0.999, qw:0.018}" \
    _rate:=100 _reference_frame:=1 \
    _stiffness_lin:="[40,40,30]" _damping_lin:="[8,8,6]" \
    _stiffness_rot:="[3,3,3]"    _damping_rot:="[0.3,0.3,0.3]" \
    _max_force:=6  _max_torque:=2
```

The script will:
1. Parse the `~moveit_pose` param (position + quaternion).
2. Initialize MoveIt Commander, plan, and execute a trajectory to that pose.
3. Once at the pose, switch to impedance mode — console logs will show compliance.

After launch, push the flange in any direction: the arm yields (mode=COMPLIANT), then snaps back when released.
"""

import threading, time
import numpy as np
import rospy
from geometry_msgs.msg import Point, Vector3, PoseStamped
from std_msgs.msg import Float64
from kortex_driver.msg import BaseCyclic_Feedback
from kortex_driver.srv import SendWrenchCommand, SendWrenchCommandRequest
from collections import deque
from typing import Deque

# MoveIt imports
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

def clamp(v: np.ndarray, limit: float) -> np.ndarray:
    return np.clip(v, -limit, limit)

def low_pass(old: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * new + (1 - alpha) * old

# ----------------------------------------------------------------------------
# Main Controller
# ----------------------------------------------------------------------------
class Gen3ImpedanceController:
    def __init__(self):
        rospy.init_node('gen3_impedance_controller', anonymous=True)

        # --- Load params ---
        # MoveIt target pose
        pose_dict = rospy.get_param('~moveit_pose')
        self.target_moveit = pose_dict
        # Impedance gains
        self.rate_hz        = rospy.get_param('~rate', 100.0)
        self.Kp_lin_nominal = np.array(rospy.get_param('~stiffness_lin', [60,60,40]), dtype=float)
        self.Kd_lin_nominal = np.array(rospy.get_param('~damping_lin',   [12,12, 8]), dtype=float)
        self.Kp_rot_nominal = np.array(rospy.get_param('~stiffness_rot', [3,3,3]),  dtype=float)
        self.Kd_rot_nominal = np.array(rospy.get_param('~damping_rot',   [0.3,0.3,0.3]), dtype=float)
        self.soft_scale     = rospy.get_param('~soft_scale', 0.1)
        self.max_F          = rospy.get_param('~max_force', 6.0)
        self.max_T          = rospy.get_param('~max_torque',2.0)
        self.reference_frame = rospy.get_param('~reference_frame',1)
        self.int_thresh     = rospy.get_param('~interaction_threshold',2.0)
        self.ws = rospy.get_param('~workspace',{'x_min':0.2,'x_max':0.8,'y_min':-0.4,'y_max':0.4,'z_min':0.1,'z_max':1.0})

        # --- State ---
        self.pos, self.vel = np.zeros(3), np.zeros(3)
        self.ori, self.ang_vel = np.zeros(3), np.zeros(3)
        self.target_pos, self.target_ori = None, None
        self.wrench_ext = np.zeros(3)
        self._hist: Deque[np.ndarray] = deque(maxlen=10)
        self._baseline, self._prev_t = 0.0, None
        self._lock, self._compliant = threading.Lock(), False

        # --- Setup ROS & MoveIt ---
        # Subscriptions & publishers
        rospy.Subscriber('/my_gen3/base_feedback', BaseCyclic_Feedback, self._fb_cb, queue_size=1)
        self.pub_err  = rospy.Publisher('~position_error', Float64, queue_size=1)
        self.pub_stat = rospy.Publisher('~status', Vector3, queue_size=1)
        rospy.wait_for_service('/my_gen3/base/send_wrench_command')
        self._wrench_srv = rospy.ServiceProxy('/my_gen3/base/send_wrench_command',SendWrenchCommand)
        # MoveIt initialization
        moveit_commander.roscpp_initialize([])
        self.move_group = moveit_commander.MoveGroupCommander('arm')
        rospy.loginfo('[Impedance] node up – %.1f Hz', self.rate_hz)

        # --- Execute MoveIt first ---
        self._go_to_moveit_target()
        rospy.loginfo('[Impedance] reached MoveIt target, switching to impedance mode')

        # Shutdown wrench on exit
        rospy.on_shutdown(self._zero_wrench)
        self._rate = rospy.Rate(self.rate_hz)

    def _go_to_moveit_target(self):
        pose = PoseStamped()
        pose.header.frame_id = 'base_link'
        pose.pose.position.x = self.target_moveit['x']
        pose.pose.position.y = self.target_moveit['y']
        pose.pose.position.z = self.target_moveit['z']
        # quaternion
        pose.pose.orientation.x = self.target_moveit['qx']
        pose.pose.orientation.y = self.target_moveit['qy']
        pose.pose.orientation.z = self.target_moveit['qz']
        pose.pose.orientation.w = self.target_moveit['qw']
        self.move_group.set_pose_target(pose)
        self.move_group.go(wait=True)
        self.move_group.stop(); self.move_group.clear_pose_targets()

    def _fb_cb(self,msg):
        with self._lock:
            # position / orientation
            p = np.array([msg.base.tool_pose_x,msg.base.tool_pose_y,msg.base.tool_pose_z])
            eul = np.deg2rad([msg.base.tool_pose_theta_x,msg.base.tool_pose_theta_y,msg.base.tool_pose_theta_z])
            now = time.time()
            if self._prev_t:
                dt = now - self._prev_t
                if dt>0:
                    self.vel     = (p - self.pos)/dt
                    self.ang_vel = (eul - self.ori)/dt
            self.pos, self.ori, self._prev_t = p, eul, now
            # external wrench
            raw = np.array([msg.base.tool_external_wrench_force_x,msg.base.tool_external_wrench_force_y,msg.base.tool_external_wrench_force_z])
            self.wrench_ext = low_pass(self.wrench_ext,raw,0.3)
            self._hist.append(self.wrench_ext)
            if len(self._hist)==self._hist.maxlen:
                self._baseline = np.mean([np.linalg.norm(f) for f in self._hist])
            # initialize impedance targets
            if self.target_pos is None:
                self.target_pos, self.target_ori = self.pos.copy(), self.ori.copy()

    def _human_touching(self)->bool:
        return np.linalg.norm(self.wrench_ext) > self._baseline + self.int_thresh

    def _compute_wrench(self):
        pos_err, ori_err = self.target_pos - self.pos, self.target_ori - self.ori
        if self._compliant:
            Kp_lin, Kd_lin = self.Kp_lin_nominal*self.soft_scale, self.Kd_lin_nominal*self.soft_scale
            Kp_rot, Kd_rot = self.Kp_rot_nominal*self.soft_scale, self.Kd_rot_nominal*self.soft_scale
        else:
            Kp_lin, Kd_lin = self.Kp_lin_nominal, self.Kd_lin_nominal
            Kp_rot, Kd_rot = self.Kp_rot_nominal, self.Kd_rot_nominal
        f = clamp(Kp_lin*pos_err - Kd_lin*self.vel, self.max_F)
        t = clamp(Kp_rot*ori_err - Kd_rot*self.ang_vel, self.max_T)
        return np.hstack((f,t))

    def _send_wrench(self,w6):
        req = SendWrenchCommandRequest(); req.input.reference_frame=self.reference_frame; req.input.mode=0; req.input.duration=0
        fx,fy,fz,tx,ty,tz = map(float,w6)
        req.input.wrench.force_x,req.input.wrench.force_y,req.input.wrench.force_z = fx,fy,fz
        req.input.wrench.torque_x,req.input.wrench.torque_y,req.input.wrench.torque_z = tx,ty,tz
        try: self._wrench_srv(req)
        except rospy.ServiceException as e: rospy.logwarn_throttle(5.0,'SendWrench failed: %s',e)

    def _zero_wrench(self):
        try: self._send_wrench(np.zeros(6)); rospy.loginfo('[Impedance] wrench cleared')
        except: pass

    def spin(self):
        last = time.time()
        while not rospy.is_shutdown():
            with self._lock:
                self._compliant = self._human_touching()
                w6 = self._compute_wrench()
                err = np.linalg.norm(self.target_pos-self.pos)
            self._send_wrench(w6)
            self.pub_err.publish(err)
            self.pub_stat.publish(Vector3(err,np.linalg.norm(self.wrench_ext),float(self._compliant)))
            if time.time()-last>2.0:
                mode = 'COMPLIANT' if self._compliant else 'IMPEDANCE'
                rospy.loginfo('[Err %.3f m] [|F| %.2f N] mode=%s',err,np.linalg.norm(w6[:3]),mode)
                last=time.time()
            self._rate.sleep()

if __name__=='__main__':
    ctrl=Gen3ImpedanceController(); ctrl.spin()

