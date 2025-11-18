#!/usr/bin/env python3

import rospy
import numpy as np
import tf
from geometry_msgs.msg import PoseStamped, Pose, Twist, WrenchStamped
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from visualization_msgs.msg import MarkerArray, Marker
import math
import threading

class CartesianImpedanceController:
    def __init__(self):
        rospy.init_node('cartesian_impedance_controller')
        
        # Initialize parameters
        self.stiffness = np.diag([800.0, 800.0, 300.0, 30.0, 30.0, 30.0])  # Diagonal stiffness matrix
        self.damping = np.diag([40.0, 40.0, 40.0, 5.0, 5.0, 5.0])  # Diagonal damping matrix
        self.mass = np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])  # Diagonal mass matrix
        
        # States
        self.target_pose = None
        self.current_pose = None
        self.waypoints = []
        self.current_waypoint_index = 0
        self.is_following_waypoints = False
        self.joint_positions = None
        self.joint_velocities = None
        self.external_wrench = np.zeros(6)  # Force and torque in Cartesian space
        
        # Detection thresholds
        self.force_threshold = 10.0  # N
        self.torque_threshold = 3.0  # Nm
        self.external_force_detected = False
        self.last_position = None
        self.last_velocity_cmd = np.zeros(6)
        self.position_change_threshold = 0.005  # m
        
        # TF listener
        self.tf_listener = tf.TransformListener()
        
        # Threading and control
        self.lock = threading.Lock()
        self.running = False
        self.control_thread = None
        
        # Waypoint tolerance
        self.position_tolerance = 0.02  # meters
        
        # Subscribe to topics
        self.target_pose_sub = rospy.Subscriber("/kinova_cartesian_impedance/target_pose", PoseStamped, self.target_pose_callback)
        self.stiffness_sub = rospy.Subscriber("/kinova_cartesian_impedance/stiffness", Float64MultiArray, self.stiffness_callback)
        self.current_pose_sub = rospy.Subscriber("/my_gen3/pose_controller/pose", PoseStamped, self.current_pose_callback)
        self.waypoint_sub = rospy.Subscriber("/visualization_marker_array", MarkerArray, self.waypoint_callback)
        self.joint_states_sub = rospy.Subscriber("/my_gen3/joint_states", JointState, self.joint_states_callback)
        
        # For direct force sensing, if available
        self.wrench_sub = rospy.Subscriber("/my_gen3/ft_sensor/wrench", WrenchStamped, self.wrench_callback)
        
        # Publishers
        self.velocity_pub = rospy.Publisher("/my_gen3/cartesian_velocity_controller/cartesian_velocity", Twist, queue_size=1)
        
        rospy.loginfo("Cartesian Impedance Controller initialized")
    
    def current_pose_callback(self, msg):
        with self.lock:
            self.current_pose = msg.pose
            
            # Initialize last_position if it's the first message
            if self.last_position is None:
                self.last_position = np.array([
                    self.current_pose.position.x,
                    self.current_pose.position.y,
                    self.current_pose.position.z
                ])
    
    def target_pose_callback(self, msg):
        with self.lock:
            self.target_pose = msg.pose
            rospy.loginfo(f"Received target pose: Position=({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f})")
    
    def stiffness_callback(self, msg):
        if len(msg.data) == 6:
            with self.lock:
                for i in range(6):
                    self.stiffness[i, i] = msg.data[i]
                rospy.loginfo(f"Updated stiffness diagonal: {msg.data}")
    
    def joint_states_callback(self, msg):
        with self.lock:
            self.joint_positions = msg.position
            self.joint_velocities = msg.velocity
    
    def wrench_callback(self, msg):
        # Update external wrench from force/torque sensor
        with self.lock:
            self.external_wrench = np.array([
                msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
                msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
            ])
            
            # Check if external force exceeds threshold
            force_magnitude = np.linalg.norm(self.external_wrench[:3])
            torque_magnitude = np.linalg.norm(self.external_wrench[3:])
            
            if force_magnitude > self.force_threshold or torque_magnitude > self.torque_threshold:
                if not self.external_force_detected:
                    self.external_force_detected = True
                    rospy.loginfo(f"External force/torque detected: Force={force_magnitude:.2f}N, Torque={torque_magnitude:.2f}Nm")
            else:
                if self.external_force_detected:
                    self.external_force_detected = False
                    rospy.loginfo("External force/torque removed, continuing motion")
    
    def waypoint_callback(self, marker_array):
        # Extract waypoints from marker array
        sphere_markers = [marker for marker in marker_array.markers 
                         if marker.type == Marker.SPHERE and marker.ns == 'waypoints']
        
        if sphere_markers and len(sphere_markers) > 2:
            with self.lock:
                # Sort by ID
                sorted_markers = sorted(sphere_markers, key=lambda m: m.id)
                self.waypoints = [marker.pose for marker in sorted_markers]
                
                if not self.is_following_waypoints:
                    rospy.loginfo(f"Received {len(self.waypoints)} waypoints")
                    
                    # Ask for confirmation to start following
                    response = input("Press Enter to start following waypoints with impedance control, or 'q' to quit: ")
                    if response.lower() != 'q':
                        self.is_following_waypoints = True
                        self.current_waypoint_index = 0
                        rospy.loginfo("Starting to follow waypoints with impedance control")
                    else:
                        rospy.loginfo("Waypoint following canceled")
    
    def start(self):
        if self.running:
            rospy.logwarn("Controller is already running")
            return
        
        self.running = True
        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.start()
        rospy.loginfo("Impedance controller started")
    
    def stop(self):
        self.running = False
        if self.control_thread:
            self.control_thread.join()
        rospy.loginfo("Impedance controller stopped")
    
    def compute_robot_jacobian(self, joint_positions):
        """
        Compute the robot's Jacobian matrix
        
        This is a simplified version that should be replaced with your robot's
        specific Jacobian computation or service call
        """
        try:
            # This is where you would call a service or API specific to your robot
            # For example, with Kinova Gen3 you might use their SDK or a ROS service
            
            # For testing, we'll create a dummy Jacobian based on number of joints
            num_joints = len(joint_positions)
            jacobian = np.zeros((6, num_joints))
            
            # In a real implementation, the Jacobian would be filled based on
            # the robot's kinematics model
            
            # For testing, we'll make a simple diagonal Jacobian
            for i in range(min(num_joints, 6)):
                jacobian[i, i] = 1.0
            
            return jacobian
            
            # TODO: Replace with actual Jacobian computation or service call
            # For Kinova Gen3, check if there's a service like:
            # from kortex_driver.srv import ComputeKinematics
            # compute_kinematics = rospy.ServiceProxy('/my_gen3/base/compute_kinematics', ComputeKinematics)
            # response = compute_kinematics(input)
            # return np.array(response.jacobian).reshape((6, num_joints))
            
        except Exception as e:
            rospy.logerr(f"Error computing Jacobian: {str(e)}")
            # Return identity Jacobian in case of failure
            return np.eye(6, num_joints)
    
    def quaternion_error_to_axis_angle(self, current_quat, target_quat):
        """Convert quaternion error to axis-angle representation"""
        # Calculate quaternion difference (q_error = q_target * q_current^-1)
        q_error = tf.transformations.quaternion_multiply(
            target_quat,
            tf.transformations.quaternion_inverse(current_quat)
        )
        
        # Convert to axis-angle
        angle = 2.0 * math.acos(max(-1.0, min(1.0, q_error[3])))  # w component, clamp to [-1, 1]
        
        # Avoid division by zero
        if abs(angle) < 1e-6:
            return np.array([0.0, 0.0, 0.0])
        
        sin_half_angle = math.sin(angle/2.0)
        if abs(sin_half_angle) < 1e-6:
            return np.array([0.0, 0.0, 0.0])
        
        axis = np.array([q_error[0], q_error[1], q_error[2]]) / sin_half_angle
        
        # Return axis-angle as a 3D vector (magnitude represents angle)
        return axis * angle
    
    def detect_external_force(self):
        """
        Detect external forces by monitoring robot motion
        This is used when no force/torque sensor is available
        """
        if self.last_position is None or self.joint_velocities is None:
            return False
        
        current_position = np.array([
            self.current_pose.position.x,
            self.current_pose.position.y,
            self.current_pose.position.z
        ])
        
        # Calculate actual position change
        position_change = np.linalg.norm(current_position - self.last_position)
        
        # If commanded velocity is significant but position barely changed
        commanded_velocity = np.linalg.norm(self.last_velocity_cmd[:3])
        joint_velocity_magnitude = np.linalg.norm(self.joint_velocities)
        
        # Detect forces by checking for discrepancy between commanded and actual motion
        if commanded_velocity > 0.05 and position_change < self.position_change_threshold and joint_velocity_magnitude < 0.1:
            return True
        
        return False
    
    def control_loop(self):
        rate = rospy.Rate(50)  # 50Hz control loop
        
        while self.running and not rospy.is_shutdown():
            try:
                with self.lock:
                    # Skip if no target pose or current pose yet
                    if self.target_pose is None or self.current_pose is None:
                        rate.sleep()
                        continue
                    
                    # Check if we're following waypoints and have waypoints
                    if self.is_following_waypoints and self.waypoints:
                        # Update target pose to current waypoint
                        if self.current_waypoint_index < len(self.waypoints):
                            self.target_pose = self.waypoints[self.current_waypoint_index]
                            
                            # Check if we've reached the waypoint
                            position_error = math.sqrt(
                                (self.current_pose.position.x - self.target_pose.position.x)**2 +
                                (self.current_pose.position.y - self.target_pose.position.y)**2 +
                                (self.current_pose.position.z - self.target_pose.position.z)**2
                            )
                            
                            if position_error < self.position_tolerance:
                                rospy.loginfo(f"Reached waypoint {self.current_waypoint_index+1}/{len(self.waypoints)}")
                                self.current_waypoint_index += 1
                                
                                # If we're at the last waypoint, stop following
                                if self.current_waypoint_index >= len(self.waypoints):
                                    rospy.loginfo("Reached final waypoint. Stopping.")
                                    self.is_following_waypoints = False
                    
                    # Calculate position error
                    position_error = np.array([
                        self.target_pose.position.x - self.current_pose.position.x,
                        self.target_pose.position.y - self.current_pose.position.y,
                        self.target_pose.position.z - self.current_pose.position.z
                    ])
                    
                    # Calculate orientation error using quaternions
                    target_quat = np.array([
                        self.target_pose.orientation.x,
                        self.target_pose.orientation.y,
                        self.target_pose.orientation.z,
                        self.target_pose.orientation.w
                    ])
                    current_quat = np.array([
                        self.current_pose.orientation.x,
                        self.current_pose.orientation.y,
                        self.current_pose.orientation.z,
                        self.current_pose.orientation.w
                    ])
                    
                    # Convert quaternion error to axis-angle representation
                    orientation_error = self.quaternion_error_to_axis_angle(current_quat, target_quat)
                    
                    # Combine errors into Cartesian error vector [position; orientation]
                    cartesian_error = np.concatenate([position_error, orientation_error])
                    
                    # Get current Cartesian velocity (approximate by position difference)
                    current_position = np.array([
                        self.current_pose.position.x,
                        self.current_pose.position.y,
                        self.current_pose.position.z
                    ])
                    
                    # Calculate velocity if we have previous position
                    if self.last_position is not None:
                        cartesian_velocity = (current_position - self.last_position) * 50.0  # dt = 1/rate
                    else:
                        cartesian_velocity = np.zeros(3)
                    
                    # For angular velocity, we would ideally use the robot state
                    # Here we'll assume zero for simplicity
                    angular_velocity = np.zeros(3)
                    
                    # Combine into Cartesian velocity vector
                    cartesian_velocity_full = np.concatenate([cartesian_velocity, angular_velocity])
                    
                    # Check if external force is detected (either from sensor or motion)
                    if not self.external_force_detected and len(self.joint_positions or []) > 0:
                        self.external_force_detected = self.detect_external_force()
                        if self.external_force_detected:
                            rospy.loginfo("External force detected from motion analysis")
                    
                    # Compute desired Cartesian force from impedance law:
                    # F = Kp*x + Kd*v (where K is stiffness, D is damping)
                    cartesian_force = np.dot(self.stiffness, cartesian_error) - np.dot(self.damping, cartesian_velocity_full)
                    
                    # Whether to apply force or stop
                    if self.external_force_detected:
                        # Send zero velocity to stop the robot
                        twist_msg = Twist()
                        self.velocity_pub.publish(twist_msg)
                        self.last_velocity_cmd = np.zeros(6)
                        rospy.loginfo_throttle(1.0, "Robot stopped due to external force")
                    else:
                        # Convert force to velocity command for velocity-controlled robots
                        # v = M^-1 * F (where M is the virtual mass matrix)
                        cartesian_velocity_cmd = np.dot(np.linalg.inv(self.mass), cartesian_force) * 0.01  # Scaling factor
                        
                        # Limit velocities for safety
                        max_lin_vel = 0.1  # m/s
                        max_ang_vel = 0.2  # rad/s
                        
                        cartesian_velocity_cmd[:3] = np.clip(cartesian_velocity_cmd[:3], -max_lin_vel, max_lin_vel)
                        cartesian_velocity_cmd[3:] = np.clip(cartesian_velocity_cmd[3:], -max_ang_vel, max_ang_vel)
                        
                        # Create Twist message
                        twist_msg = Twist()
                        twist_msg.linear.x = cartesian_velocity_cmd[0]
                        twist_msg.linear.y = cartesian_velocity_cmd[1]
                        twist_msg.linear.z = cartesian_velocity_cmd[2]
                        twist_msg.angular.x = cartesian_velocity_cmd[3]
                        twist_msg.angular.y = cartesian_velocity_cmd[4]
                        twist_msg.angular.z = cartesian_velocity_cmd[5]
                        
                        # Save velocity command for force detection
                        self.last_velocity_cmd = cartesian_velocity_cmd
                        
                        # Publish the command
                        self.velocity_pub.publish(twist_msg)
                        
                        if self.is_following_waypoints:
                            rospy.loginfo_throttle(1.0, f"Following waypoint {self.current_waypoint_index+1}/{len(self.waypoints)}, "
                                                f"Position error: {np.linalg.norm(position_error):.4f}m")
                    
                    # Update last position for next iteration
                    self.last_position = current_position
                    
            except Exception as e:
                rospy.logerr(f"Error in control loop: {str(e)}")
            
            rate.sleep()

def main():
    controller = CartesianImpedanceController()
    controller.start()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
    finally:
        controller.stop()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
