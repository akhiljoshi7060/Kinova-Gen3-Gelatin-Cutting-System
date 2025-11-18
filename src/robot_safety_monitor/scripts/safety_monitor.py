#!/usr/bin/env python3

import rospy
import tf2_ros
import numpy as np
from geometry_msgs.msg import Point, Quaternion, PoseStamped, TwistStamped, Pose
from visualization_msgs.msg import Marker, MarkerArray
from moveit_msgs.msg import PlanningScene, PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState
import actionlib
from moveit_msgs.msg import MoveGroupAction, MoveGroupGoal


class EnhancedSafetyMonitor:
    def __init__(self):
        rospy.init_node('safety_monitor')
        
        # Get bounds from ROS parameters or use defaults
        self.min_x = rospy.get_param('~min_x', -0.5)
        self.max_x = rospy.get_param('~max_x', 0.8) 
        self.min_y = rospy.get_param('~min_y', -0.5)
        self.max_y = rospy.get_param('~max_y', 0.5)
        self.min_z = rospy.get_param('~min_z', 0.0)
        self.max_z = rospy.get_param('~max_z', 1.0)
        
        # Safety margin to stop before hitting boundary
        self.safety_margin = rospy.get_param('~safety_margin', 0.05)
        
        # Enhanced safety parameters
        self.violation_threshold = 3  # Number of consecutive violations before emergency stop
        self.violation_count = 0
        self.max_velocity_near_boundary = 0.1  # m/s - slow down near boundaries
        self.boundary_proximity_threshold = 0.1  # Distance from boundary to start slowing down
        
        # TF listener to get robot positions
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Publisher for visualization markers
        self.marker_pub = rospy.Publisher('/safety_bounds_visualization', 
                                         MarkerArray, queue_size=1, latch=True)
        
        # Publisher for emergency stop signal
        self.estop_pub = rospy.Publisher('/emergency_stop', Bool, queue_size=1)
        
        # Enhanced command interception system
        self.setup_command_interceptors()
        
        # Emergency stop flag and recovery
        self.emergency_stop_active = False
        self.last_safe_pose = None
        self.recovery_position = self.get_safe_center_position()
        
        # MoveIt action client for emergency recovery
        self.move_group_client = actionlib.SimpleActionClient('move_group', MoveGroupAction)
        
        # Wait for services and action servers
        rospy.sleep(2.0)
        
        # Create and publish the workspace boundary visualization
        self.publish_boundary_markers()
        
        # Publish markers repeatedly to ensure they remain visible
        rospy.Timer(rospy.Duration(1.0), self.republish_markers)
        
        # Enhanced monitoring with multiple frequencies and direct emergency stop
        rospy.Timer(rospy.Duration(0.02), self.check_robot_position_with_immediate_stop)  # 50 Hz critical safety
        rospy.Timer(rospy.Duration(0.1), self.check_boundary_proximity)  # 10 Hz for velocity scaling
        
        rospy.loginfo("Enhanced Safety Monitor initialized with ACTIVE BOUNDARY ENFORCEMENT")
        rospy.loginfo(f"Workspace bounds: X:[{self.min_x}, {self.max_x}], Y:[{self.min_y}, {self.max_y}], Z:[{self.min_z}, {self.max_z}]")
        rospy.loginfo(f"Safety margin: {self.safety_margin}m, Violation threshold: {self.violation_threshold}")
        rospy.loginfo("Robot movement will be actively constrained to workspace bounds!")

    def setup_command_interceptors(self):
        """Enhanced command interception system"""
        # Wait for topics to be available
        rospy.sleep(1.0)
        all_topics = rospy.get_published_topics()
        topic_names = [topic[0] for topic in all_topics]
        
        rospy.loginfo("Setting up enhanced command interceptors...")
        
        # Enhanced topic detection with Kinova Gen3 support
        cartesian_topics = [
            '/cartesian_impedance/target',
            '/cartesian_impedance_example_controller/target',
            '/cartesian_position_controller/target_pose',
            '/cmd_cartesian',
            '/cartesian_command',
            '/pose_target',
            '/move_group/goal',
            '/cartesian_pose',
            '/arm_controller/cartesian/pose',
            '/ur_hardware_interface/script_command'
        ]
        
        velocity_topics = [
            '/cartesian_impedance/velocity',
            '/cartesian_velocity_controller/command',
            '/cmd_vel_cartesian', 
            '/cartesian_velocity',
            '/cmd_cartesian_velocity',
            '/arm_controller/cartesian/velocity',
            '/my_gen3/in/cartesian_velocity'  # Kinova Gen3 velocity topic
        ]
        
        # Kinova Gen3 trajectory action topics
        trajectory_action_topics = [
            '/my_gen3/cartesian_trajectory_controller/follow_cartesian_trajectory'
        ]
        
        # Set up interceptors
        self.active_interceptors = {}
        
        # Kinova Gen3 velocity interceptor (this is the main one for your robot)
        velocity_topic = '/my_gen3/in/cartesian_velocity'
        if velocity_topic in topic_names:
            rospy.loginfo(f"Found Kinova Gen3 velocity topic: {velocity_topic}")
            # Create unsafe topic for interception
            unsafe_topic = velocity_topic + '_unsafe'
            
            # Subscribe to unsafe commands
            sub = rospy.Subscriber(unsafe_topic, TwistStamped,
                                 lambda msg: self.enhanced_velocity_callback(msg, velocity_topic))
            # Publisher for filtered commands
            pub = rospy.Publisher(velocity_topic, TwistStamped, queue_size=1)
            
            self.active_interceptors[velocity_topic] = {
                'subscriber': sub,
                'publisher': pub,
                'type': 'velocity'
            }
            rospy.loginfo(f"Kinova Gen3 safety interceptor active: publish to {unsafe_topic}")
        
        # Kinova Gen3 trajectory action interceptor
        trajectory_topic = '/my_gen3/cartesian_trajectory_controller/follow_cartesian_trajectory'
        if trajectory_topic + '/goal' in topic_names:
            rospy.loginfo(f"Found Kinova Gen3 trajectory topic: {trajectory_topic}")
            # Set up action client to intercept trajectory goals
            from control_msgs.msg import FollowJointTrajectoryAction
            # We'll monitor the action server and potentially cancel unsafe trajectories
            self.trajectory_client = actionlib.SimpleActionClient(
                trajectory_topic, FollowJointTrajectoryAction)
        
        # Standard cartesian position interceptors
        for topic in cartesian_topics:
            if topic in topic_names:
                rospy.loginfo(f"Setting up cartesian interceptor for: {topic}")
                # Create interceptor
                original_topic = topic + '_original'
                
                # Subscribe to original commands (users should publish to _original topics)
                sub = rospy.Subscriber(original_topic, PoseStamped, 
                                     lambda msg, t=topic: self.enhanced_cartesian_callback(msg, t))
                
                # Publisher for filtered commands
                pub = rospy.Publisher(topic, PoseStamped, queue_size=1)
                
                self.active_interceptors[topic] = {
                    'subscriber': sub,
                    'publisher': pub,
                    'type': 'cartesian'
                }
        
        # Standard velocity interceptors  
        for topic in velocity_topics:
            if topic in topic_names and topic not in self.active_interceptors:
                rospy.loginfo(f"Setting up velocity interceptor for: {topic}")
                original_topic = topic + '_original'
                
                sub = rospy.Subscriber(original_topic, TwistStamped,
                                     lambda msg, t=topic: self.enhanced_velocity_callback(msg, t))
                pub = rospy.Publisher(topic, TwistStamped, queue_size=1)
                
                self.active_interceptors[topic] = {
                    'subscriber': sub,
                    'publisher': pub,
                    'type': 'velocity'
                }
        
        # If no interceptors were set up, create a direct joint override
        if not self.active_interceptors:
            rospy.logwarn("No standard topics found - setting up joint-level safety override")
            self.setup_joint_level_safety()
        else:
            rospy.loginfo(f"Set up {len(self.active_interceptors)} command interceptors")
            for topic, info in self.active_interceptors.items():
                rospy.loginfo(f"  - {info['type']}: {topic} (subscribe to {topic}_original)")

    def get_safe_center_position(self):
        """Calculate a safe center position within the workspace"""
        center_x = (self.min_x + self.max_x) / 2.0
        center_y = (self.min_y + self.max_y) / 2.0
        center_z = (self.min_z + self.max_z) / 2.0
        return (center_x, center_y, center_z)

    def is_position_safe(self, x, y, z, use_margin=True):
        """Enhanced safety check with optional margin"""
        margin = self.safety_margin if use_margin else 0.0
        
        return (self.min_x + margin <= x <= self.max_x - margin and
                self.min_y + margin <= y <= self.max_y - margin and
                self.min_z + margin <= z <= self.max_z - margin)
    
    def clamp_position_to_bounds(self, x, y, z):
        """Clamp position to safe bounds with margin"""
        clamped_x = max(self.min_x + self.safety_margin, 
                       min(x, self.max_x - self.safety_margin))
        clamped_y = max(self.min_y + self.safety_margin, 
                       min(y, self.max_y - self.safety_margin))
        clamped_z = max(self.min_z + self.safety_margin, 
                       min(z, self.max_z - self.safety_margin))
        
        return clamped_x, clamped_y, clamped_z
    
    def get_distance_to_boundary(self, x, y, z):
        """Calculate minimum distance to any boundary"""
        distances = [
            x - (self.min_x + self.safety_margin),  # Distance to min_x
            (self.max_x - self.safety_margin) - x,  # Distance to max_x
            y - (self.min_y + self.safety_margin),  # Distance to min_y
            (self.max_y - self.safety_margin) - y,  # Distance to max_y
            z - (self.min_z + self.safety_margin),  # Distance to min_z
            (self.max_z - self.safety_margin) - z   # Distance to max_z
        ]
        
        return min(distances)
    
    def scale_velocity_near_boundary(self, velocity, current_pos):
        """Scale down velocity when approaching boundaries"""
        x, y, z = current_pos
        min_distance = self.get_distance_to_boundary(x, y, z)
        
        if min_distance < self.boundary_proximity_threshold:
            # Scale velocity based on proximity to boundary
            scale_factor = max(0.1, min_distance / self.boundary_proximity_threshold)
            scaled_velocity = [v * scale_factor for v in velocity]
            
            rospy.loginfo_throttle(2.0, f"Scaling velocity near boundary: factor={scale_factor:.2f}, distance={min_distance:.3f}")
            return scaled_velocity
        
        return velocity

    def enhanced_cartesian_callback(self, msg, topic_name):
        """Enhanced cartesian command processing with predictive safety"""
        if topic_name not in self.active_interceptors:
            return
            
        pub = self.active_interceptors[topic_name]['publisher']
        
        # Extract commanded position
        x, y, z = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        
        # Check if position is safe
        if self.is_position_safe(x, y, z):
            # Position is safe - forward command
            pub.publish(msg)
            self.last_safe_pose = msg
            
            # Reset violation counter
            if self.violation_count > 0:
                self.violation_count = 0
                rospy.loginfo("Robot back within safe bounds - violation counter reset")
                
            if self.emergency_stop_active:
                self.emergency_stop_active = False
                rospy.loginfo("Emergency stop deactivated - normal operation resumed")
                
        else:
            # Position is unsafe - apply safety measures
            self.violation_count += 1
            
            if self.violation_count >= self.violation_threshold:
                # Emergency stop and recovery
                self.trigger_emergency_recovery()
                return
            
            # Clamp to safe position
            clamped_x, clamped_y, clamped_z = self.clamp_position_to_bounds(x, y, z)
            
            # Create safe command
            safe_msg = PoseStamped()
            safe_msg.header = msg.header
            safe_msg.pose = msg.pose
            safe_msg.pose.position.x = clamped_x
            safe_msg.pose.position.y = clamped_y
            safe_msg.pose.position.z = clamped_z
            
            pub.publish(safe_msg)
            
            rospy.logwarn_throttle(1.0, 
                f"BOUNDARY VIOLATION #{self.violation_count}: Commanded ({x:.3f}, {y:.3f}, {z:.3f}) "
                f"clamped to ({clamped_x:.3f}, {clamped_y:.3f}, {clamped_z:.3f})")

    def enhanced_velocity_callback(self, msg, topic_name):
        """Enhanced velocity command processing with boundary prediction"""
        if topic_name not in self.active_interceptors:
            return
            
        pub = self.active_interceptors[topic_name]['publisher']
        
        try:
            # Get current end effector position
            transform = self.tf_buffer.lookup_transform("base_link", "end_effector_link", 
                                                      rospy.Time(0), rospy.Duration(0.1))
            current_pos = transform.transform.translation
            
            # Predict future position with longer lookahead
            lookahead_times = [0.1, 0.2, 0.5]  # Multiple time horizons
            
            safe_command = True
            for dt in lookahead_times:
                future_x = current_pos.x + msg.twist.linear.x * dt
                future_y = current_pos.y + msg.twist.linear.y * dt
                future_z = current_pos.z + msg.twist.linear.z * dt
                
                if not self.is_position_safe(future_x, future_y, future_z):
                    safe_command = False
                    break
            
            if safe_command:
                # Scale velocity if approaching boundaries
                scaled_velocity = self.scale_velocity_near_boundary(
                    [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z],
                    (current_pos.x, current_pos.y, current_pos.z)
                )
                
                # Create scaled message
                scaled_msg = TwistStamped()
                scaled_msg.header = msg.header
                scaled_msg.twist.linear.x = scaled_velocity[0]
                scaled_msg.twist.linear.y = scaled_velocity[1]
                scaled_msg.twist.linear.z = scaled_velocity[2]
                scaled_msg.twist.angular = msg.twist.angular  # Keep angular velocity
                
                pub.publish(scaled_msg)
                
            else:
                # Block unsafe velocity - publish zero velocity
                stop_msg = TwistStamped()
                stop_msg.header = msg.header
                # All velocities default to 0.0
                pub.publish(stop_msg)
                
                self.violation_count += 1
                if self.violation_count >= self.violation_threshold:
                    self.trigger_emergency_recovery()
                else:
                    rospy.logwarn_throttle(1.0, f"VELOCITY BLOCKED: Would violate boundaries (violation #{self.violation_count})")
                    
        except Exception as e:
            # Safety fallback - stop all motion
            stop_msg = TwistStamped()
            stop_msg.header = msg.header
            pub.publish(stop_msg)
            rospy.logwarn(f"Error in velocity processing - stopping motion: {e}")

    def trigger_emergency_recovery(self):
        """Enhanced emergency stop and recovery for Kinova Gen3"""
        if not self.emergency_stop_active:
            self.emergency_stop_active = True
            rospy.logerr("EMERGENCY STOP ACTIVATED - Multiple boundary violations detected!")
            
            # Publish emergency stop signal
            estop_msg = Bool()
            estop_msg.data = True
            self.estop_pub.publish(estop_msg)
            
            # For Kinova Gen3: Send zero velocity to stop immediately
            self.send_kinova_stop_command()
            
            # Cancel any active trajectories
            self.cancel_kinova_trajectories()
            
            # Initiate recovery to safe position
            rospy.Timer(rospy.Duration(1.0), self.initiate_recovery_motion, oneshot=True)
    
    def send_kinova_stop_command(self):
        """Send immediate stop command to Kinova Gen3"""
        try:
            stop_msg = TwistStamped()
            stop_msg.header.stamp = rospy.Time.now()
            stop_msg.header.frame_id = "base_link"
            # All velocities are 0.0 by default
            
            # Find the Kinova velocity publisher
            kinova_topic = '/my_gen3/in/cartesian_velocity'
            if kinova_topic in self.active_interceptors:
                pub = self.active_interceptors[kinova_topic]['publisher']
                # Send stop command multiple times to ensure it gets through
                for _ in range(10):
                    pub.publish(stop_msg)
                    rospy.sleep(0.01)
                rospy.loginfo("Sent emergency stop to Kinova Gen3")
            else:
                # Direct publish if interceptor not available
                stop_pub = rospy.Publisher(kinova_topic, TwistStamped, queue_size=1)
                rospy.sleep(0.1)  # Allow publisher to initialize
                for _ in range(10):
                    stop_pub.publish(stop_msg)
                    rospy.sleep(0.01)
                rospy.loginfo("Sent direct emergency stop to Kinova Gen3")
                
        except Exception as e:
            rospy.logerr(f"Failed to send Kinova stop command: {e}")
    
    def cancel_kinova_trajectories(self):
        """Cancel any active Kinova trajectory executions"""
        try:
            # Cancel trajectory action if it exists
            if hasattr(self, 'trajectory_client'):
                self.trajectory_client.cancel_all_goals()
                rospy.loginfo("Cancelled Kinova trajectory goals")
        except Exception as e:
            rospy.logerr(f"Failed to cancel Kinova trajectories: {e}")

    def initiate_recovery_motion(self):
        """Move robot to safe recovery position"""
        recovery_x, recovery_y, recovery_z = self.recovery_position
        
        rospy.loginfo(f"Initiating recovery motion to safe position: ({recovery_x:.3f}, {recovery_y:.3f}, {recovery_z:.3f})")
        
        # Create recovery pose
        recovery_pose = PoseStamped()
        recovery_pose.header.frame_id = "base_link"
        recovery_pose.header.stamp = rospy.Time.now()
        recovery_pose.pose.position.x = recovery_x
        recovery_pose.pose.position.y = recovery_y
        recovery_pose.pose.position.z = recovery_z
        recovery_pose.pose.orientation.w = 1.0  # Identity quaternion
        
        # Send recovery command to all active cartesian controllers
        for topic_name, interceptor in self.active_interceptors.items():
            if interceptor['type'] == 'cartesian':
                interceptor['publisher'].publish(recovery_pose)
                rospy.loginfo(f"Sent recovery command to {topic_name}")

    def setup_joint_level_safety(self):
        """Fallback joint-level safety when cartesian topics unavailable"""
        rospy.loginfo("Setting up joint-level safety monitoring")
        self.joint_states_sub = rospy.Subscriber('/joint_states', JointState, self.joint_states_callback)

    def joint_states_callback(self, msg):
        """Monitor joint states and calculate forward kinematics for safety"""
        # This would require forward kinematics implementation
        # For now, we'll rely on TF monitoring
        pass

    def check_boundary_proximity(self, event=None):
        """Check proximity to boundaries and adjust behavior"""
        try:
            transform = self.tf_buffer.lookup_transform("base_link", "end_effector_link", 
                                                      rospy.Time(0), rospy.Duration(0.1))
            pos = transform.transform.translation
            
            min_distance = self.get_distance_to_boundary(pos.x, pos.y, pos.z)
            
            if min_distance < self.boundary_proximity_threshold:
                rospy.loginfo_throttle(5.0, f"Approaching boundary: {min_distance:.3f}m remaining")
                
        except Exception as e:
            pass  # TF lookup failed, continue

    def check_robot_position_with_immediate_stop(self, event=None):
        """Enhanced robot position monitoring with immediate emergency stop"""
        violations = []
        
        try:
            # Check end effector position (most critical)
            transform = self.tf_buffer.lookup_transform("base_link", "end_effector_link", 
                                                      rospy.Time(0), rospy.Duration(0.05))
            pos = transform.transform.translation
            
            # Immediate check - if end effector is outside bounds, STOP NOW
            if not self.is_position_safe(pos.x, pos.y, pos.z, use_margin=False):
                # IMMEDIATE EMERGENCY STOP
                if not self.emergency_stop_active:
                    rospy.logerr(f"IMMEDIATE EMERGENCY STOP: End effector at ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}) outside bounds!")
                    self.emergency_stop_active = True
                    
                    # Send immediate stop commands
                    self.send_kinova_stop_command()
                    
                    # Trigger recovery after stopping
                    rospy.Timer(rospy.Duration(0.5), self.initiate_recovery_motion, oneshot=True)
                
                violations.append(("end_effector_link", pos.x, pos.y, pos.z, True))
            
            # Check other critical links
            critical_links = [
                "bracelet_link",
                "spherical_wrist_2_link", 
                "spherical_wrist_1_link",
                "forearm_link"
            ]
            
            for link_name in critical_links:
                try:
                    transform = self.tf_buffer.lookup_transform("base_link", link_name, 
                                                              rospy.Time(0), rospy.Duration(0.05))
                    pos = transform.transform.translation
                    
                    if not self.is_position_safe(pos.x, pos.y, pos.z, use_margin=False):
                        violations.append((link_name, pos.x, pos.y, pos.z, True))
                        
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                        tf2_ros.ExtrapolationException):
                    continue
            
            # Update violation status
            if violations:
                was_active = self.emergency_stop_active
                self.emergency_stop_active = True
                
                if not was_active:
                    for link_name, x, y, z, is_critical in violations:
                        rospy.logerr(f"CRITICAL BOUNDARY VIOLATION: {link_name} at ({x:.3f}, {y:.3f}, {z:.3f})")
                    
                    # Send multiple stop commands to ensure robot stops
                    for _ in range(5):
                        self.send_kinova_stop_command()
                        rospy.sleep(0.02)
                        
            else:
                # No violations - normal operation
                if self.emergency_stop_active:
                    self.emergency_stop_active = False
                    self.violation_count = 0
                    rospy.loginfo("All robot links within boundaries - emergency stop deactivated")
                    
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"Error in position monitoring: {e}")
            # Safety fallback - if we can't monitor, stop the robot
            if not self.emergency_stop_active:
                self.send_kinova_stop_command()

    def publish_boundary_markers(self):
        """Enhanced boundary visualization with status indication"""
        marker_array = MarkerArray()
        
        # Main boundary box
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "safety_bounds"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        # Position and size
        marker.pose.position.x = (self.min_x + self.max_x) / 2.0
        marker.pose.position.y = (self.min_y + self.max_y) / 2.0
        marker.pose.position.z = (self.min_z + self.max_z) / 2.0
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = self.max_x - self.min_x
        marker.scale.y = self.max_y - self.min_y
        marker.scale.z = self.max_z - self.min_z
        
        # Enhanced color coding based on system status
        if self.emergency_stop_active:
            # Red - Emergency stop active
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0
            marker.color.a = 0.6
        elif self.violation_count > 0:
            # Orange - Violations detected but not emergency
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.5, 0.0
            marker.color.a = 0.4
        else:
            # Green - Normal safe operation
            marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
            marker.color.a = 0.2
        
        marker_array.markers.append(marker)
        
        # Add wireframe edges with enhanced visibility
        self.add_enhanced_wireframe(marker_array)
        
        # Add safety margin visualization
        self.add_safety_margin_visualization(marker_array)
        
        # Publish the enhanced marker array
        self.marker_pub.publish(marker_array)

    def add_enhanced_wireframe(self, marker_array):
        """Enhanced wireframe with better visibility"""
        edges = self.get_box_edges()
        
        for i, (start, end) in enumerate(edges):
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "safety_bounds_edges"
            marker.id = i + 100
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            
            marker.points = [Point(*start), Point(*end)]
            marker.scale.x = 0.05  # Thicker lines
            
            # Match color with main box
            if self.emergency_stop_active:
                marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0
            elif self.violation_count > 0:
                marker.color.r, marker.color.g, marker.color.b = 1.0, 0.5, 0.0
            else:
                marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
            marker.color.a = 1.0
            
            marker.pose.orientation.w = 1.0
            marker_array.markers.append(marker)

    def add_safety_margin_visualization(self, marker_array):
        """Visualize the safety margin area"""
        margin_marker = Marker()
        margin_marker.header.frame_id = "base_link"
        margin_marker.header.stamp = rospy.Time.now()
        margin_marker.ns = "safety_margin"
        margin_marker.id = 200
        margin_marker.type = Marker.CUBE
        margin_marker.action = Marker.ADD
        
        # Inner box showing the actual safe area (with margin applied)
        margin_marker.pose.position.x = (self.min_x + self.max_x) / 2.0
        margin_marker.pose.position.y = (self.min_y + self.max_y) / 2.0
        margin_marker.pose.position.z = (self.min_z + self.max_z) / 2.0
        margin_marker.pose.orientation.w = 1.0
        
        margin_marker.scale.x = (self.max_x - self.min_x) - 2 * self.safety_margin
        margin_marker.scale.y = (self.max_y - self.min_y) - 2 * self.safety_margin
        margin_marker.scale.z = (self.max_z - self.min_z) - 2 * self.safety_margin
        
        # Semi-transparent blue for safety margin
        margin_marker.color.r, margin_marker.color.g, margin_marker.color.b = 0.0, 0.0, 1.0
        margin_marker.color.a = 0.1
        
        marker_array.markers.append(margin_marker)

    def get_box_edges(self):
        """Get all edges of the bounding box"""
        return [
            # Bottom face
            [(self.min_x, self.min_y, self.min_z), (self.max_x, self.min_y, self.min_z)],
            [(self.max_x, self.min_y, self.min_z), (self.max_x, self.max_y, self.min_z)],
            [(self.max_x, self.max_y, self.min_z), (self.min_x, self.max_y, self.min_z)],
            [(self.min_x, self.max_y, self.min_z), (self.min_x, self.min_y, self.min_z)],
            # Top face
            [(self.min_x, self.min_y, self.max_z), (self.max_x, self.min_y, self.max_z)],
            [(self.max_x, self.min_y, self.max_z), (self.max_x, self.max_y, self.max_z)],
            [(self.max_x, self.max_y, self.max_z), (self.min_x, self.max_y, self.max_z)],
            [(self.min_x, self.max_y, self.max_z), (self.min_x, self.min_y, self.max_z)],
            # Vertical edges
            [(self.min_x, self.min_y, self.min_z), (self.min_x, self.min_y, self.max_z)],
            [(self.max_x, self.min_y, self.min_z), (self.max_x, self.min_y, self.max_z)],
            [(self.max_x, self.max_y, self.min_z), (self.max_x, self.max_y, self.max_z)],
            [(self.min_x, self.max_y, self.min_z), (self.min_x, self.max_y, self.max_z)]
        ]

    def republish_markers(self, event=None):
        """Republish markers to ensure visibility"""
        self.publish_boundary_markers()

    def run(self):
        """Main run loop with enhanced status reporting"""
        rospy.loginfo("Enhanced Safety Monitor is running with ACTIVE BOUNDARY ENFORCEMENT")
        rospy.loginfo("=== OPERATION MODES ===")
        rospy.loginfo("GREEN: Normal operation - robot within bounds")
        rospy.loginfo("ORANGE: Boundary violations detected - commands being filtered") 
        rospy.loginfo("RED: Emergency stop active - robot being moved to recovery position")
        rospy.loginfo("=== COMMAND INTERCEPTION ===")
        
        for topic_name, info in self.active_interceptors.items():
            rospy.loginfo(f"  {info['type']}: {topic_name} (input: {topic_name}_original)")
            
        rospy.loginfo("=== Safety Parameters ===")
        rospy.loginfo(f"  Violation threshold: {self.violation_threshold}")
        rospy.loginfo(f"  Safety margin: {self.safety_margin}m")
        rospy.loginfo(f"  Boundary proximity threshold: {self.boundary_proximity_threshold}m")
        rospy.loginfo("Press Ctrl+C to stop.")
        
        rate = rospy.Rate(1)  # 1 Hz status updates
        while not rospy.is_shutdown():
            # Periodic status logging
            if self.emergency_stop_active:
                rospy.loginfo_throttle(10.0, "STATUS: EMERGENCY STOP ACTIVE - Recovery in progress")
            elif self.violation_count > 0:
                rospy.loginfo_throttle(5.0, f"STATUS: {self.violation_count} boundary violations - Filtering active")
            else:
                rospy.loginfo_throttle(30.0, "STATUS: Normal operation - Robot within safe bounds")
                
            rate.sleep()


if __name__ == '__main__':
    try:
        safety_monitor = EnhancedSafetyMonitor()
        safety_monitor.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Enhanced Safety Monitor shutdown requested")
    except KeyboardInterrupt:
        rospy.loginfo("Enhanced Safety Monitor stopped by user")
    except Exception as e:
        rospy.logerr(f"Enhanced Safety Monitor crashed: {e}")
        raise
