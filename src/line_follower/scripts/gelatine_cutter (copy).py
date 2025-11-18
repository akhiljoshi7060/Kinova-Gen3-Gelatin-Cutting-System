#!/usr/bin/env python3
import ctypes
# Initialize Xlib for multi-threaded use (if needed)
libX11 = ctypes.CDLL("libX11.so")
libX11.XInitThreads()

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import moveit_commander
import sys
from geometry_msgs.msg import Pose, Point, Vector3
import tf
import tf.transformations as transformations
import threading
import math
import actionlib
import control_msgs.msg
import copy
import moveit_msgs.msg
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

class CurvedLineFollower:
    def __init__(self):
        rospy.init_node('curved_line_follower', anonymous=True)
        
        # Initialize MoveIt
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander(
            robot_description="/my_gen3/robot_description", ns="/my_gen3")
        self.arm_group = moveit_commander.MoveGroupCommander(
            "arm", robot_description="/my_gen3/robot_description", ns="/my_gen3")
        
        # Setup planning parameters
        self.arm_group.set_max_velocity_scaling_factor(0.2)
        self.arm_group.set_max_acceleration_scaling_factor(0.2)
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Create a window for debugging
        cv2.namedWindow("Line Detection Debug", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Line Detection Debug", 800, 600)
        
        # Camera calibration parameters
        self.camera_matrix = np.array([
            [656.58992,    0.00000, 313.35052],
            [   0.00000,  657.52092, 281.68754],
            [   0.00000,    0.00000,   1.00000]
        ])
        
        # Initialize TF listener for transformations
        self.tf_listener = tf.TransformListener()
        
        # Detection parameters
        self.min_line_points = 15       # Minimum points to consider it a line
        self.required_stability = 7     # Stability count before confirming detection
        self.num_waypoints = 11         # Number of final waypoints (subsample)
        
        # State variables
        self.line_points = []          # Full 2D skeleton points
        self.waypoints = []            # Subsampled points + tangents
        self.line_detected = False
        self.detection_stable_count = 0
        self.detection_active = False   # Start off. Will enable after moving to viewing position
        
        # Z-height parameters for cutting
        self.default_working_height = 0.05  # Default working height above surface
        self.cutting_depth = 0.02          # Depth to cut into the surface
        
        # Debug information
        self.debug_info = {
            "detected_points": 0,
            "z_height": 0,
            "working_height": 0,
            "cutting_depth": self.cutting_depth
        }
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Publishers for visualization
        self.marker_pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=10)
        self.display_trajectory_publisher = rospy.Publisher(
            '/my_gen3/move_group/display_planned_path',
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20)
        
        # Subscribe to camera image
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image,
                                          self.image_callback, queue_size=1)
        
        rospy.loginfo("Curved Line Follower initialized (detection inactive until robot is positioned).")
    
    def move_to_defined_position(self):
        """
        Move the robot to a predefined viewing position to see the line.
        """
        rospy.loginfo("Moving to the defined viewing position...")
        # Example joint angles in degrees
        joint_angles = [355.174, 11.702, 184.05, 259.145, 359.699, 302.432, 87.176]
        
        # Convert to radians
        joint_positions = [self.kinova_to_radians(angle) for angle in joint_angles]
        
        # Set target and execute
        self.arm_group.set_joint_value_target(joint_positions)
        success = self.arm_group.go(wait=True)
        if success:
            rospy.loginfo("Successfully moved to the defined viewing position")
            # Get current Z-height for reference
            current_pose = self.arm_group.get_current_pose().pose
            self.debug_info["z_height"] = current_pose.position.z
            self.debug_info["working_height"] = current_pose.position.z - self.default_working_height
            # Enable detection now that we are in position
            self.detection_active = True
            return True
        else:
            rospy.logerr("Failed to move to the defined viewing position")
            return False

    def kinova_to_radians(self, deg):
        """Convert Kinova degrees to radians. Adjust if above pi."""
        rad = (deg % 360) * math.pi / 180.0
        if rad > math.pi:
            rad -= 2 * math.pi
        return rad

    def image_callback(self, data):
        """
        Main image callback:
         - If detection is active and line not yet stably detected, detect the line.
         - If stable, lock in the waypoints and publish the line to RViz.
        """
        if not self.detection_active or self.line_detected:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            with self.lock:
                # Detect the line
                line_points_2d, debug_image = self.detect_curved_line(cv_image)
                self.debug_info["detected_points"] = len(line_points_2d)

                # Check if we have enough points
                is_line_detected_now = len(line_points_2d) >= self.min_line_points
                
                if is_line_detected_now:
                    # Increase stability count
                    self.detection_stable_count += 1
                    self.line_points = line_points_2d  # Save the full set
                    # Create waypoints (subsample + tangents) from these points
                    self.waypoints = self.create_waypoints(self.line_points, self.num_waypoints)
                    
                    if self.detection_stable_count >= self.required_stability:
                        self.line_detected = True
                        self.detection_active = False
                        rospy.loginfo("Line stably detected. Waypoints locked.")
                        
                        # Now publish the raw line (the same line_points) to RViz
                        self.publish_raw_line_to_rviz(self.line_points)
                else:
                    self.detection_stable_count = max(0, self.detection_stable_count - 1)
                    if self.detection_stable_count == 0:
                        self.line_detected = False
                
                # Show debug window
                self.visualize_detection(cv_image, self.waypoints, debug_image)
        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")

    def detect_curved_line(self, image):
        """
        Simple approach: find largest contour + skeletonize.
        Returns a list of (x, y) pixel coords for skeleton points.
        """
        debug_image = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive threshold to isolate the line
        mask = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological cleanup
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
        
        if not valid_contours:
            return [], debug_image
        
        # Largest contour = line
        largest_contour = max(valid_contours, key=cv2.contourArea)
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)
        
        # Skeletonize
        skeleton = self.morphological_skeleton(contour_mask)
        
        # Overlay skeleton in red
        skeleton_bgr = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
        skeleton_bgr[np.where((skeleton_bgr == [255, 255, 255]).all(axis=2))] = [0, 0, 255]
        debug_image = cv2.addWeighted(debug_image, 1.0, skeleton_bgr, 1.0, 0)
        
        # Draw largest contour in green
        cv2.drawContours(debug_image, [largest_contour], -1, (0, 255, 0), 1)
        
        # Extract skeleton coords
        coords = np.column_stack(np.where(skeleton == 255))
        line_points = [(int(x), int(y)) for y, x in coords]
        
        # Debug text
        cv2.putText(debug_image, f"Line points: {len(line_points)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return line_points, debug_image

    def morphological_skeleton(self, bin_img):
        """Perform morphological skeletonization on a binary image."""
        bin_img = bin_img.copy()
        bin_img[bin_img > 0] = 255
        skel = np.zeros_like(bin_img, dtype=np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        while True:
            eroded = cv2.erode(bin_img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(bin_img, temp)
            skel = cv2.bitwise_or(skel, temp)
            bin_img = eroded.copy()
            if cv2.countNonZero(bin_img) == 0:
                break
        
        return skel

    def create_waypoints(self, line_points, num_waypoints=11):
        """
        Direct approach, no polynomial fitting:
         - Sort the skeleton points by descending y (optional).
         - Subsample to get ~num_waypoints points.
         - Compute tangents with finite differences.
        """
        if len(line_points) < self.min_line_points:
            rospy.logwarn("Not enough line points to create waypoints")
            return []
        
        # Optional: sort by descending y to impose an order from top to bottom
        points_array = np.array(line_points)
        sorted_indices = np.argsort(-points_array[:, 1])  # sort by -y
        sorted_points = points_array[sorted_indices]
        
        # Subsample
        step = max(1, len(sorted_points) // num_waypoints)
        sampled_points = sorted_points[::step]  # e.g., keep every step
        
        # Compute tangents
        waypoints_with_tangents = []
        for i, pt in enumerate(sampled_points):
            px, py = pt[0], pt[1]
            if i == 0 and len(sampled_points) > 1:
                # forward difference
                nx, ny = sampled_points[i+1]
                tx, ty = nx - px, ny - py
            elif i == len(sampled_points) - 1 and len(sampled_points) > 1:
                # backward difference
                px2, py2 = sampled_points[i-1]
                tx, ty = px - px2, py - py2
            elif len(sampled_points) > 2:
                # central difference
                pxp, pyp = sampled_points[i-1]
                pxn, pyn = sampled_points[i+1]
                tx, ty = pxn - pxp, pyn - pyp
            else:
                # single point or something degenerate
                tx, ty = 1, 0
            
            mag = math.sqrt(tx**2 + ty**2)
            if mag > 1e-9:
                tx /= mag
                ty /= mag
            
            waypoints_with_tangents.append((int(px), int(py), tx, ty))
            rospy.loginfo(f"Waypoint {i+1}: Image({int(px)}, {int(py)}) - Tangent({tx:.2f}, {ty:.2f})")
        
        return waypoints_with_tangents

    def visualize_detection(self, image, waypoints, debug_image):
        """Show the detected line + waypoints in OpenCV window."""
        vis_image = image.copy()
        
        # Draw waypoints as red circles + arrow tangents
        for i, wp in enumerate(waypoints):
            x, y, tx, ty = wp[0], wp[1], wp[2], wp[3]
            arrow_length = 50
            end_x = int(x + tx * arrow_length)
            end_y = int(y + ty * arrow_length)
            cv2.arrowedLine(vis_image, (x, y), (end_x, end_y), (255, 0, 0), 3)
            
            cv2.circle(vis_image, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(vis_image, f"{i+1}", (x+5, y+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 105, 180), 2)
        
        # Combine original + debug side-by-side
        h, w = image.shape[:2]
        combined = np.zeros((h, w*2, 3), dtype=np.uint8)
        combined[:, :w] = vis_image
        combined[:, w:] = debug_image
        
        # Add text
        cv2.putText(combined, f"Waypoints: {len(waypoints)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(combined, f"Z-height: {self.debug_info['z_height']:.4f}m", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(combined, f"Working height: {self.debug_info['working_height']:.4f}m", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(combined, f"Cutting depth: {self.debug_info['cutting_depth']:.4f}m", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.putText(combined, f"Line points: {self.debug_info['detected_points']}",
                    (w - 250, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.putText(combined, f"Stability: {self.detection_stable_count}/{self.required_stability}",
                    (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        if self.line_detected:
            text = "BLACK LINE: DETECTED"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = int((w - text_size[0]) / 2)
            text_y = h - 20
            cv2.rectangle(combined, (text_x - 10, text_y - 30), 
                          (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
            cv2.putText(combined, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        else:
            cv2.putText(combined, "NO LINE DETECTED", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        
        cv2.imshow("Line Detection Debug", combined)
        cv2.waitKey(1)

    def publish_raw_line_to_rviz(self, line_points):
        """
        Transform the same 2D line_points into 3D (base_link) and publish as a GREEN LINE.
        This ensures it matches exactly the data used for waypoints.
        """
        marker_array = MarkerArray()
        
        # We'll transform them to 3D
        robot_points = self.transform_2d_points_no_tangent(line_points)
        
        # Create a line strip marker
        line_marker = Marker()
        line_marker.header.frame_id = "base_link"
        line_marker.header.stamp = rospy.Time.now()
        line_marker.ns = "detected_line_raw"
        line_marker.id = 999
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.pose.orientation.w = 1.0
        
        line_marker.scale.x = 0.01   # thicker line
        line_marker.color = ColorRGBA(0.0, 1.0, 0.0, 1.0)  # green
        line_marker.lifetime = rospy.Duration(0)
        
        for pt in robot_points:
            p = Point()
            p.x = pt[0]
            p.y = pt[1]
            p.z = pt[2]
            line_marker.points.append(p)
        
        marker_array.markers.append(line_marker)
        self.marker_pub.publish(marker_array)
        rospy.loginfo("Published 'raw line' to RViz (green).")

    def transform_2d_points_no_tangent(self, image_points):
        """
        Convert 2D pixel coords into the same 3D space as we do for waypoints,
        but no orientation needed. We just produce a (x, y, z) triple for each point.
        """
        # Try the actual TF
        try:
            self.tf_listener.waitForTransform('/base_link', '/camera_color_frame',
                                              rospy.Time(0), rospy.Duration(2.0))
            (trans, rot) = self.tf_listener.lookupTransform('/base_link',
                                                            '/camera_color_frame',
                                                            rospy.Time(0))
            transform_matrix = tf.transformations.quaternion_matrix(rot)
            transform_matrix[0:3, 3] = trans
            
            # We'll guess the same distance. If you have a real camera TF, adapt accordingly.
            estimated_depth = 0.45  # same as we use in transform_image_to_robot_frame
            result_3d = []
            for (px, py) in image_points:
                x_norm = (px - self.camera_matrix[0, 2]) / self.camera_matrix[0, 0]
                y_norm = (py - self.camera_matrix[1, 2]) / self.camera_matrix[1, 1]
                cam_x = x_norm * estimated_depth
                cam_y = y_norm * estimated_depth
                cam_z = estimated_depth
                
                base_pt = transform_matrix.dot(np.array([cam_x, cam_y, cam_z, 1.0]))
                result_3d.append((base_pt[0], base_pt[1], base_pt[2]))
            return result_3d
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("No camera->base TF. Using fallback approximate transform.")
            
            # fallback approximate approach
            result_3d = []
            img_cx = self.camera_matrix[0, 2]
            img_cy = self.camera_matrix[1, 2]
            distance_estimate = 0.45
            fov_factor = 0.6
            width_in_px = 640
            mm_per_pixel = (2 * distance_estimate * math.tan(math.radians(fov_factor * 30))) / width_in_px
            x_scale = mm_per_pixel / 1000
            y_scale = mm_per_pixel / 1000
            
            # We'll assume the robot base is at (0,0). Adjust as needed.
            for (px, py) in image_points:
                x_off = (px - img_cx) * x_scale
                y_off = (py - img_cy) * y_scale
                # We'll store them as if they're in front of the robot at some offset
                # This is purely a guess for visualization
                # e.g. x => -y_off, y => -x_off
                # z => 0 for fallback or a small fixed distance
                base_x = -y_off
                base_y = -x_off
                base_z = 0.0
                result_3d.append((base_x, base_y, base_z))
            return result_3d

    def publish_waypoints_to_rviz(self, robot_poses):
        """Publish the final 'waypoints' path in RViz (red line + spheres + arrows)."""
        marker_array = MarkerArray()
        
        # Connect waypoints with a red line
        line_marker = Marker()
        line_marker.header.frame_id = "base_link"
        line_marker.header.stamp = rospy.Time.now()
        line_marker.ns = "waypoints"
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.pose.orientation.w = 1.0
        line_marker.scale.x = 0.01
        line_marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)  # Red
        line_marker.lifetime = rospy.Duration(0)
        
        for i, pose in enumerate(robot_poses):
            p = Point()
            p.x = pose.position.x
            p.y = pose.position.y
            p.z = pose.position.z
            line_marker.points.append(p)
            
            # Blue sphere
            sphere = Marker()
            sphere.header.frame_id = "base_link"
            sphere.header.stamp = rospy.Time.now()
            sphere.ns = "waypoints"
            sphere.id = i + 1
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose = pose
            sphere.scale = Vector3(0.02, 0.02, 0.02)
            sphere.color = ColorRGBA(0.0, 0.0, 1.0, 1.0)  # Blue
            sphere.lifetime = rospy.Duration(0)
            
            # Green arrow for orientation
            arrow = Marker()
            arrow.header.frame_id = "base_link"
            arrow.header.stamp = rospy.Time.now()
            arrow.ns = "waypoints"
            arrow.id = i + 100
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.pose = pose
            arrow.scale = Vector3(0.07, 0.01, 0.01)
            arrow.color = ColorRGBA(0.0, 1.0, 0.0, 1.0)
            arrow.lifetime = rospy.Duration(0)
            
            marker_array.markers.append(sphere)
            marker_array.markers.append(arrow)
        
        marker_array.markers.append(line_marker)
        self.marker_pub.publish(marker_array)
        rospy.loginfo("Published waypoints to RViz (marker ns=waypoints).")

    def transform_image_to_robot_frame(self, image_points_with_tangents):
        """
        Convert 2D pixel coords + tangents => 3D robot coords with orientation.
        Returns a list of Pose() for each waypoint.
        """
        robot_poses = []
        
        try:
            self.tf_listener.waitForTransform('/base_link', '/camera_color_frame',
                                              rospy.Time(0), rospy.Duration(2.0))
            (trans, rot) = self.tf_listener.lookupTransform('/base_link',
                                                            '/camera_color_frame',
                                                            rospy.Time(0))
            current_pose = self.arm_group.get_current_pose().pose
            self.debug_info["z_height"] = current_pose.position.z
            working_height = current_pose.position.z - self.default_working_height
            self.debug_info["working_height"] = working_height
            
            transform_matrix = tf.transformations.quaternion_matrix(rot)
            transform_matrix[0:3, 3] = trans
            
            estimated_depth = 0.45
            for i, wp in enumerate(image_points_with_tangents):
                px, py, tx, ty = wp[0], wp[1], wp[2], wp[3]
                x_norm = (px - self.camera_matrix[0, 2]) / self.camera_matrix[0, 0]
                y_norm = (py - self.camera_matrix[1, 2]) / self.camera_matrix[1, 1]
                
                camera_x = x_norm * estimated_depth
                camera_y = y_norm * estimated_depth
                camera_z = estimated_depth
                
                # Transform to base
                base_pt = transform_matrix.dot(np.array([camera_x, camera_y, camera_z, 1.0]))
                
                # Tangent point
                cam_tangent_pt = np.array([
                    camera_x + tx * 0.1,
                    camera_y + ty * 0.1,
                    camera_z,
                    1.0
                ])
                base_tangent_pt = transform_matrix.dot(cam_tangent_pt)
                
                robot_tangent_x = base_tangent_pt[0] - base_pt[0]
                robot_tangent_y = base_tangent_pt[1] - base_pt[1]
                mag = math.sqrt(robot_tangent_x**2 + robot_tangent_y**2)
                if mag > 1e-9:
                    robot_tangent_x /= mag
                    robot_tangent_y /= mag
                
                # Orientation
                new_orientation = self.calculate_orientation_from_tangent(
                    robot_tangent_x, robot_tangent_y, current_pose.orientation
                )
                
                # Build Pose
                pose_msg = Pose()
                pose_msg.position.x = base_pt[0]
                pose_msg.position.y = base_pt[1]
                pose_msg.position.z = working_height
                pose_msg.orientation.x = new_orientation[0]
                pose_msg.orientation.y = new_orientation[1]
                pose_msg.orientation.z = new_orientation[2]
                pose_msg.orientation.w = new_orientation[3]
                
                robot_poses.append(pose_msg)
                rospy.loginfo(f"Waypoint {i+1}: Image({px},{py}) -> Robot({pose_msg.position.x:.3f},"
                              f"{pose_msg.position.y:.3f},{pose_msg.position.z:.3f}), "
                              f"Orientation=({new_orientation[0]:.2f}, {new_orientation[1]:.2f}, "
                              f"{new_orientation[2]:.2f}, {new_orientation[3]:.2f})")
        
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("No camera->base TF. Using fallback for transform_image_to_robot_frame.")
            
            # fallback approximate approach
            robot_poses = []
            current_pose = self.arm_group.get_current_pose().pose
            working_height = current_pose.position.z - self.default_working_height
            self.debug_info["working_height"] = working_height
            self.debug_info["z_height"] = current_pose.position.z
            
            img_cx = self.camera_matrix[0, 2]
            img_cy = self.camera_matrix[1, 2]
            distance_estimate = 0.45
            fov_factor = 0.6
            width_in_px = 640
            mm_per_pixel = (2 * distance_estimate * math.tan(math.radians(fov_factor * 30))) / width_in_px
            x_scale = mm_per_pixel / 1000
            y_scale = mm_per_pixel / 1000
            
            for i, wp in enumerate(image_points_with_tangents):
                px, py, tx, ty = wp[0], wp[1], wp[2], wp[3]
                
                x_off = (px - img_cx) * x_scale
                y_off = (py - img_cy) * y_scale
                
                # Build Pose
                pose_msg = Pose()
                # Example mapping
                pose_msg.position.x = current_pose.position.x - y_off
                pose_msg.position.y = current_pose.position.y - x_off
                pose_msg.position.z = working_height
                
                # Orientation from tangent
                robot_tx = -ty
                robot_ty = -tx
                new_orientation = self.calculate_orientation_from_tangent(
                    robot_tx, robot_ty, current_pose.orientation
                )
                pose_msg.orientation.x = new_orientation[0]
                pose_msg.orientation.y = new_orientation[1]
                pose_msg.orientation.z = new_orientation[2]
                pose_msg.orientation.w = new_orientation[3]
                
                robot_poses.append(pose_msg)
                rospy.loginfo(
                    f"Fallback WP {i+1}: Image({px},{py}) -> Robot({pose_msg.position.x:.3f},"
                    f"{pose_msg.position.y:.3f},{pose_msg.position.z:.3f}), "
                    f"Orientation=({new_orientation[0]:.2f}, {new_orientation[1]:.2f}, "
                    f"{new_orientation[2]:.2f}, {new_orientation[3]:.2f})"
                )
        return robot_poses

    def calculate_orientation_from_tangent(self, tangent_x, tangent_y, current_orientation):
        """
        Align the end-effector +X axis with the line's tangent.
        If your physical tool is oriented differently, add an offset (e.g. angle - math.pi/2).
        """
        current_euler = tf.transformations.euler_from_quaternion([
            current_orientation.x,
            current_orientation.y,
            current_orientation.z,
            current_orientation.w
        ])
        angle = math.atan2(tangent_y, tangent_x)
        
        # Keep current roll & pitch, set yaw to 'angle'
        new_euler = (current_euler[0], current_euler[1], angle)
        new_quaternion = tf.transformations.quaternion_from_euler(
            new_euler[0], new_euler[1], new_euler[2]
        )
        return new_quaternion
    
    def control_gripper(self, position, force=0.5):
        """
        Control Robotiq gripper: position in [0.0..1.0], force in [0..1.0].
        """
        try:
            gripper_client = actionlib.SimpleActionClient(
                '/my_gen3/robotiq_2f_140_gripper_controller/gripper_cmd',
                control_msgs.msg.GripperCommandAction
            )
            if gripper_client.wait_for_server(rospy.Duration(5.0)):
                goal = control_msgs.msg.GripperCommandGoal()
                goal.command.position = position
                goal.command.max_effort = force * 100.0
                
                gripper_client.send_goal(goal)
                gripper_client.wait_for_result(rospy.Duration(5.0))
                
                result = gripper_client.get_result()
                if result and hasattr(result, 'reached_goal') and result.reached_goal:
                    rospy.loginfo(f"Gripper moved to position {position}")
                    return True
                else:
                    rospy.logwarn(f"Gripper did not reach position {position}")
                    return False
            else:
                rospy.logerr("Failed to connect to gripper action server")
                return False
        except Exception as e:
            rospy.logerr(f"Error controlling gripper: {str(e)}")
            return False
    
    def perform_cutting_action(self):
        """
        Move down by cutting_depth, close and open the gripper, then go back up.
        """
        rospy.loginfo("Performing cutting action")
        
        current_pose = self.arm_group.get_current_pose().pose
        cutting_pose = copy.deepcopy(current_pose)
        cutting_pose.position.z -= self.cutting_depth
        
        self.debug_info["z_height"] = cutting_pose.position.z
        
        self.arm_group.set_pose_target(cutting_pose)
        success = self.arm_group.go(wait=True)
        
        if success:
            rospy.loginfo("Contact made with surface for cutting")
            self.control_gripper(0.0)  # close
            rospy.sleep(1.0)
            self.control_gripper(1.0)  # open
            
            # Return up
            up_pose = copy.deepcopy(cutting_pose)
            up_pose.position.z += self.cutting_depth + 0.01
            self.arm_group.set_pose_target(up_pose)
            self.arm_group.go(wait=True)
            
            self.debug_info["z_height"] = up_pose.position.z
            rospy.loginfo("Cutting action completed")
        else:
            rospy.logerr("Failed to make contact for cutting")
    
    def follow_line(self):
        """
        Follow the line using the final waypoints. We do a Cartesian path if possible.
        """
        with self.lock:
            if not self.waypoints or len(self.waypoints) < 3:
                rospy.logwarn("Not enough waypoints to follow")
                return
            waypoints_copy = self.waypoints.copy()
        
        rospy.loginfo(f"Following line with {len(waypoints_copy)} waypoints...")
        robot_poses = self.transform_image_to_robot_frame(waypoints_copy)
        
        if len(robot_poses) < 3:
            rospy.logerr("Failed to transform waypoints to robot coordinates")
            return
        
        # Publish final path to RViz
        self.publish_waypoints_to_rviz(robot_poses)
        
        key = input("Waypoints visible in RViz. Press Enter to execute motion, or 'q' to quit: ")
        if key.lower() == 'q':
            rospy.loginfo("Motion execution canceled by user.")
            return
        
        # Slow speed for precise motion
        original_velocity = 0.2
        original_accel = 0.2
        self.arm_group.set_max_velocity_scaling_factor(0.05)
        self.arm_group.set_max_acceleration_scaling_factor(0.05)
        
        try:
            (plan, fraction) = self.arm_group.compute_cartesian_path(
                robot_poses,
                0.01,  # eef_step
                0.0,   # jump_threshold
                True   # avoid_collisions
            )
            
            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.trajectory_start = self.robot.get_current_state()
            display_trajectory.trajectory.append(plan)
            self.display_trajectory_publisher.publish(display_trajectory)
            
            rospy.loginfo("Trajectory displayed in RViz. Executing in 3 seconds...")
            rospy.sleep(3)
            
            if fraction > 0.7:
                rospy.loginfo(f"Executing Cartesian path (coverage: {fraction*100:.1f}%)")
                self.arm_group.execute(plan, wait=True)
                
                # Perform cutting
                self.perform_cutting_action()
            else:
                rospy.logwarn(f"Low Cartesian coverage ({fraction*100:.1f}%). Fallback to point-to-point.")
                self._execute_point_to_point(robot_poses)
                if len(robot_poses) > 0:
                    self.perform_cutting_action()
        
        except Exception as e:
            rospy.logerr(f"Error in cartesian planning: {str(e)}")
            rospy.logwarn("Falling back to point-to-point motion.")
            self._execute_point_to_point(robot_poses)
            if len(robot_poses) > 0:
                try:
                    self.perform_cutting_action()
                except Exception as cut_err:
                    rospy.logerr(f"Error during cutting action: {str(cut_err)}")
        
        # Restore speed
        self.arm_group.set_max_velocity_scaling_factor(original_velocity)
        self.arm_group.set_max_acceleration_scaling_factor(original_accel)
        
        rospy.loginfo("Line following completed")
        self.detection_stable_count = 0

    def _execute_point_to_point(self, poses):
        """Fallback method: move to each pose in sequence."""
        for i, pose in enumerate(poses):
            try:
                rospy.loginfo(f"Moving to waypoint {i+1}")
                self.arm_group.set_pose_target(pose)
                success = self.arm_group.go(wait=True)
                if success:
                    rospy.loginfo(f"Reached waypoint {i+1}")
                    current_pose = self.arm_group.get_current_pose().pose
                    self.debug_info["z_height"] = current_pose.position.z
                else:
                    rospy.logwarn(f"Failed to reach waypoint {i+1}")
                rospy.sleep(0.7)
            except Exception as e:
                rospy.logerr(f"Error moving to waypoint {i+1}: {str(e)}")

def main():
    follower = CurvedLineFollower()
    
    # Move to defined view
    if not follower.move_to_defined_position():
        rospy.logerr("Could not move to defined position. Exiting.")
        return
    
    # Wait up to 30s for stable detection
    rospy.loginfo("Detection active. Waiting for stable line detection (max 30s)...")
    timeout = rospy.Time.now() + rospy.Duration(30)
    while not rospy.is_shutdown() and not follower.line_detected:
        if rospy.Time.now() > timeout:
            rospy.logwarn("Line not detected within 30s. Exiting.")
            break
        rospy.sleep(0.1)
    
    if follower.line_detected:
        rospy.loginfo("Stable line detected. Pausing 5 seconds for debug view...")
        rospy.sleep(5)
        rospy.loginfo("Now executing slow trajectory along final waypoints...")
        
        key = input("Press Enter to start following the line, or 'q' to quit: ")
        if key.lower() != 'q':
            follower.follow_line()
            rospy.loginfo("Line following completed.")
        else:
            rospy.loginfo("Line following canceled by user.")
    else:
        rospy.logwarn("No stable line was detected. Exiting.")
    
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

