#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import moveit_commander
import sys
from geometry_msgs.msg import Pose
import tf
import tf.transformations as transformations
import os

class CoordinateCalibrator:
    def __init__(self):
        rospy.init_node('coordinate_calibrator', anonymous=True)
        
        # Force GUI to display
        os.environ['DISPLAY'] = ':0'
        
        # Initialize MoveIt
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander(
            robot_description="/my_gen3/robot_description", 
            ns="/my_gen3"
        )
        self.arm_group = moveit_commander.MoveGroupCommander(
            "arm", 
            robot_description="/my_gen3/robot_description", 
            ns="/my_gen3"
        )
        
        # Setup planning parameters
        self.arm_group.set_max_velocity_scaling_factor(0.1)
        self.arm_group.set_max_acceleration_scaling_factor(0.1)
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Create debug windows
        cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera View", 800, 600)
        cv2.moveWindow("Camera View", 0, 0)
        
        # TF listener
        self.tf_listener = tf.TransformListener()
        
        # Camera calibration
        self.camera_matrix = np.array([
            [656.58992, 0.00000, 313.35052],
            [0.00000, 657.52092, 281.68754],
            [0.00000, 0.00000, 1.00000]
        ])
        self.distortion_coeffs = np.array([0.01674, -0.07567, 0.02052, -0.00511, 0.00000])
        
        # Parameters
        self.estimated_camera_distance = 0.38  # 15 inches
        self.calibration_offset_x = 0.0
        self.calibration_offset_y = 0.0
        self.scale_factor = 1.0  # Additional scaling factor for fine-tuning
        
        # Camera-to-robot transform
        self.camera_to_robot_transform = None
        
        # Mouse callback state
        self.click_position = None
        
        # Subscribe to camera topic
        self.camera_topic = '/camera/color/image_raw'
        rospy.loginfo(f"Subscribing to camera topic: {self.camera_topic}")
        self.image_sub = rospy.Subscriber(self.camera_topic, Image, self.image_callback, queue_size=1)
        
        # Create trackbars for calibration parameters
        cv2.createTrackbar("Est. Distance (cm)", "Camera View", 38, 100, self.update_distance)
        cv2.createTrackbar("Offset X (mm)", "Camera View", 0, 100, self.update_offset_x)
        cv2.createTrackbar("Offset Y (mm)", "Camera View", 0, 100, self.update_offset_y)
        cv2.createTrackbar("Scale (%)", "Camera View", 100, 200, self.update_scale)
        
        # Mouse callback
        cv2.setMouseCallback("Camera View", self.mouse_callback)
        
        rospy.loginfo("Coordinate Calibrator initialized")
        
        # Move to viewing position
        self.move_to_viewing_position()
    
    def update_distance(self, value):
        self.estimated_camera_distance = value / 100.0  # Convert cm to meters
    
    def update_offset_x(self, value):
        self.calibration_offset_x = (value - 50) / 1000.0  # Convert to meters, centered at 0
    
    def update_offset_y(self, value):
        self.calibration_offset_y = (value - 50) / 1000.0  # Convert to meters, centered at 0
    
    def update_scale(self, value):
        self.scale_factor = value / 100.0  # Convert percentage to factor
    
    def mouse_callback(self, event, x, y, flags, param):
        # Handle mouse click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_position = (x, y)
            rospy.loginfo(f"Clicked at image position: ({x}, {y})")
            
            # Transform to robot coordinates
            robot_pose = self.transform_point_to_robot(x, y)
            
            if robot_pose:
                rospy.loginfo(f"Transformed to robot pose: ("
                            f"{robot_pose.position.x:.4f}, "
                            f"{robot_pose.position.y:.4f}, "
                            f"{robot_pose.position.z:.4f})")
                
                # Ask if user wants to move robot to this position
                rospy.loginfo("Press 'm' to move robot to this position, or any other key to cancel")
                self.pending_move_pose = robot_pose
            else:
                rospy.logwarn("Failed to transform point to robot coordinates")
    
    def move_to_viewing_position(self):
        rospy.loginfo("Moving to viewing position...")
        joint_angles = [355, 3.4, 183, 267, 359, 298, 89]
        joint_positions = [self.kinova_to_radians(a) for a in joint_angles]
        
        self.arm_group.set_joint_value_target(joint_positions)
        success = self.arm_group.go(wait=True)
        if success:
            rospy.loginfo("Successfully moved to viewing position")
            
            # Get camera transform
            try:
                self.tf_listener.waitForTransform(
                    '/base_link', 
                    '/camera_color_frame', 
                    rospy.Time(0), 
                    rospy.Duration(2.0)
                )
                (trans, rot) = self.tf_listener.lookupTransform(
                    '/base_link', 
                    '/camera_color_frame', 
                    rospy.Time(0)
                )
                self.camera_to_robot_transform = (trans, rot)
                rospy.loginfo(f"Camera transform: trans={trans}, rot={rot}")
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logwarn(f"Could not get camera transform: {e}")
        else:
            rospy.logerr("Failed to move to viewing position")
    
    def kinova_to_radians(self, deg):
        deg = deg % 360
        if deg > 180:
            deg -= 360
        return deg * np.pi / 180.0
    
    def transform_point_to_robot(self, x, y):
        """Transform a single image point to robot coordinates"""
        if not self.camera_to_robot_transform:
            rospy.logwarn("No camera transform available")
            return None
        
        try:
            current_pose = self.arm_group.get_current_pose().pose
            current_height = current_pose.position.z
            
            trans, rot = self.camera_to_robot_transform
            
            # Camera intrinsics
            cx = self.camera_matrix[0, 2]
            cy = self.camera_matrix[1, 2]
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            
            # Handle distortion if needed
            if np.any(self.distortion_coeffs):
                pt_np = np.array([[x, y]], dtype=np.float32)
                undistorted = cv2.undistortPoints(
                    pt_np, self.camera_matrix, self.distortion_coeffs,
                    None, self.camera_matrix
                )
                undist_pt = (undistorted[0][0][0], undistorted[0][0][1])
                x_norm = (undist_pt[0] - cx) / fx
                y_norm = (undist_pt[1] - cy) / fy
            else:
                x_norm = (x - cx) / fx
                y_norm = (y - cy) / fy
            
            # Apply scale factor adjustment
            x_norm *= self.scale_factor
            y_norm *= self.scale_factor
            
            # Project to 3D
            depth = self.estimated_camera_distance
            point_camera = np.array([x_norm * depth, y_norm * depth, depth, 1.0])
            
            # Transform to robot frame
            rot_matrix = transformations.quaternion_matrix(rot)
            transform_matrix = rot_matrix.copy()
            transform_matrix[0:3, 3] = trans
            point_base = np.dot(transform_matrix, point_camera)
            
            # Create pose
            target_pose = Pose()
            target_pose.position.x = point_base[0] + self.calibration_offset_x
            target_pose.position.y = point_base[1] + self.calibration_offset_y
            target_pose.position.z = current_height  # Maintain current height
            target_pose.orientation = current_pose.orientation
            
            return target_pose
            
        except Exception as e:
            rospy.logerr(f"Error transforming point: {e}")
            return None
    
    def image_callback(self, data):
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
            # Create a copy for visualization
            vis_image = cv_image.copy()
            
            # Get current robot pose
            current_pose = self.arm_group.get_current_pose().pose
            
            # Add robot pose information
            h, w = vis_image.shape[:2]
            cv2.putText(vis_image, f"Robot Position:", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_image, f"X: {current_pose.position.x:.4f}", (10, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_image, f"Y: {current_pose.position.y:.4f}", (10, 90),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_image, f"Z: {current_pose.position.z:.4f}", (10, 120),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add calibration parameters
            cv2.putText(vis_image, f"Depth: {self.estimated_camera_distance:.2f}m", (w-250, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_image, f"Offset X: {self.calibration_offset_x*1000:.1f}mm", (w-250, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_image, f"Offset Y: {self.calibration_offset_y*1000:.1f}mm", (w-250, 90),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_image, f"Scale: {self.scale_factor:.2f}", (w-250, 120),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw crosshair at center
            center_x = w // 2
            center_y = h // 2
            cv2.line(vis_image, (center_x-20, center_y), (center_x+20, center_y), (0, 0, 255), 2)
            cv2.line(vis_image, (center_x, center_y-20), (center_x, center_y+20), (0, 0, 255), 2)
            cv2.putText(vis_image, "CENTER", (center_x+10, center_y-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Draw grid overlay (every 50 pixels)
            grid_spacing = 50
            grid_color = (0, 255, 255)  # Yellow
            
            for x in range(0, w, grid_spacing):
                cv2.line(vis_image, (x, 0), (x, h), grid_color, 1)
            
            for y in range(0, h, grid_spacing):
                cv2.line(vis_image, (0, y), (w, y), grid_color, 1)
            
            # Draw the last clicked position
            if self.click_position:
                x, y = self.click_position
                cv2.circle(vis_image, (x, y), 10, (0, 255, 0), 2)
                cv2.line(vis_image, (x-15, y), (x+15, y), (0, 255, 0), 2)
                cv2.line(vis_image, (x, y-15), (x, y+15), (0, 255, 0), 2)
                cv2.putText(vis_image, f"({x}, {y})", (x+20, y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Draw vector from center to click
                cv2.line(vis_image, (center_x, center_y), (x, y), (255, 0, 255), 2)
            
            # Add instructions
            cv2.putText(vis_image, "Click anywhere to transform coordinates", (10, h-60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_image, "Adjust trackbars to calibrate transformation", (10, h-30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display the image
            cv2.imshow("Camera View", vis_image)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('m') and hasattr(self, 'pending_move_pose'):
                # Move robot to the transformed position
                rospy.loginfo("Moving robot to clicked position...")
                self.arm_group.set_pose_target(self.pending_move_pose)
                success = self.arm_group.go(wait=True)
                
                if success:
                    rospy.loginfo("Successfully moved to target position")
                else:
                    rospy.logwarn("Failed to move to target position")
                
                # Clear pending move
                delattr(self, 'pending_move_pose')
            
            elif key == ord('r'):
                # Reset to viewing position
                self.move_to_viewing_position()
            
            elif key == ord('s'):
                # Save current calibration
                params = {
                    'estimated_camera_distance': self.estimated_camera_distance,
                    'calibration_offset_x': self.calibration_offset_x,
                    'calibration_offset_y': self.calibration_offset_y,
                    'scale_factor': self.scale_factor
                }
                
                # Save to file
                with open('calibration_params.txt', 'w') as f:
                    for key, value in params.items():
                        f.write(f"{key}: {value}\n")
                
                rospy.loginfo("Saved calibration parameters to calibration_params.txt")
                
                # Save current frame
                cv2.imwrite('calibration_frame.jpg', vis_image)
                rospy.loginfo("Saved calibration frame to calibration_frame.jpg")
            
        except Exception as e:
            rospy.logerr(f"Error in image callback: {e}")


def main():
    calibrator = CoordinateCalibrator()
    
    rospy.loginfo("Coordinate Calibrator running")
    rospy.loginfo("Click on image to transform to robot coordinates")
    rospy.loginfo("Press 'm' to move robot to clicked position")
    rospy.loginfo("Press 'r' to reset to viewing position")
    rospy.loginfo("Press 's' to save calibration parameters")
    rospy.loginfo("Press Ctrl+C to exit")
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
