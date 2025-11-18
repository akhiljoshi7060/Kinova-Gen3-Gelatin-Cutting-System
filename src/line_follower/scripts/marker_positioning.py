#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Import for RViz visualization and interactive marker support
from visualization_msgs.msg import Marker, InteractiveMarker, InteractiveMarkerControl
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from geometry_msgs.msg import Point, Quaternion

class LineDetector:
    def __init__(self):
        rospy.init_node('line_detector', anonymous=True)
        self.bridge = CvBridge()
        
        # Subscribe to camera image topic
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', 
                                          Image, 
                                          self.image_callback,
                                          queue_size=1)
        
        # Create an Interactive Marker Server for RViz visualization
        self.im_server = InteractiveMarkerServer("line_detector_im")
        
        # Create a fixed marker right away (no need to wait for an image)
        self.create_fixed_marker()
        
        rospy.loginfo("Line Detector Node Initialized. Waiting for images...")
    
    def create_fixed_marker(self):
        """Create a fixed marker directly in front of the robot arm"""
        # Create an InteractiveMarker
        im = InteractiveMarker()
        im.header.frame_id = "base_link"  # Use the robot's base frame
        im.header.stamp = rospy.Time.now()
        im.name = "fixed_line_marker"
        im.description = "Fixed Line Marker"
        
        # Position the marker directly in front of the robot
        # These values place the marker about 0.5m in front of the robot base
        im.pose.position.x = 0.5  # Forward
        im.pose.position.y = 0.0  # Centered
        im.pose.position.z = 0.3  # Slightly above the ground
        
        # Create a LINE_STRIP marker to visualize a circle
        marker = Marker()
        marker.type = Marker.LINE_STRIP
        marker.scale.x = 0.01  # Make it thicker for visibility
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        # Create a circle shape with points
        radius = 0.1  # 10cm radius
        num_points = 20
        for i in range(num_points + 1):
            angle = 2.0 * np.pi * i / num_points
            p = Point()
            p.x = radius * np.cos(angle)
            p.y = radius * np.sin(angle)
            p.z = 0
            marker.points.append(p)
        
        # Create a control for the marker
        control = InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(marker)
        im.controls.append(control)
        
        # Add control for moving the marker
        move_control = InteractiveMarkerControl()
        move_control.name = "move_plane"
        move_control.orientation.w = 1
        move_control.orientation.z = 1
        move_control.interaction_mode = InteractiveMarkerControl.MOVE_PLANE
        im.controls.append(move_control)
        
        # Insert the marker into the server
        self.im_server.insert(im, self.process_feedback)
        self.im_server.applyChanges()
        rospy.loginfo("Fixed marker created")
    
    def process_feedback(self, feedback):
        rospy.loginfo(f"Marker '{feedback.marker_name}' position: " +
                     f"x:{feedback.pose.position.x}, " +
                     f"y:{feedback.pose.position.y}, " +
                     f"z:{feedback.pose.position.z}")
    
    def detect_reference_line(self, image):
        """Simple line detection for visualization purposes"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        mask = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV,
            15, 4
        )
        
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
        
        if not valid_contours:
            return None, mask, None
        
        largest_contour = max(valid_contours, key=cv2.contourArea)
        return None, mask, largest_contour
    
    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", e)
            return
        
        # Detect line and retrieve largest contour
        _, mask, largest_contour = self.detect_reference_line(cv_image)
        
        # Visualize the largest contour on the original image
        contour_image = cv_image.copy()
        if largest_contour is not None:
            cv2.drawContours(contour_image, [largest_contour], -1, (0, 255, 0), 2)
        
        # Display images
        cv2.imshow("Original Image", cv_image)
        if mask is not None:
            cv2.imshow("Binary Mask", mask)
        if largest_contour is not None:
            cv2.imshow("Contour", contour_image)
        
        cv2.waitKey(1)

def main():
    ld = LineDetector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down Line Detector node.")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
