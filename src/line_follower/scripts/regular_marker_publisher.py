#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import tf2_ros
import tf.transformations as tft
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point

class LineDetector:
    def __init__(self):
        rospy.init_node('line_detector', anonymous=True)
        self.bridge = CvBridge()
        
        # Subscribe to the camera image topic.
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', 
                                          Image, 
                                          self.image_callback,
                                          queue_size=1)
        # Publisher for visualization markers (for RViz/MoveIt).
        self.marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
        
        # Set up tf2 listener to get transform from camera to robot base frame.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        rospy.loginfo("Line Detector Node Initialized. Waiting for images...")

        # Camera intrinsic parameters from default_color_calib_640x480.ini.
        self.fx = 656.58992
        self.fy = 657.52092
        self.cx = 313.35052
        self.cy = 281.68754
        
        # Assumed depth (meters) for the plane on which the line lies.
        # Adjust this value or use depth data for better accuracy.
        self.depth_assumption = 1.0

    def morphological_skeleton(self, bin_img):
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

    def detect_reference_line(self, image):
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
            rospy.logwarn("No valid contours found.")
            return None, mask, None
        
        largest_contour = max(valid_contours, key=cv2.contourArea)
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)
        
        skeleton = self.morphological_skeleton(contour_mask)
        return skeleton, mask, largest_contour

    def get_transform_matrix(self, transform):
        # Extract translation and rotation from the transform message.
        t = transform.transform.translation
        q = transform.transform.rotation
        trans = [t.x, t.y, t.z]
        quat = [q.x, q.y, q.z, q.w]
        T = tft.quaternion_matrix(quat)
        T[0:3, 3] = trans
        return T

    def image_to_base_frame(self, u, v, T_cam_to_base):
        # Back-project the pixel (u, v) into a 3D point in the camera coordinate system.
        x_cam = (u - self.cx) * self.depth_assumption / self.fx
        y_cam = (v - self.cy) * self.depth_assumption / self.fy
        z_cam = self.depth_assumption
        point_cam = np.array([x_cam, y_cam, z_cam, 1.0])
        # Transform the point to the robot's base frame.
        point_base = T_cam_to_base.dot(point_cam)
        return point_base[:3]

    def publish_line_marker(self, contour, header, T_cam_to_base):
        marker = Marker()
        marker.header = header
        # Set marker's frame to the robot base frame.
        marker.header.frame_id = "base_link"
        marker.ns = "detected_line"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.005  # line width in meters
        
        # Set marker color (green).
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        # Convert each pixel in the contour from image coordinates to base frame.
        for pt in contour:
            u = pt[0][0]
            v = pt[0][1]
            p_base = self.image_to_base_frame(u, v, T_cam_to_base)
            marker.points.append(Point(x=p_base[0], y=p_base[1], z=p_base[2]))
        
        self.marker_pub.publish(marker)

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", e)
            return
        
        # Look up the transform from the camera's frame to the robot's base frame.
        try:
            transform = self.tf_buffer.lookup_transform("base_link", data.header.frame_id, data.header.stamp, rospy.Duration(1.0))
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("Transform lookup failed: %s", e)
            return
        
        T_cam_to_base = self.get_transform_matrix(transform)
        
        skeleton, mask, largest_contour = self.detect_reference_line(cv_image)
        
        cv2.imshow("Original Image", cv_image)
        cv2.imshow("Binary Mask", mask)
        if skeleton is not None:
            cv2.imshow("Skeletonized Line", skeleton)
        else:
            cv2.imshow("Skeletonized Line", np.zeros_like(cv_image))
        
        contour_image = cv_image.copy()
        if largest_contour is not None:
            cv2.drawContours(contour_image, [largest_contour], -1, (0, 255, 0), 2)
            # Publish the detected line as a marker in the base frame.
            header = data.header  # re-use the image header (timestamp, etc.)
            self.publish_line_marker(largest_contour, header, T_cam_to_base)
        cv2.imshow("Largest Contour", contour_image)
        
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

