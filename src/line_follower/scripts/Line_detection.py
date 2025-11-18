#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class LineDetector:
    def __init__(self):
        rospy.init_node('line_detector', anonymous=True)
        self.bridge = CvBridge()
        
        # Subscribe to Kinova camera image topic
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', 
                                          Image, 
                                          self.image_callback,
                                          queue_size=1)
        rospy.loginfo("Line Detector Node Initialized. Waiting for images...")
    
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
        """
        Returns:
          skeleton: Skeletonized line (if found), otherwise None
          mask: The binary mask used for contour detection
          largest_contour: The largest detected contour (if found), otherwise None
        """
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

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", e)
            return
        
        # Detect line and retrieve largest contour
        skeleton, mask, largest_contour = self.detect_reference_line(cv_image)
        
        # Show the original image
        cv2.imshow("Original Image", cv_image)
        
        # Show the binary mask
        cv2.imshow("Binary Mask", mask)
        
        # Show the skeletonized line
        if skeleton is not None:
            cv2.imshow("Skeletonized Line", skeleton)
        else:
            cv2.imshow("Skeletonized Line", np.zeros_like(cv_image))
        
        # (NEW) Visualize the largest contour on the original image
        contour_image = cv_image.copy()
        if largest_contour is not None:
            cv2.drawContours(contour_image, [largest_contour], -1, (0, 255, 0), 2)
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

