#!/usr/bin/env python3
import ctypes
# Initialize Xlib for multi-threaded use (if needed)
try:
    libX11 = ctypes.CDLL("libX11.so")
    libX11.XInitThreads()
except Exception:
    pass  # headless or non-X11 system

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import moveit_commander
import sys
from geometry_msgs.msg import Pose, Point, Vector3
import tf
import tf.transformations as transformations  # noqa: F401 (import side effects)
import threading
import math
import actionlib
import control_msgs.msg
import copy
import moveit_msgs.msg
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from collections import deque

# Optional smoothing: try SciPy; fallback to simple box filter
try:
    from scipy.signal import savgol_filter
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# ArUco (OpenCV contrib)
try:
    import cv2.aruco as aruco
    _HAS_ARUCO = True
except Exception:
    _HAS_ARUCO = False


class CurvedLineFollower:
    def __init__(self):
        rospy.init_node('curved_line_follower', anonymous=True)

        # Initialize MoveIt
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander(
            robot_description="/my_gen3/robot_description", ns="/my_gen3")
        self.arm_group = moveit_commander.MoveGroupCommander(
            "arm", robot_description="/my_gen3/robot_description", ns="/my_gen3")

        # Setup planning parameters (default; we’ll slow further during execution)
        self.arm_group.set_max_velocity_scaling_factor(0.2)
        self.arm_group.set_max_acceleration_scaling_factor(0.2)

        # Initialize CV Bridge
        self.bridge = CvBridge()

        # Debug window
        try:
            cv2.namedWindow("Line Detection Debug", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Line Detection Debug", 800, 600)
        except Exception:
            pass  # headless

        # Camera intrinsics (update to your camera if needed)
        self.camera_matrix = np.array([
            [656.58992,    0.00000, 313.35052],
            [   0.00000,  657.52092, 281.68754],
            [   0.00000,    0.00000,   1.00000]
        ])

        # Initialize TF listener
        self.tf_listener = tf.TransformListener()

        # Detection parameters / state
        self.min_line_points = 15
        self.required_stability = 7
        self.num_waypoints = 11  # kept but unused by arc-length sampler
        self.line_points = []
        self.waypoints = []
        self.line_detected = False
        self.detection_stable_count = 0
        self.detection_active = False

        # Z-height parameters for cutting
        self.default_working_height = 0.05  # meters above current pose.z
        self.cutting_depth = 0.02

        # Debug info
        self.debug_info = {
            "detected_points": 0,
            "z_height": 0,
            "working_height": 0,
            "cutting_depth": self.cutting_depth
        }

        # Locks
        self.lock = threading.Lock()

        # RViz publishers
        self.marker_pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=10)
        self.display_trajectory_publisher = rospy.Publisher(
            '/my_gen3/move_group/display_planned_path',
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20)

        # Subscribe to camera image
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image,
                                          self.image_callback, queue_size=1)

        # ----- Vision & geometry config -----
        self.line_min_area_px = 300
        self.line_min_length_px = 120
        self.skeleton_thin_iterations = 0   # 0=full thinning; >0 caps runtime
        self.equal_waypoint_spacing_mm = 8.0
        self.savgol_window = 9             # odd; increase for more smoothing
        self.savgol_poly = 2

        # ArUco marker on the surface (recommended)
        self._last_plane_scale_mm_per_px = -1.0
        self.T_base_plane = None
        self.surface_frame = 'cutting_plane'
        self.fallback_depth_m = 0.45
        self.image_width_px = 640

        if _HAS_ARUCO:
            self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
            self.aruco_params = aruco.DetectorParameters_create()
            self.board_marker_length_m = 0.03  # side length (meters)
            self.board_marker_id = 0           # marker id
        else:
            rospy.logwarn("cv2.aruco not available. Falling back to approximate pixel->metric scale.")

        rospy.loginfo("Curved Line Follower initialized (detection inactive until robot is positioned).")

    # ------------------- Robot helpers -------------------
    def move_to_defined_position(self):
        rospy.loginfo("Moving to the defined viewing position...")
        # Example joint angles in degrees (adjust to your setup)
        joint_angles = [355.174, 11.702, 184.05, 259.145, 359.699, 302.432, 87.176]
        joint_positions = [self.kinova_to_radians(angle) for angle in joint_angles]
        self.arm_group.set_joint_value_target(joint_positions)
        success = self.arm_group.go(wait=True)
        if success:
            rospy.loginfo("Successfully moved to the defined viewing position")
            current_pose = self.arm_group.get_current_pose().pose
            self.debug_info["z_height"] = current_pose.position.z
            self.debug_info["working_height"] = current_pose.position.z - self.default_working_height
            self.detection_active = True
            return True
        else:
            rospy.logerr("Failed to move to the defined viewing position")
            return False

    def kinova_to_radians(self, deg):
        rad = (deg % 360) * math.pi / 180.0
        if rad > math.pi:
            rad -= 2 * math.pi
        return rad

    # ------------------- Image callback -------------------
    def image_callback(self, data):
        if not self.detection_active or self.line_detected:
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            with self.lock:
                # Update plane & scale via ArUco (if available/visible)
                try:
                    if _HAS_ARUCO:
                        self._maybe_update_plane_from_aruco(cv_image)
                except Exception:
                    pass

                # Detect the line
                line_points_2d, debug_image = self.detect_curved_line(cv_image)
                self.debug_info["detected_points"] = len(line_points_2d)

                is_line_detected_now = len(line_points_2d) >= self.min_line_points
                if is_line_detected_now:
                    self.detection_stable_count += 1
                    self.line_points = line_points_2d
                    self.waypoints = self.create_waypoints(self.line_points, self.num_waypoints)

                    if self.detection_stable_count >= self.required_stability:
                        self.line_detected = True
                        self.detection_active = False
                        rospy.loginfo("Line stably detected. Waypoints locked.")
                        self.publish_raw_line_to_rviz(self.line_points)
                else:
                    self.detection_stable_count = max(0, self.detection_stable_count - 1)
                    if self.detection_stable_count == 0:
                        self.line_detected = False

                self.visualize_detection(cv_image, self.waypoints, debug_image)
        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")

    # ------------------- Vision: detection -------------------
    def detect_curved_line(self, image):
        """
        Robust black-line extractor across variable thickness/texture:
        - CLAHE on L channel to normalize lighting
        - Fuse Otsu + adaptive + 'blackness' masks
        - Pick most elongated contour, then thin to 1-px centerline
        """
        dbg = image.copy()
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        Ln = clahe.apply(L)
        labn = cv2.merge([Ln, A, B])
        norm = cv2.cvtColor(labn, cv2.COLOR_LAB2BGR)
        gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)

        # multi-cue masks
        _, m_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        m_adap = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 31, 5)
        _, m_black = cv2.threshold(Ln, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        mask = cv2.bitwise_and(m_otsu, m_adap)
        mask = cv2.bitwise_or(mask, m_black)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return [], dbg

        def contour_score(cnt):
            area = cv2.contourArea(cnt)
            if area < self.line_min_area_px:
                return -1
            length = len(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            elong = max(w, h) / (min(w, h) + 1e-6)
            return 0.5 * np.log1p(area) + 0.5 * np.log1p(length) + 0.8 * elong

        best = max(contours, key=contour_score)
        if contour_score(best) < 0 or len(best) < self.line_min_length_px:
            return [], dbg

        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [best], -1, 255, thickness=-1)

        skeleton = self.thin_skeleton(contour_mask, max_iter=self.skeleton_thin_iterations)

        ys, xs = np.where(skeleton > 0)
        if len(xs) < self.min_line_points:
            return [], dbg

        pts = np.column_stack([xs, ys]).tolist()
        skmask = (skeleton > 0).astype(np.uint8)
        ordered = self._trace_skeleton_order(skmask, pts)

        # Debug overlay
        try:
            dbg[:, :, 1] = cv2.max(dbg[:, :, 1], mask)  # green = mask
            sk_vis = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
            sk_vis[np.where(skeleton > 0)] = [0, 0, 255]
            dbg = cv2.addWeighted(dbg, 0.9, sk_vis, 1.0, 0)
            cv2.putText(dbg, f"skel pts: {len(ordered)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        except Exception:
            pass

        return ordered, dbg

    def thin_skeleton(self, bin_img, max_iter=0):
        """Zhang–Suen thinning. bin_img: 0/255 uint8. max_iter=0 for full."""
        img = (bin_img.copy() > 0).astype(np.uint8)
        changed = True
        iters = 0
        H, W = img.shape
        while changed:
            changed = False
            # sub-iteration 1
            to_del = []
            for y in range(1, H - 1):
                for x in range(1, W - 1):
                    if img[y, x] == 0:
                        continue
                    p2 = img[y - 1, x]
                    p3 = img[y - 1, x + 1]
                    p4 = img[y, x + 1]
                    p5 = img[y + 1, x + 1]
                    p6 = img[y + 1, x]
                    p7 = img[y + 1, x - 1]
                    p8 = img[y, x - 1]
                    p9 = img[y - 1, x - 1]
                    neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                    if neighbors < 2 or neighbors > 6:
                        continue
                    transitions = int((p2 == 0 and p3 == 1)) + int((p3 == 0 and p4 == 1)) + int((p4 == 0 and p5 == 1)) + \
                                  int((p5 == 0 and p6 == 1)) + int((p6 == 0 and p7 == 1)) + int((p7 == 0 and p8 == 1)) + \
                                  int((p8 == 0 and p9 == 1)) + int((p9 == 0 and p2 == 1))
                    if transitions != 1:
                        continue
                    if p2 * p4 * p6 != 0:
                        continue
                    if p4 * p6 * p8 != 0:
                        continue
                    to_del.append((y, x))
            for (y, x) in to_del:
                img[y, x] = 0
            changed |= bool(to_del)

            # sub-iteration 2
            to_del = []
            for y in range(1, H - 1):
                for x in range(1, W - 1):
                    if img[y, x] == 0:
                        continue
                    p2 = img[y - 1, x]
                    p3 = img[y - 1, x + 1]
                    p4 = img[y, x + 1]
                    p5 = img[y + 1, x + 1]
                    p6 = img[y + 1, x]
                    p7 = img[y + 1, x - 1]
                    p8 = img[y, x - 1]
                    p9 = img[y - 1, x - 1]
                    neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                    if neighbors < 2 or neighbors > 6:
                        continue
                    transitions = int((p2 == 0 and p3 == 1)) + int((p3 == 0 and p4 == 1)) + int((p4 == 0 and p5 == 1)) + \
                                  int((p5 == 0 and p6 == 1)) + int((p6 == 0 and p7 == 1)) + int((p7 == 0 and p8 == 1)) + \
                                  int((p8 == 0 and p9 == 1)) + int((p9 == 0 and p2 == 1))
                    if transitions != 1:
                        continue
                    if p2 * p4 * p8 != 0:
                        continue
                    if p2 * p6 * p8 != 0:
                        continue
                    to_del.append((y, x))
            for (y, x) in to_del:
                img[y, x] = 0
            changed |= bool(to_del)

            iters += 1
            if max_iter > 0 and iters >= max_iter:
                break

        return (img * 255).astype(np.uint8)

    def _trace_skeleton_order(self, skmask, pts):
        """Order skeleton pixels by walking from an endpoint."""
        H, W = skmask.shape

        def neighbors8(x, y):
            for dx, dy in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and skmask[ny, nx] > 0:
                    yield nx, ny

        deg = {}
        for (x, y) in pts:
            deg[(x, y)] = sum(1 for _ in neighbors8(x, y))
        endpoints = [p for p, d in deg.items() if d == 1]
        start = endpoints[0] if endpoints else pts[0]

        path = []
        visited = set()
        cur = start
        prev = None
        while True:
            path.append((cur[0], cur[1]))
            visited.add(cur)
            nbrs = [n for n in neighbors8(cur[0], cur[1]) if n not in visited]
            if prev and prev in nbrs and len(nbrs) > 1:
                nbrs.remove(prev)
            if not nbrs:
                break
            prev, cur = cur, nbrs[0]
        return path

    # ------------------- Waypoints (arc-length spacing) -------------------
    def create_waypoints(self, line_points, num_waypoints=None):
        if len(line_points) < self.min_line_points:
            rospy.logwarn("Not enough line points to create waypoints")
            return []

        pts = np.array(line_points, dtype=np.float32)  # (x,y) in px
        pts_s = self._smooth_polyline(pts)

        diffs = np.diff(pts_s, axis=0)
        seglen = np.linalg.norm(diffs, axis=1)
        s = np.concatenate([[0.0], np.cumsum(seglen)])
        total_px = s[-1]
        if total_px < 5:
            return []

        mm_per_px = self._estimate_mm_per_px()
        spacing_px = max(3.0, self.equal_waypoint_spacing_mm / (mm_per_px + 1e-9))

        target_s = np.arange(0, total_px, spacing_px)
        if target_s[-1] != total_px:
            target_s = np.append(target_s, total_px)

        xs = np.interp(target_s, s, pts_s[:, 0])
        ys = np.interp(target_s, s, pts_s[:, 1])
        sampled = np.stack([xs, ys], axis=1)

        waypoints_with_tangents = []
        for i in range(len(sampled)):
            if i == 0:
                v = sampled[i + 1] - sampled[i]
            elif i == len(sampled) - 1:
                v = sampled[i] - sampled[i - 1]
            else:
                v = sampled[i + 1] - sampled[i - 1]
            n = np.linalg.norm(v)
            t = v / n if n > 1e-9 else np.array([1.0, 0.0])
            p = sampled[i]
            waypoints_with_tangents.append((int(round(p[0])), int(round(p[1])), float(t[0]), float(t[1])))
        return waypoints_with_tangents

    def _smooth_polyline(self, pts):
        if len(pts) < max(self.savgol_window, 5):
            return pts
        if _HAS_SCIPY:
            try:
                x = savgol_filter(pts[:, 0], self.savgol_window, self.savgol_poly, mode='interp')
                y = savgol_filter(pts[:, 1], self.savgol_window, self.savgol_poly, mode='interp')
                return np.stack([x, y], axis=1)
            except Exception:
                pass
        # Fallback simple box filter
        k = 5
        pad = k // 2
        padpts = np.pad(pts, ((pad, pad), (0, 0)), mode='edge')
        sm = np.zeros_like(pts, dtype=np.float32)
        for i in range(len(pts)):
            sm[i] = np.mean(padpts[i:i + k], axis=0)
        return sm

    # ------------------- Pixel -> metric helpers -------------------
    def _estimate_mm_per_px(self):
        if self._last_plane_scale_mm_per_px and self._last_plane_scale_mm_per_px > 0:
            return self._last_plane_scale_mm_per_px
        # approximate using intrinsics and fallback depth
        mm_per_px = (2 * self.fallback_depth_m * math.tan(math.radians(0.6 * 30))) / self.image_width_px * 1000.0
        return mm_per_px

    def _maybe_update_plane_from_aruco(self, bgr):
        if not _HAS_ARUCO:
            return False
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        if ids is None:
            return False
        ids = ids.flatten()
        if self.board_marker_id not in ids:
            return False

        idx = np.where(ids == self.board_marker_id)[0][0]
        c = corners[idx].reshape(-1, 2)  # 4x2

        objp = np.array([
            [-self.board_marker_length_m / 2,  self.board_marker_length_m / 2, 0],
            [ self.board_marker_length_m / 2,  self.board_marker_length_m / 2, 0],
            [ self.board_marker_length_m / 2, -self.board_marker_length_m / 2, 0],
            [-self.board_marker_length_m / 2, -self.board_marker_length_m / 2, 0],
        ], dtype=np.float32)

        ret, rvec, tvec = cv2.solvePnP(objp, c.astype(np.float32),
                                       self.camera_matrix, np.zeros((5,)))
        if not ret:
            return False

        # scale (mm/px) from marker side length
        px_side = np.linalg.norm(c[0] - c[1])
        if px_side > 1e-3:
            self._last_plane_scale_mm_per_px = (self.board_marker_length_m * 1000.0) / px_side

        R, _ = cv2.Rodrigues(rvec)
        T_cam_plane = np.eye(4)
        T_cam_plane[:3, :3] = R
        T_cam_plane[:3, 3] = tvec.flatten()

        try:
            self.tf_listener.waitForTransform('/base_link', '/camera_color_frame',
                                              rospy.Time(0), rospy.Duration(0.5))
            (trans, rot) = self.tf_listener.lookupTransform('/base_link', '/camera_color_frame', rospy.Time(0))
            T_base_cam = tf.transformations.quaternion_matrix(rot)
            T_base_cam[:3, 3] = trans
            self.T_base_plane = T_base_cam.dot(T_cam_plane)
            return True
        except Exception:
            return False

    # ------------------- Visualization -------------------
    def visualize_detection(self, image, waypoints, debug_image):
        vis_image = image.copy()
        for i, wp in enumerate(waypoints):
            x, y, tx, ty = wp[0], wp[1], wp[2], wp[3]
            arrow_length = 50
            end_x = int(x + tx * arrow_length)
            end_y = int(y + ty * arrow_length)
            cv2.arrowedLine(vis_image, (x, y), (end_x, end_y), (255, 0, 0), 3)
            cv2.circle(vis_image, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(vis_image, f"{i+1}", (x + 5, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 105, 180), 2)

        h, w = image.shape[:2]
        combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
        combined[:, :w] = vis_image
        combined[:, w:] = debug_image

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

        try:
            cv2.imshow("Line Detection Debug", combined)
            cv2.waitKey(1)
        except Exception:
            pass  # headless

    def publish_raw_line_to_rviz(self, line_points):
        marker_array = MarkerArray()
        robot_points = self.transform_2d_points_no_tangent(line_points)

        line_marker = Marker()
        line_marker.header.frame_id = "base_link"
        line_marker.header.stamp = rospy.Time.now()
        line_marker.ns = "detected_line_raw"
        line_marker.id = 999
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.pose.orientation.w = 1.0
        line_marker.scale.x = 0.01
        line_marker.color = ColorRGBA(0.0, 1.0, 0.0, 1.0)  # green

        for pt in robot_points:
            p = Point(x=pt[0], y=pt[1], z=pt[2])
            line_marker.points.append(p)

        marker_array.markers.append(line_marker)
        self.marker_pub.publish(marker_array)
        rospy.loginfo("Published 'raw line' to RViz (green).")

    def transform_2d_points_no_tangent(self, image_points):
        """Approximate 3D for RViz line-only preview (no orientations)."""
        try:
            self.tf_listener.waitForTransform('/base_link', '/camera_color_frame',
                                              rospy.Time(0), rospy.Duration(2.0))
            (trans, rot) = self.tf_listener.lookupTransform('/base_link',
                                                            '/camera_color_frame',
                                                            rospy.Time(0))
            transform_matrix = tf.transformations.quaternion_matrix(rot)
            transform_matrix[0:3, 3] = trans

            estimated_depth = self.fallback_depth_m
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
            result_3d = []
            img_cx = self.camera_matrix[0, 2]
            img_cy = self.camera_matrix[1, 2]
            distance_estimate = self.fallback_depth_m
            fov_factor = 0.6
            width_in_px = self.image_width_px
            mm_per_pixel = (2 * distance_estimate * math.tan(math.radians(fov_factor * 30))) / width_in_px
            x_scale = mm_per_pixel / 1000
            y_scale = mm_per_pixel / 1000
            for (px, py) in image_points:
                x_off = (px - img_cx) * x_scale
                y_off = (py - img_cy) * y_scale
                base_x = -y_off
                base_y = -x_off
                base_z = 0.0
                result_3d.append((base_x, base_y, base_z))
            return result_3d

    # ------------------- Waypoints -> robot poses -------------------
    def publish_waypoints_to_rviz(self, robot_poses):
        marker_array = MarkerArray()

        line_marker = Marker()
        line_marker.header.frame_id = "base_link"
        line_marker.header.stamp = rospy.Time.now()
        line_marker.ns = "waypoints"
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.pose.orientation.w = 1.0
        line_marker.scale.x = 0.01
        line_marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)

        for i, pose in enumerate(robot_poses):
            p = Point(x=pose.position.x, y=pose.position.y, z=pose.position.z)
            line_marker.points.append(p)

            sphere = Marker()
            sphere.header.frame_id = "base_link"
            sphere.header.stamp = rospy.Time.now()
            sphere.ns = "waypoints"
            sphere.id = i + 1
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose = pose
            sphere.scale = Vector3(0.02, 0.02, 0.02)
            sphere.color = ColorRGBA(0.0, 0.0, 1.0, 1.0)

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

            marker_array.markers.append(sphere)
            marker_array.markers.append(arrow)

        marker_array.markers.append(line_marker)
        self.marker_pub.publish(marker_array)
        rospy.loginfo("Published waypoints to RViz (ns=waypoints).")

    def transform_image_to_robot_frame(self, image_points_with_tangents):
        """
        Convert 2D pixel coords + tangents => 3D robot coords with orientation.
        Prefers intersecting camera rays with detected surface plane (ArUco),
        falls back to single-depth approximation if needed.
        """
        robot_poses = []
        current_pose = self.arm_group.get_current_pose().pose
        self.debug_info["z_height"] = current_pose.position.z
        working_height = current_pose.position.z - self.default_working_height
        self.debug_info["working_height"] = working_height

        has_plane = self.T_base_plane is not None

        try:
            self.tf_listener.waitForTransform('/base_link', '/camera_color_frame',
                                              rospy.Time(0), rospy.Duration(1.0))
            (trans, rot) = self.tf_listener.lookupTransform('/base_link', '/camera_color_frame', rospy.Time(0))
            T_base_cam = tf.transformations.quaternion_matrix(rot)
            T_base_cam[:3, 3] = trans

            if has_plane:
                Tbp = self.T_base_plane
                R_bp = Tbp[:3, :3]
                p_bp = Tbp[:3, 3]
                R_base_cam = T_base_cam[:3, :3]
                O_base = T_base_cam.dot(np.array([0, 0, 0, 1]))[:3]

            for (px, py, tx, ty) in image_points_with_tangents:
                x_norm = (px - self.camera_matrix[0, 2]) / self.camera_matrix[0, 0]
                y_norm = (py - self.camera_matrix[1, 2]) / self.camera_matrix[1, 1]
                ray_cam = np.array([x_norm, y_norm, 1.0], dtype=np.float64)
                ray_cam = ray_cam / np.linalg.norm(ray_cam)

                if has_plane:
                    ray_base = R_base_cam.dot(ray_cam)
                    O_plane = R_bp.T.dot(O_base - p_bp)
                    d_plane = R_bp.T.dot(ray_base)
                    t = 0.0 if abs(d_plane[2]) < 1e-6 else -O_plane[2] / d_plane[2]
                    P_plane = O_plane + t * d_plane
                    P_base = p_bp + R_bp.dot(P_plane)
                else:
                    depth = self.fallback_depth_m
                    P_cam = np.array([x_norm * depth, y_norm * depth, depth, 1.0])
                    P_base = T_base_cam.dot(P_cam)[:3]

                # tangent via nearby pixel step
                px2, py2 = px + tx * 5.0, py + ty * 5.0
                x2 = (px2 - self.camera_matrix[0, 2]) / self.camera_matrix[0, 0]
                y2 = (py2 - self.camera_matrix[1, 2]) / self.camera_matrix[1, 1]
                ray2_cam = np.array([x2, y2, 1.0], dtype=np.float64)
                ray2_cam = ray2_cam / np.linalg.norm(ray2_cam)

                if has_plane:
                    ray2_base = R_base_cam.dot(ray2_cam)
                    O_plane = R_bp.T.dot(O_base - p_bp)
                    d2_plane = R_bp.T.dot(ray2_base)
                    t2 = 0.0 if abs(d2_plane[2]) < 1e-6 else -O_plane[2] / d2_plane[2]
                    P2_plane = O_plane + t2 * d2_plane
                    P2_base = p_bp + R_bp.dot(P2_plane)
                else:
                    P2_cam = np.array([x2 * self.fallback_depth_m, y2 * self.fallback_depth_m,
                                       self.fallback_depth_m, 1.0])
                    P2_base = T_base_cam.dot(P2_cam)[:3]

                tvec = P2_base - P_base
                txy = tvec[:2]
                n = np.linalg.norm(txy)
                if n > 1e-9:
                    txy /= n

                q = self.calculate_orientation_from_tangent(txy[0], txy[1], current_pose.orientation)

                pose = Pose()
                pose.position.x = float(P_base[0])
                pose.position.y = float(P_base[1])
                pose.position.z = float(working_height)
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
                robot_poses.append(pose)

        except Exception as e:
            rospy.logwarn(f"Transform error: {e}. Falling back to approximate mapping.")
            # approximate fallback using current pose as reference
            current_pose = self.arm_group.get_current_pose().pose
            working_height = current_pose.position.z - self.default_working_height
            self.debug_info["working_height"] = working_height
            self.debug_info["z_height"] = current_pose.position.z

            img_cx = self.camera_matrix[0, 2]
            img_cy = self.camera_matrix[1, 2]
            distance_estimate = self.fallback_depth_m
            fov_factor = 0.6
            width_in_px = self.image_width_px
            mm_per_pixel = (2 * distance_estimate * math.tan(math.radians(fov_factor * 30))) / width_in_px
            x_scale = mm_per_pixel / 1000
            y_scale = mm_per_pixel / 1000

            for i, wp in enumerate(image_points_with_tangents):
                px, py, tx, ty = wp
                x_off = (px - img_cx) * x_scale
                y_off = (py - img_cy) * y_scale

                pose_msg = Pose()
                pose_msg.position.x = current_pose.position.x - y_off
                pose_msg.position.y = current_pose.position.y - x_off
                pose_msg.position.z = working_height

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
        return robot_poses

    def calculate_orientation_from_tangent(self, tangent_x, tangent_y, current_orientation):
        """Align end-effector +X with the line tangent (keep roll & pitch)."""
        current_euler = tf.transformations.euler_from_quaternion([
            current_orientation.x,
            current_orientation.y,
            current_orientation.z,
            current_orientation.w
        ])
        angle = math.atan2(tangent_y, tangent_x)
        new_euler = (current_euler[0], current_euler[1], angle)
        new_quaternion = tf.transformations.quaternion_from_euler(
            new_euler[0], new_euler[1], new_euler[2]
        )
        return new_quaternion

    # ------------------- Tooling: gripper & cutting -------------------
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

            up_pose = copy.deepcopy(cutting_pose)
            up_pose.position.z += self.cutting_depth + 0.01
            self.arm_group.set_pose_target(up_pose)
            self.arm_group.go(wait=True)

            self.debug_info["z_height"] = up_pose.position.z
            rospy.loginfo("Cutting action completed")
        else:
            rospy.logerr("Failed to make contact for cutting")

    # ------------------- Execution: follow the line -------------------
    def follow_line(self):
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

        # Extra slow & gentle
        original_velocity = 0.2
        original_accel = 0.2
        self.arm_group.set_max_velocity_scaling_factor(0.03)
        self.arm_group.set_max_acceleration_scaling_factor(0.03)

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
                rospy.sleep(1.0)  # steadier on tissue/gelatin
            except Exception as e:
                rospy.logerr(f"Error moving to waypoint {i+1}: {str(e)}")

# ------------------- Main -------------------
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
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

