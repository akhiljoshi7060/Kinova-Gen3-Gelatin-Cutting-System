#!/usr/bin/env python3
import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped

if __name__ == "__main__":
    rospy.init_node("go_to_pose", anonymous=True)
    moveit_commander.roscpp_initialize([])
    group = moveit_commander.MoveGroupCommander("arm")
    # Define your target pose:
    target = PoseStamped()
    target.header.frame_id = "base_link"
    target.pose.position.x = 0.583
    target.pose.position.y = -0.016
    target.pose.position.z = 0.748
    # Use the quaternion you provided:
    target.pose.orientation.x = 0.056
    target.pose.orientation.y = -0.002
    target.pose.orientation.z = 0.999
    target.pose.orientation.w = 0.018

    group.set_pose_target(target)
    group.go(wait=True)
    group.stop()
    group.clear_pose_targets()
    rospy.loginfo("Reached target pose")

