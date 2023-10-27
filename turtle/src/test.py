#!/usr/bin/env python3

import rospy
from move_base_msgs.msg import MoveBaseGoal, MoveBaseAction
from actionlib import SimpleActionClient
from geometry_msgs.msg import Pose, Quaternion

def main():
    rospy.init_node('simple_navigation_goals')

    # Tell the action client that we want to spin a thread by default
    ac = SimpleActionClient('move_base', MoveBaseAction)
    ac.wait_for_server(rospy.Duration(5.0))

    goal = MoveBaseGoal()

    goal.target_pose.header.frame_id = "map"  # "map" for global frame. "base_link" for local frame;
    goal.target_pose.header.stamp = rospy.Time.now()

    # Goal 1
    goal.target_pose.pose.position.x = 1.9836953484104076
    goal.target_pose.pose.position.y = -0.2047552294254606
    goal.target_pose.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)  # Quaternion

    print("Sending goal 1")
    ac.send_goal_and_wait(goal, rospy.Duration(10.0, 0), rospy.Duration(10.0, 0))

    if ac.get_state() == 3:  # actionlib.SimpleClientGoalState.SUCCEEDED
        print("Goal arrived!")
    else:
        print("The base failed to move to goal for some reason")

    # Goal 2
    goal.target_pose.pose.position.x = 0.35052772035182855
    goal.target_pose.pose.position.y = -0.1556799485199058
    goal.target_pose.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)  # Quaternion

    print("Sending goal 2")
    ac.send_goal_and_wait(goal, rospy.Duration(20.0, 0), rospy.Duration(20.0, 0))

    if ac.get_state() == 3:  # actionlib.SimpleClientGoalState.SUCCEEDED
        print("Goal arrived!")
    else:
        print("The base failed to move to goal for some reason")

    '''
    # Goal 3
    goal.target_pose.pose.position.x = 0.0
    goal.target_pose.pose.position.y = -0.5
    goal.target_pose.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)  # Quaternion

    print("Sending goal 3")
    ac.send_goal_and_wait(goal, rospy.Duration(20.0, 0), rospy.Duration(20.0, 0))

    if ac.get_state() == 3:  # actionlib.SimpleClientGoalState.SUCCEEDED
        print("Goal arrived!")
    else:
        print("The base failed to move to goal for some reason")
    '''

if __name__ == '__main__':
    main()
