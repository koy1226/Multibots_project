#!/usr/bin/env python3

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

def move_to_goal(x_goal, y_goal):

    # 액션 서버에 연결
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    client.wait_for_server()

    # 목표 설정
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = x_goal
    goal.target_pose.pose.position.y = y_goal
    goal.target_pose.pose.orientation.w = 1.0

    # 목표 지점으로 이동
    client.send_goal(goal)
    wait = client.wait_for_result()

    if not wait:
        rospy.logerr("Action server not available!")
        rospy.signal_shutdown("Action server not available!")
    else:
        return client.get_result()

if __name__ == '__main__':
    try:
        rospy.init_node('move_base_sequence')

        # 좌표 설정
        coords = [
            (0.8150421028616327, -0.7554679836632177),  # 1번
            (0.9878010600029122, 0.09899063881582439),    # 9번
            (0.35052772035182855, -0.1556799485199058)] # 시작 위치

        # 시작 위치에서 1번, 9번을 찍고 다시 시작 위치로 돌아오기
        for coord in coords:
            result = move_to_goal(coord[0], coord[1])
            if result:
                rospy.loginfo("Goal execution done!")

    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation test finished.")
