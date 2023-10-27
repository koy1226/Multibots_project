#!/usr/bin/env python3

# 1. 최단 거리 경로 좌표로 이동하는 코드
# 2. db cart table 만들고 product table이랑 브렌치, 사용자가 넣은 상품의 좌표 값 불러오기
# 3. 최단 거리 경로 좌표로 이동 코드 2번에 맞게 수정
# 

import rospy
from move_base_msgs.msg import MoveBaseGoal, MoveBaseAction
from actionlib import SimpleActionClient
from geometry_msgs.msg import Pose, Quaternion, Twist
import sys
import tty
import termios
import mysql.connector
import networkx as nx
from itertools import permutations
import math
import threading


def fetch_coordinates_from_mysql(product_name):
    try:
        # MySQL 서버에 연결
        conn = mysql.connector.connect(user='root', password='971226', host='3.39.54.145', database='Mart')
        cursor = conn.cursor()

        # 좌표를 불러오는 SQL 쿼리 실행
        cursor.execute(f"SELECT x_coordinate, y_coordinate FROM Products WHERE product_name = '{product_name}'")
        coordinates = cursor.fetchall()

        return coordinates

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

#키보드 인터럽트        
def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

########## 목표 지점 이동 #########
#turtlebot 목표 지점으로 이동
def move_to_goal(x_goal, y_goal):
    ac = SimpleActionClient('move_base', MoveBaseAction)
    ac.wait_for_server(rospy.Duration(5.0))

    goal = MoveBaseGoal()

    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()

    goal.target_pose.pose.position.x = x_goal
    goal.target_pose.pose.position.y = y_goal
    goal.target_pose.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)

    ac.send_goal_and_wait(goal, rospy.Duration(10.0, 0), rospy.Duration(10.0, 0))

    if ac.get_state() == 3:  # actionlib.SimpleClientGoalState.SUCCEEDED
        print(f"Goal arrived at ({x_goal}, {y_goal})!")
    else:
        print(f"The base failed to move to goal ({x_goal}, {y_goal}) for some reason")
        
#중복 처리
def remove_duplicates_2d(input_list):
    # 2차원 리스트를 1차원 리스트로 변환하고 중복을 제거한 뒤 다시 2차원 리스트로 변환
    return [list(item) for item in set(tuple(row) for row in input_list)]

#최단 이동 거리 계산
def find_shortest_path_among_all(graph, all_paths):
    shortest_path = None
    shortest_distance = float('inf')

    for path in all_paths:
        total_distance = calculate_total_distance(graph, path)
        if total_distance < shortest_distance:
            shortest_distance = total_distance
            shortest_path = path

    return shortest_path, shortest_distance

#db로 부터 좌표값 불러오기
def get_all_waypoints_coordinates(product_names):
    waypoint_coordinates = []

    for product_name in product_names:
        coordinates = fetch_coordinates_from_mysql(product_name)
        if coordinates:
            for x, y in coordinates:
                waypoint_coordinates.append((x, y))
                print(f"상품 '{product_name}'의 좌표는 ({x}, {y})입니다.")
        else:
            print(f"상품 '{product_name}'의 좌표를 찾을 수 없습니다.")

    return waypoint_coordinates

#모든 경우의 수를 계산
def get_all_possible_paths(waypoints):
    all_possible_paths = list(permutations(waypoints))
    
    print(f"모든 경우의 수: '{all_possible_paths}'")
    return all_possible_paths

############# 유클리디안 ##############
#all_possibel_paths에서 튜플들간의 누적 거리값 & 이 거리값 중 가장 작은 값 찾기
def calculate_total_distance(tuples):
    #초기 위치(현재 위치로 다시 수정)
    
    
    # 각 튜플의 좌표들을 순회하며 이동 거리 계산
    #모든 경우의 수의 list 속 튜플 추출
    total_distances = []

    for tuple in tuples:
        # 초기 위치
        current_position = (0, 0)

        # 이동 거리 누적 변수 초기화
        total_distance = 0

        # 각 튜플의 좌표들을 순회하며 이동 거리 계산
        for coord in tuple:
            # 현재 위치에서 다음 좌표까지의 거리 계산
            distance = math.sqrt((coord[0] - current_position[0])**2 + (coord[1] - current_position[1])**2)
        
            # 현재 위치를 다음 좌표로 업데이트
            current_position = coord
        
            # 누적 이동 거리에 추가
            total_distance += distance
    
        # 누적 이동 거리를 리스트에 추가
        total_distances.append(total_distance)

    # 가장 작은 누적 이동 거리를 가진 튜플을 추출
    min_distance_index = total_distances.index(min(total_distances))
    min_distance_tuple = tuples[min_distance_index]
    
    print(f"최종 최단 이동 거리 좌표는 '{min_distance_tuple}'입니다.")
    
    return min_distance_tuple

#turtlebot 정지하는 함수
def stop_turtlebot():
    rospy.init_node('stop_turtlebot_node', anonymous=True)
    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rate = rospy.Rate(10)  # 10Hz

    stop_msg = Twist()
    cmd_vel_pub.publish(stop_msg)
    rospy.loginfo("TurtleBot3 stopped.")

# 원제님 코드
# 고객을 추적하는 함수
def tracking_customer():
    while True:
        # 고객 추적 로직
        distance = 0  # 고객과의 거리를 측정 (예시)
        
        if distance > 10:  # 10m 이상 떨어진 경우
            print("Customer is too far. Stopping turtle bot.")
            # 터틀봇 정지 로직
            stop_turtlebot()
            break
        else:
            main_thread.resume()
            
            
        rospy.sleep(1)  # 1초에 한 번씩 거리를 체크


def main_thread():
    while not rospy.is_shutdown():
        #db debugging
        product_name_input = input("상품 이름을 입력하세요: ")
        product_names = product_name_input.split()
    
        # 모든 좌표를 수집
        waypoint_coordinates = get_all_waypoints_coordinates(product_names)
    
        # 중복된 좌표 제거
        waypoint_coordinates = remove_duplicates_2d(waypoint_coordinates)
        print(f"중복 제거 좌표는 '{waypoint_coordinates}' 입니다.")
    
        #모든 경우의 수를 여러개의 튜플로 계산
        all_possible_paths = get_all_possible_paths(waypoint_coordinates)

        #최소 이동 거리 path 계산
        calculate_total_distance(all_possible_paths)
    
        #goals = [
        #    (1.9836953484104076, -0.2047552294254606),
        #    (1.1321254607786913, 0.22083921565921813),
        #    (0.35052772035182855, -0.1556799485199058)
        #    ] 기존은 list안 튜플, all_possible_paths는 튜플안 list ([],[]...)
    
        #목표 지점으로 이동
        for goal in all_possible_paths:
            while True:
                print("터틀봇의 이동을 원한다면 g키를 누르세요")
                key = getch()
                if key == 'g':
                    break
            #w = 0.01
            move_to_goal(goal[0], goal[1])

if __name__ == '__main__':
    try:
        rospy.init_node('simple_navigation_goals')
        
        #1. turtlebot3 - main 스레드
        main_thread = threading.Thread(target=main_thread)
        main_thread.start()
        
        #turtle 경로이동 중 고객이 일정거리 이상 떨어지면(depth캠에서 측정) turtle bot 정지
        #문제 상황 해결되면 경로이동 이어서
        
        #2. AI - 고객 추적 스레드(mutithreading $ interrupt)
        tracking_customer = threading.Thread(target = tracking_customer)
        tracking_customer.start()
        #3. AI - 미아 찾기 스레드(interrupt)
        
        # 스레드들이 종료될 때까지 기다림
        main_thread.join()
        tracking_customer.join()
        
    except rospy.ROSInterruptException:
        print("ROS interrupted. Exiting.")
        
    
