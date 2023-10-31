#!/usr/bin/env python3

import rospy
from move_base_msgs.msg import MoveBaseGoal, MoveBaseAction
from actionlib import SimpleActionClient
from geometry_msgs.msg import Pose, Quaternion
import sys
import tty
import termios
import mysql.connector
import networkx as nx
import math

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
        

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


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

#유클리디안 거리 계산
def calculate_distance(node1, node2):
    x1, y1 = node1
    x2, y2 = node2
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

#A* 알고리즘
def find_shortest_path(graph, start, end):
    return nx.astar_path(graph, start, end, heuristic=calculate_distance)

def main():
    
    coordinate_list = []
    result = []
    #db debugging
    product_name_input = input("상품 이름을 입력하세요: ")
    product_names = product_name_input.split()

    for product_name in product_names:
        coordinates = fetch_coordinates_from_mysql(product_name)
        if coordinates:
            for x, y in coordinates:
                coordinate_list.append((x,y))
                print(f"상품 '{product_name}'의 좌표는 ({x}, {y})입니다.")
        else:
            print(f"상품 '{product_name}'의 좌표를 찾을 수 없습니다.")
  
    coordinate_list = remove_duplicates_2d(coordinate_list)
    print(f"중복 제거  좌표는 '{coordinate_list}' 입니다.") 
    
    graph = nx.Graph()

    #좌표 그래프에 추가
    for coordinate in coordinate_list:
        #list를 튜플로 변환해줘야 함(list와 달리 불변하므로 해시 가능)
        x, y = coordinate
        graph.add_node((x, y))

    coordinate_list = [tuple(coord) for coord in coordinate_list]
    
    #모든 좌표 사이의 거리 계산하여 그래프에  추가
    for i in range(len(coordinate_list)):
        for j in range(i+1, len(coordinate_list)):
           # x1, y1 = coordinate_list[i]
           # x2, y2 = coordinate_list[j]
            distance = calculate_distance(coordinate_list[i], coordinate_list[j])
            graph.add_edge(coordinate_list[i], coordinate_list[j], weight=distance)

    shortest_paths = []
    for i in range(len(coordinate_list)):
        for j in range(i+1, len(coordinate_list)):
                start = coordinate_list[i]
                end = coordinate_list[j]
                path = find_shortest_path(graph, start, end)
                shortest_paths.append((path, sum(calculate_distance(start, end)for start, end in zip(path, path[1:]))))
    shortest_paths.sort(key = lambda x: x[1])                
    print(f"최단 경로는 '{shortest_paths}' 순 입니다.")        

    #navigation debugging
    rospy.init_node('simple_navigation_goals')

    goals = [
        (1.9836953484104076, -0.2047552294254606),
        (1.1321254607786913, 0.22083921565921813),
        (0.35052772035182855, -0.1556799485199058)
        # Add more goals if needed
        ]

    for goal in goals:
        while True:
            print("터틀봇의 이동을 원한다면 g키를 누르세요")
            key = getch()
            if key == 'g':
                break

        move_to_goal(goal[0], goal[1])


if __name__ == '__main__':
    main()
    
