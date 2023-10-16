import networkx as nx
import heapq

# 그래프 생성 및 초기화 (상품 위치도 노드로 추가)
G = nx.Graph()
# ... (노드와 엣지를 실제 지도에 맞게 추가)
product_coordinates = {
    'apple': 'node_5',
    'banana': 'node_10',
    'carrot': 'node_7',
    # ...
}

# A* 알고리즘을 이용한 경로 계획 함수
def astar_path(graph, start, end, heuristic_func=None):
    open_set = [(0 + heuristic_func(start, end), 0, start, [])]
    closed_set = set()
    
    while open_set:
        _, g, current, path = heapq.heappop(open_set)
        
        if current == end:
            return path + [end]
        
        if current in closed_set:
            continue
        closed_set.add(current)
        
        for neighbor in graph.neighbors(current):
            if neighbor in closed_set:
                continue
            
            g_new = g + graph.edges[current, neighbor].get('weight', 1)
            f_new = g_new + heuristic_func(neighbor, end)
            
            heapq.heappush(open_set, (f_new, g_new, neighbor, path + [current]))

# 휴리스틱 함수 (유클리디언 거리)
def euclidean(node1, node2):
    x1, y1 = G.nodes[node1]['coordinates']
    x2, y2 = G.nodes[node2]['coordinates']
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

# 상품 리스트로 최단 경로 계획 함수
def calculate_path(graph, product_list, heuristic_func=euclidean):
    start_node = 'entrance'
    end_node = 'exit'
    path = [start_node]
    
    for product in product_list:
        next_node = product_coordinates[product]
        sub_path = astar_path(graph, path[-1], next_node, heuristic_func)
        path += sub_path[1:]
    
    path += astar_path(graph, path[-1], end_node, heuristic_func)[1:]
    return path

# 예시 상품 리스트
example_product_list = ['apple', 'banana', 'carrot']

# 경로 계획
result_path = calculate_path(G, example_product_list)
print("Planned path:", result_path)
