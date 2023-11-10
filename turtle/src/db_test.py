import mysql.connector

# MySQL 서버에 연결
conn = mysql.connector.connect(user='USER', password='PW', host='DB_IP', database='DB_NAME')
cursor = conn.cursor()

# 좌표를 불러오는 SQL 쿼리 실행
cursor.execute(f"SELECT x_coordinate, y_coordinate FROM Cart JOIN Products ON Cart.product_name = Products.product_name")
coordinates = cursor.fetchall()
print(coordinates)
