import mysql.connector

# MySQL에 연결
db_connection = mysql.connector.connect(
    host="3.39.54.145",
    user="root",
    password="971226",
    database="3.39.54.145"
)

# 커서 생성
cursor = db_connection.cursor()

# 사용자로부터 상품 이름 입력 받기
product_name = input("검색할 상품 이름을 입력하세요: ")

# 쿼리 실행
query = f"SELECT x_coordinate, y_coordinate FROM mytable WHERE product_name = '{product_name}'"
cursor.execute(query)

# 결과 불러오기
result = cursor.fetchall()

# 연결 종료
cursor.close()
db_connection.close()

# 결과 출력
if result:
    print(f"{product_name}의 좌표:")
    for row in result:
        print(f"x_coordinate: {row[0]}, y_coordinate: {row[1]}")
else:
    print(f"{product_name}은(는) 데이터베이스에 없습니다.")
