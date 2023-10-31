import mysql.connector

# 데이터베이스 연결 정보 설정
db_config = {
    "host": "3.39.54.145",
    "user": "root",
    "password": "971226",
    "database": "Mart"
}

try:
    # MySQL 데이터베이스에 연결
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    # 트랜잭션 시작
    cursor.execute("START TRANSACTION;")
    cursor.execute("SET SQL_SAFE_UPDATES = 0;")
    
    # Cart 테이블 내용 삭제
    cursor.execute("DELETE FROM Cart;")

    # shoppinglist.txt 파일에서 상품 목록 읽기
    with open("shoppinglist.txt", "r") as file:
        shoppinglist = file.read().splitlines()

    # 각 상품을 Cart 테이블에 추가
    for product in shoppinglist:
        insert_query = "INSERT INTO Cart (product_name) VALUES (%s)"
        cursor.execute(insert_query, (product,))

    cursor.execute("SET SQL_SAFE_UPDATES = 1;")
    cursor.execute("COMMIT;")
    print("SQL commands executed successfully.")

except mysql.connector.Error as err:
    print("MySQL Error:", err)

finally:
    if conn.is_connected():
        cursor.close()
        conn.close()
        print("Database connection closed.")

