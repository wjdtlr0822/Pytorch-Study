import sqlite3

conn = sqlite3.connect("test.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM TESTDB")
rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()

###원격서버에 연결하기 위해서###
#import psycopg2
#
#conn = psycopg2.connect(
#    host="127.0.0.1",
#    user="postgres",
#    password="1234",
#    dbname="testdb"
#)
###

### GROUP BY 예시
#SELECT survived, COUNT(*) 
#FROM passengers 
#GROUP BY survived;
###

### JOIN 예시
#SELECT u.name, o.amount
#FROM users u
#JOIN orders o ON u.id = o.user_id;
###