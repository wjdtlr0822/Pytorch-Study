import sqlite3

print(sqlite3.version)

conn = sqlite3.connect("test.db")
cursor = conn.cursor()

#테이블 생성
cursor.execute(
    """
    CREATE TABLE TESTDB (
        id INTEGER PRIMARY KEY,
        name TEXT,
        age INTEGER
    )
    """
)

cursor.executemany("""
    INSERT INTO TESTDB (name, age) VALUES (?, ?)
""", [
    ('Alice', 25),
    ('Bob', 30),
    ('Charlie', 35)
])

conn.commit()
conn.close()