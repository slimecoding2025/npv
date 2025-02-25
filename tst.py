import mysql.connector

try:
    connection = mysql.connector.connect(
        host="127.0.0.1",  # أو localhost
        database="aiprojet",
        user="turbo_user",
        password="hichemmajida"
    )
    if connection.is_connected():
        print("Connected to MySQL database!")
except Exception as e:
    print(f"Error: {e}")
finally:
    if connection.is_connected():
        connection.close()