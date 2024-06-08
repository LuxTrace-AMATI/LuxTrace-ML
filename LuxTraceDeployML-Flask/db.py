import os
import pymysql
from flask import jsonify

db_user = os.environ.get('CLOUD_SQL_USERNAME')
db_password = os.environ.get('CLOUD_SQL_PASSWORD')
db_name = os.environ.get('CLOUD_SQL_DATABASE_NAME')
db_connection_name = os.environ.get('CLOUD_SQL_CONNECTION_NAME')

def open_connection():
    unix_socket = '/cloudsql/{}'.format(db_connection_name)
    try:
        if os.environ.get('GAE_ENV') == 'standard':
            conn = pymysql.connect(user=db_user,
                                   password=db_password,
                                   unix_socket=unix_socket,
                                   db=db_name,
                                   cursorclass=pymysql.cursors.DictCursor
                                   )
        else:
            conn = pymysql.connect(user=db_user,
                                   password=db_password,
                                   host='localhost',  # atau host lain sesuai konfigurasi lokal Anda
                                   db=db_name,
                                   cursorclass=pymysql.cursors.DictCursor
                                   )
    except pymysql.MySQLError as e:
        return e
    return conn

def get(table_name):
    conn = open_connection()
    with conn.cursor() as cursor:
        result = cursor.execute(f'SELECT * FROM {table_name};')
        records = cursor.fetchall()
        if result > 0:
            got_records = jsonify(records)
        else:
            got_records = 'No records in DB'
        conn.close()
        return got_records

def create(table_name, record):
    conn = open_connection()
    with conn.cursor() as cursor:
        placeholders = ', '.join(['%s'] * len(record))
        columns = ', '.join(record.keys())
        sql = f'INSERT INTO {table_name} ({columns}) VALUES ({placeholders})'
        cursor.execute(sql, list(record.values()))
    conn.commit()
    conn.close()
