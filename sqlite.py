import os
import sqlite3
import datetime
from typing import Tuple, List

con = None
users_dir = "users\\"


def open_db(name: str):
    global con

    if os.path.exists(name):
        con = sqlite3.connect(name, check_same_thread=False)
    else:
        con = sqlite3.connect(name, check_same_thread=False)
        con.cursor().execute("CREATE TABLE users(login, name, lastname, status, photos)")


def add_new_user(login: str, name: str, lastname: str, status: str):
    global con, users_dir
    data = (login, name, lastname, status)
    for row in con.cursor().execute("SELECT login, name, lastname, status FROM users"):
        if row == data:
            return
    con.cursor().execute("INSERT INTO users VALUES(?, ?, ?, ?, 0)", data)
    con.commit()

    if not os.path.exists(users_dir + login):
        os.makedirs(users_dir + login)


def del_user_by_all(login: str, name: str, lastname: str, status: str):
    global con
    data = (login, name, lastname, status)
    con.cursor().execute("DELETE FROM users WHERE login=? AND name=? AND lastname=? AND status=?", data)
    con.commit()

    if os.path.exists(users_dir + login):
        os.rename(users_dir + login, users_dir + login + '-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))


def del_user_by_login(login: str):
    global con
    data = tuple([login])
    con.cursor().execute("DELETE FROM users WHERE login=?", data)
    con.commit()

    if os.path.exists(users_dir + login):
        os.rename(users_dir + login, users_dir + login + '-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))


def replace_val_in_col(column: str, val: str, user: str):
    if column == 'login':
        return
    con.cursor().execute("UPDATE users SET %s = '%s' WHERE login = '%s'" % (column, val, user))
    con.commit()


def get_users() -> Tuple[int, List[str]]:
    cursor = con.cursor()
    cursor.execute("SELECT * FROM users")
    db_users = cursor.fetchall()
    users = []
    for user in db_users:
        users.append(user[0])

    users.remove('other')
    users.append('other')

    return len(db_users), users


def get_val_from_col(column: str, user: str):
    cursor = con.cursor()
    cursor.execute("SELECT %s FROM users WHERE login = '%s'" % (column, user))
    return cursor.fetchall()[0][0]


def get_all_from_table():
    cursor = con.cursor()
    cursor.execute("SELECT * FROM users")
    return cursor.fetchall()


def users_status():
    users = get_users()
    status = {user: get_val_from_col("status", user) for user in users[1]}

    return status
