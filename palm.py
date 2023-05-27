import time

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from flask import Flask, render_template, request, make_response, redirect
from sqlite import *
from typing import List
from threading import Thread, Lock


app = Flask(__name__)
cap = None
model = None
photos_of_hand = 10
classes = None
admin_pass = "Admin1234"
con = None
ip_addr = '172.30.30.5:554'
db_lock = Lock()
cam_lock = Lock()
model_lock = Lock()
class_lock = Lock()


def process_image(image):
    img = image
    img = cv2.resize(img, [600, 600])
    # noise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise = cv2.fastNlMeansDenoising(gray)
    noise = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
#    print ("reduced noise")

    # equalist hist
    kernel = np.ones((7,7),np.uint8)
    img = cv2.morphologyEx(noise, cv2.MORPH_OPEN, kernel)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
#    print ("equalize hist")

    # invert
    inv = cv2.bitwise_not(img_output)
#    print ("inverted")

    # erode
    gray = cv2.cvtColor(inv, cv2.COLOR_BGR2GRAY)
    erosion = cv2.erode(gray,kernel,iterations = 1)
#    print ("eroded")

    # skel
    img = gray.copy()
    skel = img.copy()
    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
    iterations = 0

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break

#    print ("skel done")
    _, thr = cv2.threshold(skel, 5, 255, cv2.THRESH_BINARY)

    return thr


def train_model(num_users: int, files: List[str], num_photos: int):
    global model

    train_labels = []
    for i in range(num_users):
        train_labels = train_labels + [i] * num_photos
    train_labels = np.array(train_labels)

    pic = np.array(Image.open(files[0]))
    train_images = np.array([pic])
    files = files[1:]
    for image in files:
        pic = np.array(Image.open(image))
        train_images = np.vstack((train_images, np.array([pic])))

    train_images = train_images / 255.0

    model = keras.Sequential()
    model.add(keras.layers.Reshape((600, 600, 1), input_shape=train_images.shape[1:]))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.50))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dropout(0.6))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_users, activation='softmax'))

    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=7)
#    print("Train complete")


def configure_cap(ip_addr: str):
    global cap
    cap = cv2.VideoCapture('rtsp://' + ip_addr + '/user=admin&password=&channel=1&stream=0.sdp?real_stream')


def close_cap():
    global cap
    cap.release()


def crop_image(image):
    return image[310:900, 620:1280]


def get_image():
    configure_cap(ip_addr)
    _, frame = cap.read()
    close_cap()
    return frame


def check_predict(image) -> str:
    global model, classes
    image = process_image(crop_image(image))
    cv2.imwrite(users_dir + "tmp.bmp", image)
    image = np.array(Image.open(users_dir + "tmp.bmp"))
    image = np.array([image])
#    print("predicting result")
    predictions = model.predict(image)
    print(predictions)
#    print( "final answer:")
#    print( get_users()[1][np.argmax(predictions[0])])
    with class_lock:
        return classes[np.argmax(predictions[0])]


def image_checker_thread():
    global cam_lock, model_lock
    with cam_lock:
        image = get_image()
    user = None
    with model_lock:
        user = check_predict(image)

    if user == 'other':
        time.sleep(0.2)
        image_checker_thread()

    print("User %s detected" % (user,))

    val = None
    with db_lock:
        val = get_val_from_col("status", user)

    if val == "Online":
        val = "Offline"
    else:
        val = "Online"

    with db_lock:
        replace_val_in_col("status", val, user)

    time.sleep(5)
    image_checker_thread()


def process_image_and_save(image, folder: str, name: str, number: int):
    global image_num, processed_folder
    image = crop_image(image)
    cv2.imwrite(folder + '\\' + name + str(number) + ".bmp", image)
    processed = process_image(image)
    cv2.imwrite(folder + '\\' + name + "_prc" + str(number) + ".bmp", processed)


def process_new_user(user: str, num_photos: int):
    for i in range(num_photos):
        while True:
            image = get_image()
            cv2.imshow('frame', crop_image(image))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                break
        process_image_and_save(image, users_dir + user, user, i)
        time.sleep(0.5)

    with db_lock:
        replace_val_in_col("photos", "10", user)


def get_files_by_users(users: List[str], type: int) -> List[str]:
    """
    :param users: List of user names/logins
    :param type: Type of photos to get, i.e. simple photos or processed, where 0 is simple and 1 is processed
    :return: List of files for given users
    """
    files = []

    if type == 0:
        additional = ''
    else:
        additional = '_prc'

    for user in users:
        num_of_photos = int(get_val_from_col("photos", user))
        for i in range(num_of_photos):
            files.append(users_dir + user + "\\" + user + additional + str(i) + '.bmp')

    return files


def prepare_table() -> List[Tuple]:
    with db_lock:
        table = get_all_from_table()
    result_table = []
    counter = 0
    for row in table:
        if row[0] == 'other':
            continue
        else:
            counter = counter + 1
        result_table.append((counter, row[1] + ' ' + row[2], row[3], row[0]))

    return result_table


@app.route('/status', methods=['GET'])
def status():
    with db_lock:
        ustatus = users_status()
    return ustatus


@app.route('/register', methods=['POST', 'GET'])
def register():
    global classes
    if request.method == 'POST':
        form = request.form
        fname = form['name']
        lname = form['lastname']
        with db_lock:
            add_new_user(fname + lname, fname, lname, "Offline")
        with cam_lock:
            process_new_user(fname + lname, photos_of_hand)
        with model_lock:
            with db_lock:
                with class_lock:
                    classes = get_users()[1]
                train_model(get_users()[0], get_files_by_users(classes, 1), photos_of_hand)
        time.sleep(5)
        resp = redirect("/db", code=302)
    else:
        resp = make_response(render_template('register.html', table=prepare_table()))

    return resp


@app.route('/db', methods=['POST', 'GET'])
def main_db():
    global classes
    resp = make_response(render_template('db.html', table=prepare_table()))

    if request.method == 'POST':
        form = request.form
        if 'add-new-user' in form:
            resp = redirect("/register", code=302)
        elif 'delete-user' in form:
            with db_lock:
                del_user_by_login(form['delete-user'])
            resp = make_response(render_template('db.html', table=prepare_table()))
            with model_lock:
                with db_lock:
                    with class_lock:
                        classes = get_users()[1]
                    train_model(get_users()[0], get_files_by_users(classes, 1), photos_of_hand)
            time.sleep(5)
        elif 'calibrate-button' in form:
            with cam_lock:
                while True:
                    image = get_image()
                    cv2.imshow('frame', crop_image(image))
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                cv2.destroyAllWindows()
    else:
        pass
    return resp


@app.route('/', methods=['POST', 'GET'])
def main():
    user_id = request.cookies.get('UserID')
    if user_id == '0':
        resp = redirect("/db", code=302)
    else:
        resp = make_response(render_template('index.html'))
        if request.method == 'POST':
            form = request.form
            if form['login'] == 'admin' and form['password'] == admin_pass:
                resp = redirect("/db", code=302)
                resp.set_cookie('UserID', "0")
        else:
            resp.set_cookie('UserID', "1")

    return resp


def init():
    global classes
    configure_cap(ip_addr)
    open_db("users.db")
    t = Thread(target=image_checker_thread)
    t.start()
    with model_lock:
        with db_lock:
            with class_lock:
                classes = get_users()[1]
            train_model(get_users()[0], get_files_by_users(classes, 1), photos_of_hand)


if __name__ == '__main__':
    init()
    app.run()
#    files = get_files_by_users(get_users()[1], 1)
#    train_model(get_users()[0], files, photos_of_hand)
#
#    while True:
#        image = get_image()
#        cv2.imshow('frame', crop_image(image))
#        key = cv2.waitKey(1) & 0xFF
#        if key == ord('q'):
#            break
#        if key == ord('c'):
#            check_predict(image)
