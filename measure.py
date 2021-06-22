from flask import Flask, render_template, request
import os
import cv2
# import dlib
# # from imutils import face_utils
# import pickle
# from scipy.spatial import distance
# import pandas as pd
# from skimage.feature import hog
# from skimage.transform import resize
import numpy as np
# import math
# import matplotlib.pyplot as plt
import csv

from hog_extraction import extract_hog_feature

# Khởi tạo Flask
app = Flask(__name__)

# Cấu hình thư mục sẽ upload file lên
app.config['UPLOAD_FOLDER'] = 'static'


def euclide_distance(x, y):

    distance = 0.0
    for i in range(0, len(x)):
        distance += (x[i] - y[i])**2
    distance = np.sqrt(distance)
    return distance


def findMinAverageDistance(img):

    # Khởi tạo giá trị ban đầu
    min_average_dist = 1000000000.0
    final_filepath = []

    # Lấy ra vector đặc trưng của các vùng của ảnh đầu vào
    hog_feature_tran, hog_feature_mat, hog_feature_mui, hog_feature_mieng = extract_hog_feature(
        img)

    # Duyệt toàn bộ file hog_feature: file lưu trữ các giá trị feature của các ảnh trong csdl
    with open("hog_feature.csv") as f:
        reader = csv.reader(f, delimiter=',')
        cnt = 0
        filepath = []
        feature_tran = []
        feature_mat = []
        feature_mui = []
        feature_mieng = []

        for row in reader:
            cnt += 1
            # Dòng đầu tiên là tên file
            if cnt % 5 == 1:
                filepath = row
            # Dòng thứ 2 là vector đặc trung cho vùng trán
            elif cnt % 5 == 2:
                feature_tran = row
            # Dòng thứ 3 là vector đặc trung cho vùng mắt
            elif cnt % 5 == 3:
                feature_mat = row
            # Dòng thứ 4 là vector đặc trung cho vùng mũi
            elif cnt % 5 == 4:
                feature_mui = row
            # Dòng thứ 5 là vector đặc trung cho vùng miệng
            else:
                feature_mieng = row

            # Tiến hành tính khoảng cách euclid đối với từng vùng của ảnh trong database
            # với ảnh đầu vào
            if cnt % 5 == 0:
                dist_tran = euclide_distance(
                    np.array(hog_feature_tran), np.array(feature_tran, dtype='float64'))
                dist_mat = euclide_distance(
                    np.array(hog_feature_mat), np.array(feature_mat, dtype='float64'))
                dist_mui = euclide_distance(
                    np.array(hog_feature_mui), np.array(feature_mui, dtype='float64'))
                dist_mieng = euclide_distance(
                    np.array(hog_feature_mieng), np.array(feature_mieng, dtype='float64'))

                # Tính trung bình
                average_dist = (dist_tran + dist_mat + dist_mui + dist_mieng)/4

                # So sánh tìm avarage_dist nho nhat
                if average_dist < min_average_dist:
                    min_average_dist = average_dist
                    final_filepath = filepath

    return min_average_dist, final_filepath


# Xử lý các request
@app.route("/", methods=['POST', 'GET'])
def home():
    if request.method == 'GET':
        return render_template("index.html")
    else:
        image_file = request.files['file']
        path_to_save = os.path.join(
            app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(path_to_save)

        # Đọc ảnh
        img = cv2.imread(path_to_save)

        w, h, d = img.shape

        # Nếu ảnh đọc vào không phải 128x128 thì resize ảnh về 128x128
        if w != 128 and h != 128:
            img = cv2.resize(
                img, (128, 128), interpolation=cv2.INTER_AREA)

        # Tính khoảng cách euclid nhỏ nhất và file có khoảng cách euclid nhỏ nhất với file đầu vào
        min_average_dist, final_filepath = findMinAverageDistance(img)

        img_pred = cv2.imread(final_filepath[0])

        img_pred_filename = final_filepath[0].split('\\')[-1]
        print(img_pred_filename)
        print(min_average_dist)

        # Save img_pre in static
        path_to_save_img_pred = os.path.join(
            app.config['UPLOAD_FOLDER'], img_pred_filename)
        cv2.imwrite(path_to_save_img_pred, img_pred)

        return render_template("index.html", user_image=image_file.filename,
                               img_pre_filename=img_pred_filename,
                               msg='Tai file thanh cong',
                               similarity=min_average_dist)

    return "Day la home"


# Start server
if __name__ == '__main__':

    app.run(host='localhost', port=9999, debug=True)
