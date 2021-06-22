import os
import cv2
# import sys
import numpy as np
import matplotlib.pyplot as plt


def calculateGradient(I):

    Dx = np.array([[-1, 0, 1]])
    Dy = np.array([[1, 0, -1]]).T

    # Đạo hàm theo hướng x
    Ix = np.zeros(I.shape)
    Ix[:, 1:-1] = -I[:, :-2] + I[:, 2:]
    Ix[:, 0] = -I[:, 0] + I[:, 1]
    Ix[:, -1] = -I[:, -2] + I[:, -1]

    # Đạo hàm theo hướng y
    Iy = np.zeros(I.shape)
    Iy[1:-1, :] = I[:-2, :] - I[2:, :]
    Iy[0, :] = I[0, :] - I[1, :]
    Iy[-1, :] = I[-2, :] - I[-1, :]

    return Ix, Iy


def magnitude_orientation(Ix, Iy):
    magnitude = np.sqrt(Ix**2 + Iy**2)
    orientation = np.arctan(Ix, Iy) * 180

    return magnitude, orientation


def normalizationBlock(feature_list):
    cnt = 0
    hog_feature = []
    cnt = 0
    for i in range(0, len(feature_list)-17):
        if (i+1) % 16 == 0:
            continue
        block_feature_1 = np.append(feature_list[i], feature_list[i+1])
        block_feature_2 = np.append(feature_list[i+16], feature_list[i+17])
        block_feature = np.append(block_feature_1, block_feature_2)

        # Chuẩn hóa theo chuẩn norm 2
        k = 0.0

        for j in range(0, len(block_feature)):
            k += (block_feature[j]**2)

        norm2 = np.sqrt(k)
        for j in range(0, len(block_feature)):
            block_feature[j] = block_feature[j]/norm2

        hog_feature += list(block_feature)

    hog_feature = np.array(hog_feature)

    return hog_feature


def calculateValueInBins(m, o, bins):
    # Dặt x là giá trị hướng
    # m là độ lớn
    x = o
    if x >= 180:
        x = x % 180
    if (x > 160 and x < 180):
        bins[0] += ((x - 160)/20) * m
        bin[int(160/20)] += ((180-x)/20) * m
    elif x % 20 == 0:
        bins[int(x/20)] += m
    else:
        # Tìm 2 bins có nhãn gần nhất với x và
        # chia giá trị độ lớn cho 2 bins đó
        for x1 in range(0, 160, 20):
            if x < x1:
                bins[int((x1-20)/20)] += ((x1-x)/20) * m
                bins[int(x1/20)] += ((x - (x1 - 20))/20) * m
                break


def createBins(magnitude, orientation):
    bins = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for i in range(0, 8):
        for j in range(0, 8):
            calculateValueInBins(magnitude[i][j], orientation[i][j], bins)
    return bins


def drawHistogram(bins):
    X = [0, 20, 40, 60, 80, 100, 120, 140, 160]
    data = bins
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(X, bins, width=0.25)
    plt.show()


def preprocessImage():
    root = "test_data"
    for folder in os.listdir(root):
        for file in os.listdir(os.path.join(root, folder)):
            img = cv2.imread(os.path.join(root, folder, file))
            print(img.shape)
            w, h, d = img.shape
            if w == 128 and h == 128:
                continue
            else:
                resized = cv2.resize(
                    img, (128, 128), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join("resized", folder, file), resized)


def hog_feature(I):
    feature_list = []
    h, w = I.shape
    cnt = 0
    for i in range(8, h+1, 8):
        for j in range(8, w+1, 8):
            cell = I[i-8:i, j-8:j]
            Ix, Iy = calculateGradient(cell)
            magnitude, orientation = magnitude_orientation(Ix, Iy)

            bins = createBins(magnitude, orientation)
            feature_list.append(bins)
    hog_feature = normalizationBlock(feature_list)
    return hog_feature


def extract_hog_feature(img):
    # Chuyển đổi ảnh về mức xám
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Trích xuất đặc trưng vùng trán
    I = gray[0:32, 0:128]
    hog_feature_tran = hog_feature(I)

    # Trích xuất đậc trưng vùng mắt
    I = gray[32:64, 0:128]
    hog_feature_mat = hog_feature(I)
    print(hog_feature_mat)

    # Trích xuất đặc trưng vùng mũi
    I = gray[64:88, 0:128]
    hog_feature_mui = hog_feature(I)

    # Trích xuất đặc trưng vùng miệng
    I = gray[88:128, 0:128]
    hog_feature_mieng = hog_feature(I)

    return hog_feature_tran, hog_feature_mat, hog_feature_mui, hog_feature_mieng
