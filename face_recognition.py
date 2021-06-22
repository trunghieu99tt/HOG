import os
import cv2
from hog_extraction import extract_hog_feature
import csv


def main():
    root = "face_data"
    with open("hog_feature.csv", 'w', encoding='UTF8', newline='') as f:
        # Duyệt toàn bộ các file data mẫu
        for file in os.listdir(root):
            img = cv2.imread(os.path.join(root, file))

            # Lấy ra feature cho từng phân vùng: trán, mắt, mũi, miệng
            hog_feature_tran, hog_feature_mat, hog_feature_mui, hog_feature_mieng = extract_hog_feature(
                img)

            # Ghi vào file csv
            #  Write to .csv file
            filepath = [os.path.join(root, file)]

            writer = csv.writer(f)
            writer.writerow(filepath)
            writer.writerow(hog_feature_tran)
            writer.writerow(hog_feature_mat)
            writer.writerow(hog_feature_mui)
            writer.writerow(hog_feature_mieng)


if __name__ == "__main__":
    main()
