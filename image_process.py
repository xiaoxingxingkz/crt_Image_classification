import os
import numpy as np
import cv2
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import random
import shutil
from pathlib import Path

cur_path = Path(__file__).parent
cur_path = str(cur_path)


def image_clean(input_dir, output_dir):
    # 目标文件夹存在，进行删除，重复代码时需要
    if os.path.exists(output_dir):
        if len(os.listdir(output_dir)) != 0:
            shutil.rmtree(output_dir) 
    i = 1
    for filename in os.listdir(input_dir):
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        new_filename = str(i) + ".jpg"
        image_path_save = os.path.join(output_dir, new_filename)
        cv2.imwrite(image_path_save, image)
        i += 1

#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#数据初始化，将原始图像写入到新的文件夹，防止对原始图像产生不可逆的更改
image_clean("data", "label_image") #data:模型输入文件夹；label_image：输出文件夹




# 将图像中蓝色区域和绿色区域分割开
def calculate_color_area(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 定义蓝色和绿色的 HSV 范围
    blue_lower = np.array([100, 150, 0])
    blue_upper = np.array([140, 255, 255])
    
    green_lower = np.array([40, 40, 40])
    green_upper = np.array([80, 255, 255])

    # 创建掩膜
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    return blue_mask, green_mask



def process_images(input_dir, output_image):
    # 目标文件夹存在，进行删除，重复代码时需要
    if os.path.exists(output_image):
        if len(os.listdir(output_image)) != 0:
            shutil.rmtree(output_image) 

    if not os.path.exists(output_image):
        os.makedirs(output_image)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)

            image_path_save = os.path.join(output_image, filename)
            cv2.imwrite(image_path_save, image)

            blue_mask, green_mask = calculate_color_area(image)

            merged_blue_mask = cv2.merge([blue_mask, blue_mask, blue_mask])
            image_b = cv2.addWeighted(image, 0.5, merged_blue_mask, 0.5, 0)
            new_filename2 = filename.split(".")[0] + "_blue.jpg"
            image_path_save2 = os.path.join(output_image, new_filename2)
            cv2.imwrite(image_path_save2, image_b)

            merged_green_mask = cv2.merge([green_mask, green_mask, green_mask])
            image_g = cv2.addWeighted(image, 0.5, merged_green_mask, 0.5, 0)
            new_filename3 = filename.split(".")[0] + "_green.jpg"
            image_path_save3 = os.path.join(output_image, new_filename3)
            cv2.imwrite(image_path_save3, image_g)            


#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#数据初始化，将原始图像写入到新的文件夹，防止对原始图像产生不可逆的更改
process_images("label_image" , "output_image") #label_image:模型输入文件夹；output_image：输出文件夹







