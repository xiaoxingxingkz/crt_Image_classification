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






# 计算图像中蓝色区域和绿色区域像素面积
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

    # 计算蓝色和绿色区域的面积（像素）
    blue_area = cv2.countNonZero(blue_mask)
    green_area = cv2.countNonZero(green_mask)

    return blue_area, green_area

# 给图像打标签： 如果图像中绿色标记较少，表示药物在该肿瘤区域没有释放，认为无疗效，标记为1    
#               如果图像中绿色标记较多，表示药物在该肿瘤区域没有释放，认为有疗效，标记为0
def classify_image(blue_area, green_area, threshold):
    if green_area == 0:  # 避免除以零
        return 0.001  
    ratio = blue_area / green_area 
    return 1 if ratio >= threshold else 0

def process_images(input_dir, threshold = 1.0):
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)

            if image is not None:
                blue_area, green_area = calculate_color_area(image)
                label = classify_image(blue_area, green_area, threshold)
                new_filename = f"{label}_{filename}"
                # # 重命名文件
                os.rename(image_path, os.path.join(input_dir, new_filename))

            else:
                print(f"{filename} - 无法读取图像")

#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
# 调用处理函数，设置阈值，给图像打标签
process_images("label_image" , threshold=3.8)  # label_image：模型输入文件夹，threshold：阈值，可以根据需要调整阈值







# 划分训练集和测试集
def split_dataset(input_dir, train_dir, test_dir, train_ratio=0.8):
    #数据初始化，将文件夹清空，方便重复运行此代码
    if os.path.exists(train_dir):
        if len(os.listdir(train_dir)) != 0:
            shutil.rmtree(train_dir) 
    if os.path.exists(test_dir):
        if len(os.listdir(test_dir)) != 0:
            shutil.rmtree(test_dir) 

    # 确保训练集和测试集目录存在
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # 获取所有图片文件
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    # 随机打乱图片顺序
    random.shuffle(images)
    
    # 计算训练集的大小
    train_size = int(len(images) * train_ratio)
    
    # 分割训练集和测试集
    train_images = images[:train_size]
    test_images = images[train_size:]
    
    # 移动文件到对应的文件夹
    for image in train_images:
        shutil.move(os.path.join(input_dir, image), os.path.join(train_dir, image))
    for image in test_images:
        shutil.move(os.path.join(input_dir, image), os.path.join(test_dir, image))

    print(f"Split complete: {len(train_images)} images in training set, {len(test_images)} images in test set.")

#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
# 设置输入图像文件夹路径，以及训练集和测试集的输出文件夹路径
input_dir = os.path.join(cur_path, "label_image")  
train_dir = os.path.join(cur_path, "train")   
test_dir = os.path.join(cur_path, "test")     
# 划分数据集
split_dataset(input_dir, train_dir, test_dir)







# 特征和标签的提取
def extract_features_and_labels(image_folder):
    features = []
    labels = []

    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # 假设图像格式
            # 读取图像
            img_path = os.path.join(image_folder, filename)
            img = cv2.imread(img_path)

            # 特征提取（这里可以自定义特征提取方法）
            img = cv2.resize(img, (64, 64))  # 调整图像大小
            feature = img.flatten()  # 将图像展平为一维数组

            # 提取标签（文件名的第一个数字）
            label = int(filename.split('_')[0])  # 假设文件名格式为“1_xxx.jpg”

            features.append(feature)
            labels.append(label)

    return np.array(features), np.array(labels)

#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
# 读取训练集和测试集
train_folder = os.path.join(cur_path, "train")   
test_folder = os.path.join(cur_path, "test")   
  
X_train, y_train = extract_features_and_labels(train_folder)
X_test, y_test = extract_features_and_labels(test_folder)

# 训练SVM分类器
clf = svm.SVC(kernel='linear')  # 可以选择其他内核
clf.fit(X_train, y_train)

# 对测试集进行预测
y_pred = clf.predict(X_test)

# 输出结果
print("准确率:", accuracy_score(y_test, y_pred))
print("分类报告:\n", classification_report(y_test, y_pred))