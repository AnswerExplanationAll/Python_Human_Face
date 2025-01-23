import os

import cv2
import numpy as np
from PIL import Image


def getImageAndLabels(path):
    # 存储人脸数据
    facesSamples = []
    # 存储用户id
    ids = []
    # 存储图片信息
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # 加载分类器
    faceCascade = cv2.CascadeClassifier(
        "D:/soft/anaconda3/envs/OpenCV_Project/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml")
    # 遍历列表中的文件
    for imagePath in imagePaths:
        # 读取图片 灰度化PIL有九种方式“：L;LA;P;RGB;RGBA;CMYK;YCbCr;I;F”
        PIL_img = Image.open(imagePath).convert('L')
        # 将图片转化为数组，以黑白深浅
        img_numpy = np.array(PIL_img, 'uint8')
        # 获取图片人脸区域
        faces = faceCascade.detectMultiScale(img_numpy)
        # 获取每张图片的用户id
        id = int(os.path.split(imagePath)[1].split('.')[0])
        # 遍历每张图片的所有人脸区域
        for (x, y, w, h) in faces:
            # 将每张图片的所有人脸区域存入列表
            facesSamples.append(img_numpy[y:y + h, x:x + w])
            # 将每张图片的所有用户id存入列表
            ids.append(id)
    # 打印存储的所有人脸区域和用户id
    print('ids=', ids)
    print('facesSamples=', facesSamples)
    # 返回人脸区域和用户id
    return facesSamples, ids


if __name__ == '__main__':
    # 图片路径
    path = '../data'
    # 获取人脸区域和用户id
    faces, ids = getImageAndLabels(path)
    # 创建LBPH识别器
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # 训练
    recognizer.train(faces, np.array(ids))
    # 保存模型
    recognizer.write('../trainer/trainer.yml')
    # 打印训练数据
    print('训练数据完成')
