import cv2

# 1.读取图片
image = cv2.imread("../images/face1.jpg")


def face_detect_demo():
    # 灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 创建人脸检测器
    face_cascade = cv2.CascadeClassifier("D:/soft/anaconda3/envs/OpenCV_Project/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml")
    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    # 绘制矩形框
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow("face_detect", image)


# 检测函数
face_detect_demo()



# 6.等待键盘输入
cv2.waitKey(0)
# 7.释放窗口
cv2.destroyAllWindows()
