import cv2


def face_detect_demo(image):
    # 灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 创建人脸检测器
    face_cascade = cv2.CascadeClassifier(
        "D:/soft/anaconda3/envs/OpenCV_Project/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml")
    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    # 绘制矩形框
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow("face_detect", image)


# 读取摄像头
cap = cv2.VideoCapture(0)  # 0表示电脑摄像头，1表示外接摄像头
while True:
    # 读取摄像头
    ret, frame = cap.read()
    # 检测人脸
    face_detect_demo(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6.等待键盘输入
cv2.waitKey(0)
# 5.释放摄像头
cap.release()

# 7.释放内存
cv2.destroyAllWindows()
