import os

import cv2

# 确保保存图像的目录存在
save_dir = '../images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

num = 1
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow('frame', frame)

        # 等待键盘输入，避免多次调用 waitKey
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            file_path = os.path.join(save_dir, f"name{num}.jpg")
            cv2.imwrite(file_path, frame)
            print(f"Success to save picture {file_path}")
            num += 1
        elif key == ord('q'):
            break
finally:
    # 释放摄像头和所有窗口
    cap.release()
    cv2.destroyAllWindows()
