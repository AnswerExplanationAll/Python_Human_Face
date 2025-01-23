import urllib  # 导入urllib模块，用于处理URL
import urllib.request  # 导入urllib.request模块，用于打开和读取URLs

import cv2  # 导入OpenCV库，用于图像处理和计算机视觉任务

# 加载训练数据集文件
face_recognizer = cv2.face.LBPHFaceRecognizer_create()  # 创建一个LBPH人脸识别器对象
# 加载数据
face_recognizer.read('../trainer/trainer.yml')  # 从指定的YML文件中读取训练好的人脸识别模型
# 名称
names = []  # 初始化一个空列表，用于存储识别到的人脸对应的名字
# 警报全局变量
warningtime = 0  # 初始化一个全局变量，用于记录连续未识别到已知人脸的次数


# md5加密函数
def md5(str):
    import hashlib  # 导入hashlib库，用于生成哈希值
    md5_obj = hashlib.md5()  # 创建一个md5哈希对象
    md5_obj.update(str.encode('utf-8'))  # 将输入的字符串编码为UTF-8并更新到哈希对象中
    return md5_obj.hexdigest()  # 返回哈希对象的十六进制摘要


# 短信反馈状态码对应的字符串
statusStr = {
    '0': '短信发送成功',
    '-1': '参数不全',
    '-2': '服务器空间不足',
    '30': '密码错误',
    '40': '账号不存在',
    '41': '余额不足',
    '42': '帐户已过期',
    '43': 'IP地址限制',
    '50': '内容含有敏感词'
}


# 报警模块函数
def warning():
    smsapi = "https://api.smsbao.com/"  # 短信API的URL
    # 短信平台账号（这里用掩码隐藏了真实号码）
    user = '199****3693'
    # 短信平台接口秘钥（通过md5加密）
    password = md5('********')
    # 要发送的短信内容
    content = '【报警】\n您的门禁系统检测到陌生人入侵，请及时查看！'
    # 要发送的手机号码（同样用掩码隐藏了真实号码）
    phone = '199****3693'

    # 构造发送短信的URL请求
    data = urllib.parse.urlencode({'u': user, 'p': password, 'm': phone, 'c': content})
    send_url = smsapi + 'sms?' + data
    # 打开并读取URL响应
    response = urllib.request.urlopen(send_url)
    the_page = response.read().decode('utf-8')
    # 打印短信发送状态
    print(statusStr[the_page])


# 准备识别的图片函数
def face_detect_img(img):
    # 将图片转换为灰度图，因为人脸识别通常在灰度图上进行
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 加载Haar特征的人脸检测分类器
    face_cascade = cv2.CascadeClassifier(
        'D:/soft/anaconda3/envs/OpenCV_Project/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
    # 检测图片中的人脸
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (100, 100), (300, 300))
    for (x, y, w, h) in faces:
        # 在检测到的人脸周围绘制矩形框
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # 在人脸中心绘制圆点
        cv2.circle(img, (x + w // 2, y + h // 2), min(w, h) // 2, (0, 255, 0), 2)
        # 人脸识别
        ids, conf = face_recognizer.predict(gray[y:y + h, x:x + w])
        if conf > 80:  # 如果置信度大于80，认为是没有识别到已知人脸
            global warningtime  # 使用全局变量记录未识别次数
            warningtime += 1
            if warningtime > 100:  # 如果连续未识别次数超过100次，则发送报警短信
                warning()
                warningtime = 0  # 重置未识别次数
            # 在图片上标注"unkow"（应为"unknown"，此处可能是拼写错误）
            cv2.putText(img, 'unkow', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:  # 如果置信度小于或等于80，认为是识别到了已知人脸
            # 在图片上标注识别到的人脸对应的名字
            cv2.putText(img, names[ids - 1], (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # 显示处理后的图片
    cv2.imshow('result', img)


# 名字标签加载函数
def name():
    global names  # 使用全局变量names存储从文件中读取的名字列表
    # 假设姓名列表保存在 names.txt 文件中，每行一个姓名
    with open('../data/names.txt', 'r', encoding='utf-8') as file:
        # 读取文件中的所有行，去除每行末尾的换行符，并将结果存储在names列表中
        names = [line.strip() for line in file.readlines()]


# 加载视频
cap = cv2.VideoCapture(0)  # 打开默认摄像头（通常是笔记本电脑的内置摄像头）
name()  # 调用name函数加载名字列表
while True:  # 进入一个无限循环，直到用户按下'q'键退出
    # 读取一帧视频
    ret, img = cap.read()
    if not ret:  # 如果读取帧失败（可能是视频结束了）
        break
    # 调用face_detect_img函数进行人脸识别和处理
    face_detect_img(img)
    # 如果用户按下'q'键，则退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源
cap.release()
# 销毁所有OpenCV窗口
cv2.destroyAllWindows()
