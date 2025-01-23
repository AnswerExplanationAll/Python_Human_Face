import cv2

# 1.读取图片
image = cv2.imread("../images/face1.jpg")
print("原图：", image.shape)

# 2.灰度转换
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3.修改尺寸
image_resize = cv2.resize(gray, (1000, 600))
print("修改尺寸后：", image_resize.shape)

# 4.显示图片
cv2.imshow("resized", image_resize)

# 5.保存图片
cv2.imwrite("face1_resize.jpg", image_resize)

# 6.等待键盘输入
cv2.waitKey(0)
# 7.释放窗口
cv2.destroyAllWindows()
