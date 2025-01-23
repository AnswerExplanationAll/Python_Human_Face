import cv2

# 1.读取图片
image = cv2.imread("../images/face1.jpg")
print("原图：", image.shape)


# 3.修改尺寸
image_resize = cv2.resize(image, (1000, 600))

# 4.绘制矩形
# 4.1 坐标
x, y, w, h = 100, 100, 100, 100
# 4.2 矩形颜色
color = (0, 0, 255)
# 4.3 矩形线条粗细
thickness = 2
# 4.4 绘制矩形
cv2.rectangle(image_resize, (x, y), (x + w, y + h), color, thickness)
# 4.5 绘制圆形
cv2.circle(image_resize, (x+50, y+50), 50, color, thickness)

# 5.显示图片
cv2.imshow("resized", image_resize)

# 6.保存图片
# cv2.imwrite("face1_resize.jpg", image_resize)

# 6.等待键盘输入
cv2.waitKey(0)
# 7.释放窗口
cv2.destroyAllWindows()
