import cv2
import face_recognition
import sys

'''
# PIL做图比较复杂，不使用PIL库
from PIL import Image
ret = Image.open('similar_two_face.jpg')
print(ret)
'''
'''
ret:
<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=440x571 at 0x7F0D5F9FD810>
<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=440x571 at 0x7F0D7D201390>
'''
# 1.读取图像内容
face_image = face_recognition.load_image_file(file='images/similar_three_faces.jpg')  # 可以读出来每wt一个像素点的值
# print(face_image)
# 2.进行特征提取向量化，用数据展示五官特征
face_encodings = face_recognition.face_encodings(face_image=face_image)  # 两个人脸的128维度人脸编码
# print(face_encodings)  # 两个array表示两张人脸编码，放在一个列表中
# print(len(face_encodings))
if len(face_encodings) > 2:
    print("超过两张脸，该程序暂不支持多张人脸的比对！")
    sys.exit()
# 3.获取两张人脸坐标
face_locations = face_recognition.face_locations(img=face_image)  # 获取人脸的坐标
# print(face_locations)  # 一般确定左上角和右下角就可以确定一张脸的位置，[(64, 201, 219, 46), (67, 339, 196, 210)]
# 4.分别获取两张脸的人脸编码
face01 = face_encodings[0]
face02 = face_encodings[1]
# print('face01：', face01)  # list类型，[ 1.82563066e-02  2.23452318e-02 -4.11391482e-02 -8.67729932e-02，……]
# print('face02：', face02)
# 5.比较这两张脸
ret = face_recognition.compare_faces(known_face_encodings=[face01], face_encoding_to_check=face02,
                                     tolerance=0.35)  # ret： <class 'list'>；tolerance越小容忍度越小
print(ret)
if ret == [True]:
    print("识别结果是同一个人！")
    flag = "Yes!"
else:
    print("识别结果不是同一个人!")
    flag = "No"
# 6.用opencv打开图像并显示窗口界面
for i in range(len(face_encodings)):
    # 从后向前取
    # face_encoding = face_encodings[i - 1]
    # face_locations = face_locations[i - 1]
    # 从前向后取
    face_encoding = face_encodings[i]
    face_location = face_locations[i]
    top, right, bottom, left = face_location  # 方便使用opencv画框
    # print(top, right, bottom, left)  # 上、右、下、左
    '''
    64 201 219 46
    67 339 196 210
    '''
    # 7.画框
    cv2.rectangle(img=face_image, pt1=(left, top), pt2=(right, bottom), color=(0, 255, 0),
                  thickness=1)  # thickness:粗细程度
    # 8.写字
    cv2.putText(img=face_image, text=flag, org=(left - 10, top - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                color=(255, 255, 0), thickness=2)
# 9.渲染新的图像
face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
# 10.显示图像
cv2.imshow('demo', face_image_rgb)
# 11.关闭的命令，点击x才会关闭
cv2.waitKey(0)
