#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
摄像头采集图片验证
author:Administrator
datetime:2018/3/25/025 9:27
software: PyCharm
'''
import cv2
import RPi.GPIO as GPIO
import time
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
from keras.optimizers import SGD


#============= 定义电机驱动动作：前进、左右转==========#
# 初始化
def init():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(IN1,GPIO.OUT)
    GPIO.setup(IN2,GPIO.OUT)
    GPIO.setup(IN3,GPIO.OUT)
    GPIO.setup(IN4,GPIO.OUT)

# 前进
def forward(sleep_time):
    GPIO.output(IN1,GPIO.HIGH)
    GPIO.output(IN2,GPIO.LOW)
    GPIO.output(IN3,GPIO.HIGH)
    GPIO.output(IN4,GPIO.LOW)
    time.sleep(sleep_time)
# 左转    
def left(sleep_time):
    GPIO.output(IN1,False)
    GPIO.output(IN2,False)
    GPIO.output(IN3,GPIO.HIGH)
    GPIO.output(IN4,GPIO.LOW)
    time.sleep(sleep_time)
# 右转
def right(sleep_time):
    GPIO.output(IN1,GPIO.HIGH)
    GPIO.output(IN2,GPIO.LOW)
    GPIO.output(IN3,False)
    GPIO.output(IN4,False)
    time.sleep(sleep_time)
# 停止
def stop(sleep_time):
    GPIO.output(IN1,False)
    GPIO.output(IN2,False)
    GPIO.output(IN3,False)
    GPIO.output(IN4,False)
    time.sleep(sleep_time)

#================= 定义图片处理函数 ====================#    
# 图片尺寸
img_width, img_height = 80, 60
input_shape = (img_width, img_height, 1)


# 获读取集的图片并进行预处理

def img_pre(img_path='./temp.png'):
    # 读取图片，地址为文件夹中，文件名temp.png
    img = image.load_img(img_path, grayscale=True, target_size=(80, 60))
    
    # 将图像文件转化为数组形式
    x = image.img_to_array(img)

    # 将二维数组（width，height）转化为（batch，width，height）三维模式
    # 符合模型的输入格式
    x = np.expand_dims(x, axis=0)  

    # 将数组归一化（图像rgb和灰度表示数值在0-255之间，归一化将值转化为0-1之间） 
    x /= 255
    return x

# 采集图片并进行处理
def binaryMask(frame, x0, y0, width, height):
    # 显示方框
    '''
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    参数解释
    第一个参数：img是原图
    第二个参数：（x，y）是矩阵的左上点坐标
    第三个参数：（x+w，y+h）是矩阵的右下点坐标
    第四个参数：（0,255,0）是画线对应的rgb颜色
    第五个参数：2是所画的线的宽度
    '''
  
    # 提取ROI像素
    # 矩形框选 frame[y:y+height,x:x+width] x,y 原点坐标  y+height,x+width 矩形对角坐标
    roi = frame[y0:y0+height, x0:x0+width] 
    
    # 将采集图像缩小到80x60大小，interpolation是插值方法参量
    roi = cv2.resize(roi,(80,60),interpolation=cv2.INTER_CUBIC)
    
    # cv2.cvtColor(input_image , flag),flag是转换类型：cv2.COLOR_BGR2GRAY,cv2.COLOR_BGR2HSV
    # cv2.COLOR_BGR2GRAY转换为灰度图像
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 高斯模糊 斯模糊本质上是低通滤波器，输出图像的每个像素点是原图像上对应像素点与周围像素点的加权和
    # (3, 3)表示高斯矩阵的长与宽都是3，标准差取0时OpenCV会根据高斯矩阵的尺寸自己计算。概括地讲，高斯矩阵的尺寸越大，标准差越大，处理过的图像模糊程度越大
    # https://blog.csdn.net/sunny2038/article/details/9155893
    blur = cv2.GaussianBlur(gray, (3,3),3) # 高斯模糊，给出高斯模糊矩阵和标准差
    
    # https://blog.csdn.net/on2way/article/details/46812121    
    #二值化选取简单阈值当然是最简单，选取一个全局阈值，然后就把整幅图像分成了非黑即白的二值图像了。函数为cv2.threshold() 
    #这个函数有四个参数，第一个原图像，第二个进行分类的阈值，第三个是高于（低于）阈值时赋予的新值，第四个是一个方法选择参数，常用的有： 
    #cv2.THRESH_BINARY（黑白二值） 
    #cv2.THRESH_BINARY_INV（黑白二值反转） 
    #cv2.THRESH_TRUNC （得到的图像为多像素值） 
    #cv2.THRESH_TOZERO 
    #cv2.THRESH_TOZERO_INV
    # # cv2.THRESH_OTSU 
    #该函数有两个返回值，第一个retVal（得到的阈值值（在后面一个方法中会用到）），第二个就是阈值化后的图像。
    ret, res = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # ret还是bool类型
    
    return res

#============================ 创建训练模型 ====================================#
# sequential表示贯序模型：多个网络层的线性堆叠
model = Sequential()

# 二维卷积层函数：
# Conv2D(filters, 
#        kernel_size, 
#        activation=None)
# 参数：
#   filters：卷积核的数目（即输出的维度）
#   kernel_size：单个整数或由两个整数构成的list/tuple，卷积核的宽度和长度。
#   activation：激活函数，为预定义的激活函数名，常用’relu‘
# 当使用该层作为第一层时，应提供input_shape参数
# model.add()加入指定参数的层
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))

# 为空域信号施加最大值池化
# MaxPooling2D(pool_size)
# 参数：
#   pool_size：整数或长为2的整数tuple，代表在两个方向（竖直，水平）上的下采样因子，
#              如取（2，2）将使图片在两个维度上均变为原长的一半。
#              为整数意为各个维度值相同且为该数字。
model.add(MaxPooling2D(pool_size=(2, 2)))

# 随机断开输入神经元
# Dropout(rate)
# 参数：
#   rate：0~1的浮点数，控制需要断开的神经元的比例
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

# 为空域信号施加全局平均值池化
model.add(GlobalAveragePooling2D())

model.add(Dropout(0.5))

# 全连接层
# Dense(units, activation)
# 参数：units：大于0的整数，代表该层的输出维度。
#      activation：激活函数，为预定义的激活函数名   
model.add(Dense(7, activation='softmax'))

# 选取随机梯度下降法SGD作为优化方法
# 参数：
#    lr：大或等于0的浮点数，学习率
#    momentum：大或等于0的浮点数，动量参数
#    decay：大或等于0的浮点数，每次更新后的学习率衰减值
#    nesterov：布尔值，确定是否使用Nesterov动量
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# model.compile 编译模型
# 参数：
#   loss：损失函数
#   optimizer：优化器
#   metrics：性能评估函数
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# 载入训练好的权重
model.load_weights('./weights/weights.h5')

#================== 打开摄像头开始运行 =====================#
# 打开usb摄像头
cap = cv2.VideoCapture(0)  

#设置分辨率
cap.set(3,320) 
cap.set(4,240)

num = ''

#motor_pin
IN1 = 11
IN2 = 12
IN3 = 13
IN4 = 15

# ROI框的显示位置
x0 = 0
y0 = 0

# 录制的手势图片大小
width = 320
height = 240

# 初始化电机
init()

# 动作分类
classes = ['forward', 'left', 'right', 
            'stop', 'turn_left', 'turn_right', 'walk']

# wile(True) 保持循环，不停采集
while True:
    
    # 读取一帧图片
    ret, frame = cap.read()  
    
    # 图像水平翻转
    frame = cv2.flip(frame,1)

    # 调用函数binaryMask处理采集的帧图像    
    roi = binaryMask(frame, x0, y0, width, height)

    # 将采集处理后的图片保存到本地
    cv2.imwrite('temp.png', roi)
    
    # 对采集的图像进行预处理
    x = img_pre()

    # model.predict_classes(x)进行预测,返回分类结果
    classes = model.predict_classes(x)[0]
    num = str(classes)
    print(num)

    # 根据返回的分类采取行动
    key = cv2.waitKey(1) & 0xFF
    
    # num为0，前进
    if num == '0':
        forward(0.05)
    # num为1，左转0.05秒
    elif num == '1':
        left(0.05)
    
    # num为2，右转0.05秒
    elif num == '2':
        right(0.05)
    
    # num为3，停止2秒
    elif num == '3':
        stop(2)
    
    # num为4，左转0.5秒
    elif num == '4':
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        roi = binaryMask(frame, x0, y0, width, height)
        cv2.imwrite('temp.png', roi)
        x = img_pre()
        classes = model.predict_classes(x)[0]
        num = str(classes)
        if num == '4':
            left(0.5)

    # num为5，右转1秒
    elif num == '5':
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        roi = binaryMask(frame, x0, y0, width, height)
        cv2.imwrite('temp.png', roi)
        x = img_pre()
        classes = model.predict_classes(x)[0]
        num = str(classes)
        if num == '5':
            right(0.5)

    # num为6，停止3秒再继续前进
    elif num == '6':
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        roi = binaryMask(frame, x0, y0, width, height)
        cv2.imwrite('temp.png', roi)
        x = img_pre()
        classes = model.predict_classes(x)[0]
        num = str(classes)
        if num == '6':
            stop(3)
            forward(1)
    
    # 按‘q’退出
    if key == ord('q'):
        break
        
    # 显示图片
    cv2.imshow('roi', roi)  # 显示图片

# 退出后停止电机    
stop(1)

# 释放显示窗口
cv2.destroyAllWindows()
cap.release()