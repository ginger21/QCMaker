
# 图像采集程序

import cv2
import os
import RPi.GPIO as GPIO
import time

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
    #init()
    GPIO.output(IN1,GPIO.HIGH)
    GPIO.output(IN2,GPIO.LOW)
    GPIO.output(IN3,GPIO.HIGH)
    GPIO.output(IN4,GPIO.LOW)
    time.sleep(sleep_time)
    GPIO.output(IN1,False)
    GPIO.output(IN2,False)
    GPIO.output(IN3,False)
    GPIO.output(IN4,False) 

# 停止
def stop():
    #init()
    GPIO.output(IN1,False)
    GPIO.output(IN2,False)
    GPIO.output(IN3,False)
    GPIO.output(IN4,False)     

# 左转   
def left(sleep_time):
    #init()
    GPIO.output(IN1,False)
    GPIO.output(IN2,False)
    GPIO.output(IN3,GPIO.HIGH)
    GPIO.output(IN4,GPIO.LOW)        
    time.sleep(sleep_time)
    GPIO.output(IN1,False)
    GPIO.output(IN2,False)
    GPIO.output(IN3,False)
    GPIO.output(IN4,False) 

# 右转
def right(sleep_time):
    #init()
    GPIO.output(IN1,GPIO.HIGH)
    GPIO.output(IN2,GPIO.LOW)
    GPIO.output(IN3,False)
    GPIO.output(IN4,False) 
    time.sleep(sleep_time)
    GPIO.output(IN1,False)
    GPIO.output(IN2,False)
    GPIO.output(IN3,False)
    GPIO.output(IN4,False)

#=============== 定义采集图像处理函数 =======================#    
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
    cv2.rectangle(frame, (x0, y0), (x0+width, y0+height), (0, 255, 0))
    
    # 提取ROI像素
    # 矩形框选 frame[y:y+height,x:x+width] x,y 原点坐标  y+height,x+width 矩形对角坐标
    roi = frame[y0:y0+height, x0:x0+width] 
    
    # 将采集图像缩小到80x60大小，interpolation是插值方法参量
    roi = cv2.resize(roi,(80,60),interpolation=cv2.INTER_CUBIC)
    
    # cv2.cvtColor(input_image , flag),flag是转换类型：cv2.COLOR_BGR2GRAY,cv2.COLOR_BGR2HSV
    # cv2.COLOR_BGR2GRAY转换为灰度图像
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 高斯模糊 斯模糊本质上是低通滤波器，输出图像的每个像素点是原图像上对应像素点与周围像素点的加权和
    # (3, 3)表示高斯矩阵的长与宽都是3，标准差取0时OpenCV会根据高斯矩阵的尺寸自己计算。
    # 概括地讲，高斯矩阵的尺寸越大，标准差越大，处理过的图像模糊程度越大
    # https://blog.csdn.net/sunny2038/article/details/9155893
    blur = cv2.GaussianBlur(gray, (3,3),3) # 高斯模糊，给出高斯模糊矩阵和标准差
    
    # https://blog.csdn.net/on2way/article/details/46812121
    # 二值化图像选取简单阈值当然是最简单，选取一个全局阈值，然后就把整幅图像分成了非黑即白的二值图像了。函数为cv2.threshold() 
    # 这个函数有四个参数，第一个原图像，第二个进行分类的阈值，第三个是高于（低于）阈值时赋予的新值，第四个是一个方法选择参数，常用的有： 
    # cv2.THRESH_BINARY（黑白二值） 
    # cv2.THRESH_BINARY_INV（黑白二值反转） 
    # cv2.THRESH_TRUNC （得到的图像为多像素值） 
    # cv2.THRESH_TOZERO 
    # cv2.THRESH_TOZERO_INV 
    # cv2.THRESH_OTSU
    #该函数有两个返回值，第一个retVal（得到的阈值值（在后面一个方法中会用到）），第二个就是阈值化后的图像。
    ret, res = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # ret还是bool类型
        
    return res

#======================= 设置初始参数 ===================#
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

# 录制的样本数
num=1

init()

#============================= 开始采集 ========================#
# 打开摄像头，0为USB摄像头
cap = cv2.VideoCapture(0)

cap.set(3,320) #设置分辨率
cap.set(4,240)

# wile(True) 保持循环，不停采集
while (True):

    # 返回的第一个参数为bool类型，用来表示是否读取到帧，如果为False说明已经读到最后一帧。
    # frame为读取到的帧图片
    ret,frame = cap.read()
    
    # https://blog.csdn.net/jningwei/article/details/78753607
    # 图像水平翻转
    frame = cv2.flip(frame,1)
    
    # 显示ROI区域
    # 调用函数binaryMask处理采集的帧图像
    roi = binaryMask(frame, x0, y0, width, height)

    # 窗口显示，显示名为 Capture   
    cv2.imshow("capture",frame)
    cv2.imshow("roi",roi)
    
    # 等待键盘输入，
    key = cv2.waitKey(1) & 0xFF
    
    # 若检测到按键 ‘q’，退出
    if key == ord('q'):  
        print("quit")
        break
    
    # 若检测到按键‘t’，在文件目录下创建forward，stop，right，left，turn_left，turn_right, walk文件夹
    # 采集开始
    elif key == ord('t'):
        os.makedirs('./forward')
        os.makedirs('./stop')
        os.makedirs('./right')
        os.makedirs('./left')
        os.makedirs('./turn_left')
        os.makedirs('./turn_right')
        os.makedirs('./walk')
        num=1
        print('start')
    
    # 按键‘w’前进0.05秒，并采集图像，表示前进时的图像
    # num += 1 对计数加一，以下同理
    if key == ord('w'):
        print("Forward")
        cv2.imwrite('./forward/'+str(num)+'.jpg',roi)
        forward(0.05)
        num += 1
    # 按键‘a’左转0.05秒，并采集图像，表示左转时的图像
    elif key == ord('a'):
        print("Left")
        cv2.imwrite('./left/'+str(num)+'.jpg',roi)
        left(0.05)
        num += 1
    # 按键‘s’汽车停止，采集停止图标图像
    elif key == ord('s'):
        print("Stop")
        cv2.imwrite('./stop/'+str(num)+'.jpg',roi)
        stop()
        num +=1
    # 按键‘d’右转0.05秒，并采集图像，表示右转时的图像
    elif key == ord('d'):
        print("Right")
        cv2.imwrite('./right/'+str(num)+'.jpg',roi)
        right(0.05)
        num += 1
    # 按键‘j’采集左转路牌图像    
    elif key == ord('j'):
        print("turn_left")
        cv2.imwrite('./turn_left/'+str(num)+'.jpg',roi)
        num += 1
    # 按键‘k’采集右转路牌图像
    elif key == ord('k'):
        print("turn_right")
        cv2.imwrite('./turn_right/'+str(num)+'.jpg',roi)
        num += 1
    elif key == ord('i'):
        print("walk")
        cv2.imwrite('./walk/'+str(num)+'.jpg',roi)
        num += 1

# 电机状态清零
GPIO.cleanup()

# 释放图像窗口
cap.release()
cv2.destroyAllWindows()
    