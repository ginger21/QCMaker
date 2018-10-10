import os
import cv2
import sys
import serial
import time
import numpy as np
from collections import Counter
from creat_model import creat_model
import serial.tools.list_ports

### 获取输入串口号 ###
## port list 形如：
# ['COM8',
# '蓝牙链接上的标准串行 (COM8)',
# 'BTHENUM\\{00001101-0000-1000-8000-00805F9B34FB}_LOCALMFG&0000\\8&15A53DA3&0&000000000000_0000001A']
port_list = list(serial.tools.list_ports.comports())
for i in range(len(port_list)):
    com_id = list(port_list[i])[0]
    name = list(port_list[i])[1]
    hwid = list(port_list[i])[2]
    status = hwid.split('\\')
    if name[:2] == '蓝牙' and status[1][-1] != '0':
        com = com_id

# 当前文件夹路径
root_path = sys.path[0]

### creat model ###
height = 48
width = 48
channels = 1
input_shape = [height, width, channels]

model = creat_model(input_shape, num_classes = 3)

### load model trained ###
path = os.path.join(root_path, 'weights/weights.h5')
model.load_weights(path)
emo_classes = ['Angry', 'Happy', 'Neutral']

## def prepare pic ###
def prepare_image(image, target_width = 48, target_height = 48, max_zoom = 0.2):
    """Zooms and crops the image randomly for data augmentation."""

    # First, let's find the largest bounding box with the target size ratio that fits within the image
    height = image.shape[0]
    width = image.shape[1]
    image_ratio = width / height
    target_image_ratio = target_width / target_height
    crop_vertically = image_ratio < target_image_ratio
    crop_width = width if crop_vertically else int(height * target_image_ratio)
    crop_height = int(width / target_image_ratio) if crop_vertically else height
        
    # Now let's shrink this bounding box by a random factor (dividing the dimensions by a random number
    # between 1.0 and 1.0 + `max_zoom`.
    resize_factor = np.random.rand() * max_zoom + 1.0
    crop_width = int(crop_width / resize_factor)
    crop_height = int(crop_height / resize_factor)
    
    # Next, we can select a random location on the image for this bounding box.
    x0 = np.random.randint(0, width - crop_width)
    y0 = np.random.randint(0, height - crop_height)
    x1 = x0 + crop_width
    y1 = y0 + crop_height
    
    # Let's crop the image using the random bounding box we built.
    image = image[y0:y1, x0:x1]

    # Let's also flip the image horizontally with 50% probability:
    # if np.random.rand() < 0.5:
    #    image = np.fliplr(image)

    # Now, let's resize the image to the target dimensions.
    # image = resize(image, (target_width, target_height), preserve_range = True)
    image = cv2.resize(image,(48,48),interpolation=cv2.INTER_CUBIC)
    image = image.astype(np.float32) / 255
    # Finally, let's ensure that the colors are represented as
    # 32-bit floats ranging from 0.0 to 1.0 (for now):
    return image

### 打开串口 #####
ser = serial.Serial(com, 9600)

### cap pic and predict ###
# ROI框的显示位置
x0 = 100
y0 = 50

# 录制的图片大小
width = 120
height = 140

cap = cv2.VideoCapture(0)
cap.set(3,320)
cap.set(4,240)

time_sta = 0

title = 'None'

while(1):
    # get a frame
    ret, frame = cap.read()
    cv2.rectangle(frame, (x0, y0), (x0+width, y0+height), (0, 255, 0))
    # show a frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, title, (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("capture", frame)

    roi = frame[y0:y0+height, x0:x0+width]


    
    localtime = time.localtime(time.time())
    time_ind = localtime.tm_sec//3
    if time_ind != time_sta:
        time_sta = time_ind
        emo_coun = []
        for couns in range(21):
            ret, frame = cap.read()
            roi = frame[y0:y0+height, x0:x0+width]
            gray_img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
            test_img = prepare_image(gray_img)
        
            test_img = test_img.reshape(-1, 48, 48, 1)
            classes = model.predict(test_img)
            emo_coun.append(emo_classes[np.argmax(classes)])
        emo_class = Counter(emo_coun)
        emo_class = emo_class.most_common(1)[0][0]
        title = emo_class
        if emo_class == 'Angry':            
                ser.write(b'1')
        elif emo_class == 'Happy':
                ser.write(b'2')
        #print(emo_class)
    else:
        pass

    k = cv2.waitKey(10)
    if k == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
