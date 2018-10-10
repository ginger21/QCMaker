import os
import sys
import numpy as np
import pandas as pd
from PIL import Image

#当前文件夹路径
root_path = sys.path[0]
csv_data = os.path.join(root_path, 'data/fer2013.csv')

# 图片保存路径
# 训练集
tra_path_root = os.path.join(root_path, 'data/train')
# 测试集
val_path_root = os.path.join(root_path, 'data/val')

emo_data = pd.read_csv(csv_data)
emo_gb = dict(list(emo_data.groupby(['emotion', 'Usage'])))

# classes: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
# type: Training, PublicTest
# 表情种类
classes={0:'Angry', 1:'Happy', 2:'Neutral'}

# 训练集图片数据
dataset_train = [[(emo_class_ids, 'Training'), 
               os.path.join(tra_path_root, classes[emo_class_ids])] 
                for emo_class_ids in classes]

# 测试集图片数据
dataset_val = [[(emo_class_ids, 'PublicTest'), 
               os.path.join(val_path_root, classes[emo_class_ids])] 
                for emo_class_ids in classes]

# 将数据保存为图像
for classes, save_path in dataset_train:
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pixel_ind = 0
    for pixels in emo_gb[classes]['pixels']:
        pixel = np.asarray([float(p) for p in pixels.split()]).reshape(48, 48)
        im = Image.fromarray(pixel).convert('L')
        image_name = os.path.join(save_path, '{:05d}.jpg'.format(pixel_ind))
        im.save(image_name)
        pixel_ind += 1
    print(classes, 'end')

for classes, save_path in dataset_val:
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pixel_ind = 0
    for pixels in emo_gb[classes]['pixels']:
        pixel = np.asarray([float(p) for p in pixels.split()]).reshape(48, 48)
        im = Image.fromarray(pixel).convert('L')
        image_name = os.path.join(save_path, '{:05d}.jpg'.format(pixel_ind))
        im.save(image_name)
        pixel_ind += 1
    print(classes, 'end')