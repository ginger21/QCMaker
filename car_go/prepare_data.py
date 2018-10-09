import os
from random import sample
from collections import defaultdict
from PIL import Image
import numpy as np

# 定义函数 生成指定类别文件夹中的图片名称列表
def creat_paths(root_path, classes):
    # 生成空列表用于存放图片名称
    image_paths = defaultdict(list)
    # 循环对类别列表进行遍历
    for class_ in classes:
    # 生成指定类别文件夹路径
        image_dir = os.path.join(root_path, class_)
        # 对文件夹内图片文件遍历 将名称写入列表中存处
        for filepath in os.listdir(image_dir):
            if filepath.endswith('.jpg'):
                image_paths[class_].append(os.path.join(image_dir, filepath))
    # 返回所有图片名称列表
    return image_paths

# 定义函数 将图片拆分为训练集和测试集 拆分比例默认为20%
def prepare_data(image_paths, data_path, classes, ratio = 0.2):
    # 将训练集大小和测试集大小初始化为0
    train_sizes = 0
    val_sizes = 0
    # 循环遍历类别
    for class_ in classes:
        
        # 创建每类训练集的文件夹
        train_path = os.path.join(data_path,'train', class_)
        # 判断文件夹是否存在
        if os.path.exists(train_path):
            # 如果文件夹存在并且包含图片 则选取最后一个图片的文件计数作为新图片的计数开始
            # 防止存入的新图片与原有图片名称冲突 该文件夹内图片命名规则为5位数表示的数字
            # （如：第235幅图为00235.jpg，若最后一个图片为00456.jpg，则拆分到该文件夹的
            # 新图片从456开始计数 train_file_num = 456）
            if len(os.listdir(train_path)) > 0:
                train_file_num = int(os.listdir(train_path)[-1][0:5])
            else:
                # 空文件夹则计数从0开始
                train_file_num = 0
        else:
            # 如果文件夹不存在 就创建该文件夹
            os.makedirs(train_path)
            train_file_num = 0

        # 创建测试集同训练集
        val_path = os.path.join(data_path,'val', class_)
        if os.path.exists(val_path):
            if len(os.listdir(val_path)) > 0:
                val_file_num = int(os.listdir(val_path)[-1][0:5])
            else:
                val_file_num = 0
        else:
            os.makedirs(val_path)
            val_file_num = 0
        
        # 该类别 训练集 数目等于 图片总数乘以（1-测试集比例）
        train_size = int(len(image_paths[class_]) * (1 - ratio))
        # 该类别 测试集数目等于总数减去训练集数目
        val_size = len(image_paths[class_]) - train_size

        # 对训练集和测试集数目进行累加得到总的数目
        train_sizes += train_size
        val_sizes += val_size

        # 将图片列表打乱 达到随机选取的目的
        np.random.shuffle(image_paths[class_])

        # 选取训练集和测试集列表
        train_files = image_paths[class_][:train_size]
        val_files = image_paths[class_][train_size:]

        # 生成训练数据 遍历文件列表
        for name, path in enumerate(train_files):
            # 生成图片名称 规则为5位数字 前面用0填充
            # （如：235.jpg保存为00235.jpg）
            pic_name = train_path + '/{:05d}.jpg'.format(train_file_num + name + 1)
            # 读取列表中的文件
            images = Image.open(path)
            # 保存到训练集文件夹中
            images.save(pic_name)
        # 生成val数据 同训练集
        for name, path in enumerate(val_files):
            pic_name = val_path + '/{:05d}.jpg'.format(val_file_num + name + 1)
            images = Image.open(path)
            images.save(pic_name)
        print(class_+' is done')
    return train_sizes, val_sizes        

# 共需要7种图像类型
classes = ['forward', 'left', 'right'  ,'stop', 'turn_left', 'turn_right', 'walk']

# 调用函数生成训练集和测试集 将路径修改为自己所需路径
# 采集到的数据目录
path_of_data = './'
# 生成训练集和测试集的目录
path_of_train_and_val = './'

dire_paths = creat_paths(path_of_data, classes)

prepare_data(dire_paths, path_of_train_and_val, classes)
