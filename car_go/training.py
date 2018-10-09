# 训练生成图片
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD

#=================== 设置初始参数 ==================#
train_samples = 3400  # 训练样本数
val_samples = 860  # 测试样本数
epochs = 10  #训练轮数
batch_size = 32  #批次大小

# 输入参数
img_width, img_height, channels = 60, 80, 1
input_shape = (img_width, img_height, channels)

# 训练图片路径
target = './'
train_data_dir = target + 'train'
val_data_dir = target + 'val'

#================== 对训练数据进行处理 ====================#
# 设置对输入图片进行归一化到0-1区间的图片生成器
train_pic_gen = ImageDataGenerator(rescale=1. / 255)  
val_pic_gen = ImageDataGenerator(rescale=1. / 255)

# 图片分类
classes = ['forward', 'left', 'right', 
            'stop', 'turn_left', 'turn_right', 'walk']

# 通过图片生成器得到训练流
# 参数：
#   directory：图片路径（train_data_dir）
#   target_size：实现对图片的尺寸转换
#   batch_size:批次大小
#   color_mode：色彩模式
#   classes：分类的列表
#   class_mode：分类种类
train_flow = train_pic_gen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),  #调整图像大小
    batch_size=batch_size,
    color_mode='grayscale',  #输入图片为灰度图片
    classes=[class_ for class_ in classes],
    class_mode='categorical')

val_flow = val_pic_gen.flow_from_directory(
    val_data_dir,
    target_size=(img_width, img_height),  #调整图像大小
    batch_size=batch_size,
    color_mode='grayscale',  #输入图片为灰度图片
    classes=[class_ for class_ in classes],  # 标签
    class_mode='categorical')

#===================== 创建训练模型 ========================#

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

# 显示模型情况
model.summary()

# 模型检查点，用于回调函数callbacks
# 参数：
#   filepath：每个epoch后保存模型路径
#   verbose：信息展示模式，0或1
#   save_best_only：若设置为True，则只保存模型权重，否则将保存整个模型
checkpointer = ModelCheckpoint(
    filepath="./weights/weights.h5", 
    verbose=1, save_best_only=True)

if os.path.exists("./weights/weights.h5"):
    model.load_weights("./weights/weights.h5")

if not os.path.exists("./weights"):
    os.mkdir("./weights")

# 开始训练模型：
# 参数：
#   generator：数据产生器（train_flow)
#   steps_per_epoch：每个循环的训练步数（训练数据/批次大小）
#   epochs：循环次数
#   validation_data：验证集数据
#   validation_steps：验证集步数（验证集大小/批次大小）
#   callbacks：回调函数，保存模型
model.fit_generator(
    train_flow,
    steps_per_epoch=train_samples / batch_size,
    epochs=epochs,
    validation_data=val_flow,
    validation_steps=val_samples / batch_size,
    callbacks=[checkpointer])

