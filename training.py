from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import os
from creat_model import creat_model

train_samples = 16175   # 训练样本数
val_samples = 1969  # 测试样本数
epochs = 30  #训练轮数
batch_size = 32  #批次大小

img_width, img_height, channels = 48, 48, 1
input_shape = (img_width, img_height, channels)

# 训练集和测试机路径
target = 'c:/py/emo_net/k_inception/'
train_data_dir = target + '/train'
val_data_dir = target + '/val'

train_pic_gen = ImageDataGenerator(rescale=1. / 255)  # 对输入图片进行归一化到0-1区间
val_pic_gen = ImageDataGenerator(rescale=1. / 255)

classes = ['Angry', 'Happy', 'Neutral']

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

model = creat_model(input_shape, num_classes=3)
model.summary()

# 权重文件保存文件夹
weights_path = 'c:/py/emo_net/k_inception/weights'

# 权重文件保存位置
weights_file = 'c:/py/emo_net/k_inception/weights/weights.h5'

if not os.path.exists(weights_path):
    os.makedirs(weights_path)

if os.path.exists(weights_file):
    model.load_weights(weights_file)

checkpointer = ModelCheckpoint(
    # 权重保存路径
    filepath=weights_file, 
    verbose=1, save_best_only=True)
model.fit_generator(
    train_flow,
    steps_per_epoch=train_samples / batch_size,
    epochs=epochs,
    validation_data=val_flow,
    validation_steps=val_samples / batch_size,
    callbacks=[checkpointer])