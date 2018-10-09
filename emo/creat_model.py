import keras
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from keras.layers import concatenate
from keras.models import Model

def creat_model(input_shape, num_classes = 5):
    inputs = Input(shape=input_shape)
    # 48 x 48
    con_1 = Conv2D(32, (3, 3), activation='relu')(inputs) 
    # 46 x 46
    con_2 = Conv2D(32, (3, 3), activation='relu')(con_1)
    # 44 x 44
    con_2 = MaxPooling2D(strides = 2)(con_2)
    con_2 = Dropout(0.25)(con_2)
    # 22 x 22
    con_3 = Conv2D(64, (3, 3), activation='relu')(con_2)
    # 20 x 20
    con_4 = Conv2D(64, (3, 3), activation='relu')(con_3)
    # 18 x 18
    con_4 = MaxPooling2D(strides = 2)(con_4)
    con_4 = Dropout(0.25)(con_4)
    # 9 x 9
    con_5_a1 = Conv2D(96, (1, 1), padding = 'same',
        activation='relu')(con_4)
    con_5_b1 = Conv2D(64, (1, 1), padding = 'same',
        activation='relu')(con_4)
    con_5_b2 = Conv2D(96, (3, 3), padding = 'same',
        activation='relu')(con_5_b1)
    con_5_c1 = Conv2D(32, (1, 1), padding = 'same',
        activation='relu')(con_4)
    con_5_c2 = Conv2D(64, (3, 3), padding = 'same',
        activation='relu')(con_5_c1)
    con_5_c3 = Conv2D(64, (3, 3), padding = 'same',
        activation='relu')(con_5_c2)
    con_5_d1 = MaxPooling2D((3, 3), strides = (1, 1),
        padding = 'same')(con_4)
    con_5_d2 = Conv2D(64, (1, 1), padding = 'same',
        activation='relu')(con_5_d1)
    
    con_5 = keras.layers.concatenate(
        [con_5_a1, con_5_b2, con_5_c3, con_5_d2], axis=3)
    
    con_6 = GlobalAveragePooling2D()(con_5)
    con_6 = Dropout(0.5)(con_6)

    con_7 = Dense(num_classes, activation='softmax')(con_6)

    model = Model(inputs=inputs, outputs=con_7)
    model.compile(loss='categorical_crossentropy', 
        optimizer="adam", metrics=['accuracy'])
    
    return model
    
