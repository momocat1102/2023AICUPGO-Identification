from keras.layers import (
    Input, Dense, Dropout, LayerNormalization, add, MultiHeadAttention, Embedding,
    Layer, Conv2D, ReLU, Flatten, Dense, Softmax, BatchNormalization, GlobalAveragePooling2D, Multiply
)
from tensorflow_addons.layers import GELU
from keras.models import Model
# import tensorflow as tf

def sp_attention(x, heads=8):
    x0 = x
    x = Conv2D(heads, 1, activation="relu")(x)
    x = Conv2D(1, 1, activation="sigmoid")(x)
    return  x0 * x + x0

def ch_attention(x, dim):
    x0 = x
    x = GlobalAveragePooling2D()(x)
    x = Dense(dim//2, activation="relu")(x)
    x = Dense(dim, activation="sigmoid")(x)

    x = add([x0, Multiply()([x0, x])])
    return x

def conv_layer(x, dim, kernel_size=3, strides=1):
    x = Conv2D(dim, kernel_size, strides, padding='same')(x)
    x = BatchNormalization(epsilon=1e-6)(x)
    x = GELU()(x)
    return x

def res_block(x, dim, kernel_size=3):
    input_tensor = x
    x = conv_layer(x, dim//2, 1) # (n, n, dim)
    x1 = x
    x = conv_layer(x, dim//2, kernel_size) # (n, n, dim//2)
    x = conv_layer(x, dim//2, kernel_size) # (n, n, dim//2)
    x2 = x1 + x
    x = conv_layer(x2, dim//2, kernel_size) # (n, n, dim//2)
    x = conv_layer(x, dim//2, kernel_size) # (n, n, dim//2)
    x3 = x2 + x
    x = conv_layer(x3, dim, 1) # (n, n, dim)

    return x + input_tensor

def kata_cnn_2(shape=(19,19,4)):
    inputs = Input(shape=shape)
    
    x0 = Conv2D(32, 5, padding='same', activation='gelu')(inputs)
    x1 = Conv2D(32, 3, padding='same', activation='gelu')(inputs)
    x3 = Conv2D(32, 7, padding='same', activation='gelu')(inputs)
    
    x0 = conv_layer(x0, 64, 3)
    x1 = conv_layer(x1, 64, 3)
    x3 = conv_layer(x3, 64, 3)

    x0 = Conv2D(64, 3, padding='same', activation='relu')(x0)
    x1 = Conv2D(64, 3, padding='same', activation='relu')(x1)
    x3 = Conv2D(64, 3, padding='same', activation='relu')(x3)
    
    x = x0 + x1 + x3
    x = Conv2D(128, 2, 2, padding='same')(x)
    x = BatchNormalization()(x)
    
    for i, filter in enumerate([128, 128, 256, 256, 512, 512]):
        if i in [2, 4]:
            x = conv_layer(x, filter, 3, 2)
        x = ch_attention(x, filter)
        x = sp_attention(x)
        x = res_block(x, filter, 3)

    x = Conv2D(512, 2, padding='same', activation='gelu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(361, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=x)

    return model