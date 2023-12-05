import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, Conv3D, Flatten, Dense, BatchNormalization, Dropout, GlobalAveragePooling3D,\
                        GlobalAveragePooling2D, LayerNormalization, Embedding, add, MultiHeadAttention
from tensorflow_addons.layers import GroupNormalization
from tensorflow_addons.optimizers import AdamW
from keras import losses
from keras import metrics

DATA_FMT = 'channels_last'
    
def sp_attention(x, heads, dim):
    if DATA_FMT == 'channels_first':
        x = tf.transpose(x, [0,2,3,1])
        
    h = x.shape[1]
    w = x.shape[2]
    x0 = x
    x = Conv2D(heads, 1, activation="relu")(x) # [b, h, w, heads]
    x = tf.reshape(x, [-1, h*w, heads])        # [b, h*w, heads]
    x = tf.transpose(x, [0,2,1])               # [b, heads, h*w]
    x = Dense(h*w*2, activation="gelu")(x)     # [b, heads, h*w*2] 
    x = Dense(h*w  , activation="gelu")(x)     # [b, heads, h*w]          
    x = tf.transpose(x, [0,2,1])               # [b, h*w, heads]
    x = Dense(1, activation="sigmoid")(x)      # [b, h*w, 1]
    x = tf.reshape(x, [-1, h, w, 1])           # [b, h, w, 1]
    x = tf.tile(x, [1,1,1,dim])
    x = x0 + x0*x
    
    if DATA_FMT == 'channels_first':
        x = tf.transpose(x, [0,3,1,2])
    return x

def ch_attention(x, dim):
    x0 = x
    x = GlobalAveragePooling2D(data_format=DATA_FMT)(x)
    x = Dense(dim*2, activation="relu")(x)
    x = Dense(dim, activation="sigmoid")(x)
    
    if DATA_FMT == 'channels_first':
        x = tf.reshape(x, [-1,dim,1,1])
    else:
        x = tf.reshape(x, [-1,1,1,dim])
        
    x = x0 + x0*x
    
    return x


def mlp(x, hidden_units, dropout_rate=0.4, trainable=True):
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu, trainable=trainable)(x)
        x = Dropout(dropout_rate)(x)
    return x


def block(x, dim):
    
    if DATA_FMT == 'channels_first':
        GNorm_axis = 1
    else:
        GNorm_axis = -1
    
 
    x0 = x
    x = sp_attention(x, 12, dim)
    x = Conv2D(dim*4, 1, 1, padding="same", activation='gelu', data_format=DATA_FMT)(x)
    x = Conv2D(dim  , 1, 1, padding="same", activation='gelu', data_format=DATA_FMT)(x)
    x = GroupNormalization(16, epsilon=1e-6, axis=GNorm_axis)(x)
    x = Conv2D(dim  , 3, 1, padding="same", activation='relu', data_format=DATA_FMT)(x)
    x = ch_attention(x, dim)
    
    x = x + x0
    
    return x


def move_pred_mix_v2(shape, dim=64):
    
    inputs = Input(shape=shape)
    
    if DATA_FMT == 'channels_first':
        x = tf.transpose(inputs, [0,3,1,2])
        GNorm_axis = 1
    else:
        x = inputs
        GNorm_axis = -1
        
    x = Conv2D(dim  , 5, 1, padding='same', activation="gelu", data_format=DATA_FMT)(x)
    x = GroupNormalization(16, epsilon=1e-6, axis=GNorm_axis)(x)
    
    for i in range(2):
        x = block(x, dim)
    
    dim *= 2
    x = GroupNormalization(16, epsilon=1e-6, axis=GNorm_axis)(x)
    x = Conv2D(dim  , 5, 2, padding='same', activation='gelu', data_format=DATA_FMT)(x)
    

    for i in range(3):
        x = block(x, dim)
        
    dim *= 2
    x = GroupNormalization(16, epsilon=1e-6, axis=GNorm_axis)(x)
    x = Conv2D(dim  , 5, 2, padding='same', activation='gelu', data_format=DATA_FMT)(x)    
    

    for i in range(3):
        x = block(x, dim)
        
    x = GroupNormalization(16, epsilon=1e-6, axis=GNorm_axis)(x)
    x = GlobalAveragePooling2D(data_format=DATA_FMT)(x)
    x = mlp(x, hidden_units=[dim*4, dim], dropout_rate=0.3)
    
    # Classify outputs.
    x = Dense(361, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=x)

    return model
