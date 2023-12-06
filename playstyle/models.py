import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense, BatchNormalization, Dropout, GlobalAveragePooling2D, SpatialDropout2D
                        
from tensorflow_addons.layers import GroupNormalization
from tensorflow_addons.optimizers import AdamW
from keras import losses
from keras import metrics

DATA_FMT = 'channels_first'
 
def sp_attention(x, heads, dim):
    if DATA_FMT == 'channels_first':
        x = tf.transpose(x, [0,2,3,1])
        
    h = x.shape[1]
    w = x.shape[2]
    x0 = x
    x = Conv2D(heads, 1, activation="gelu")(x) # [b, h, w, heads]
    x = tf.reshape(x, [-1, h*w, heads])        # [b, h*w, heads]
    x = tf.transpose(x, [0,2,1])               # [b, heads, h*w]
    x = Dense(h*w*2, activation="relu")(x)     # [b, heads, h*w*2] 
    x = Dense(h*w  , activation=None)(x)       # [b, heads, h*w]          
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

def ps_block(x, dim):
    
    norm_axis = 1 if DATA_FMT == 'channels_first' else -1
    
    x0 = x
    x = sp_attention(x, 12, dim)
    x = Conv2D(dim  , 5, 1, padding='same', activation='gelu', data_format=DATA_FMT, groups=dim)(x)
    x = GroupNormalization(16, epsilon=1e-6, axis=norm_axis)(x)
    x = ch_attention(x, dim)
    x = SpatialDropout2D(0.15, data_format=DATA_FMT)(x)
    
    x = Conv2D(dim  , 5, 1, padding='same', activation='gelu', data_format=DATA_FMT)(x)
    x = sp_attention(x, 12, dim)
    x = ch_attention(x, dim)
    x = SpatialDropout2D(0.15, data_format=DATA_FMT)(x)
    x = x + x0
    return x

def ps_cnn_atte(dim=80, n_last_move=3, n_channels=4, lr=0.0012):
    
    inputs = Input(shape=(19,19,n_channels*n_last_move))
        
    if DATA_FMT == 'channels_first':
        x = tf.transpose(inputs, [0,3,1,2])
        norm_axis = 1
    else:
        x = inputs
        norm_axis = -1    
        
    x = Conv2D(dim, 5, 1, padding='same', activation='gelu', data_format=DATA_FMT)(x)
 
    for i in range(2):
        x = ps_block(x, dim)
        
    dim *= 2
    # downsample
    x = GroupNormalization(16, epsilon=1e-6, axis=norm_axis)(x)
    x = Conv2D(dim, 5, 2, padding='same', activation='gelu', data_format=DATA_FMT)(x)
    
    for i in range(2):
        x = ps_block(x, dim)

    dim *= 2
    # downsample
    x = GroupNormalization(16, epsilon=1e-6, axis=norm_axis)(x)
    x = Conv2D(dim, 5, 2, padding='same', activation='gelu', data_format=DATA_FMT)(x)
    
    for i in range(2):
        x = ps_block(x, dim)

    x = GroupNormalization(16, epsilon=1e-6, axis=norm_axis)(x)
    x = GlobalAveragePooling2D(data_format=DATA_FMT)(x)
    x = Dense(512, activation='gelu')(x)
    x = Dropout(0.3)(x)
    x = Dense(3, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=x)
    model.compile(loss=losses.CategoricalCrossentropy(),
                  optimizer=AdamW(learning_rate=lr, weight_decay=0.0001),
                  metrics=[metrics.CategoricalAccuracy('acc')]
                  )
    return model
