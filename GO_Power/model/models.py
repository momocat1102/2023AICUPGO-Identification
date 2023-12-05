from keras.layers import (
    Input, Dense, Dropout, LayerNormalization, add, MultiHeadAttention, Embedding,
    Layer, Conv2D, ReLU, Flatten, Dense, Softmax, BatchNormalization, GlobalAveragePooling2D, Multiply
)
from tensorflow_addons.layers import GELU
from keras.models import Model
import tensorflow as tf

DATA_FORMAT = "channels_last"
if DATA_FORMAT == "channels_first":
    nor_axis = 1
else:
    nor_axis = -1

def cnn_model(shape=(19,19,4)):
    inputs = Input(shape=shape)
    
    x0 = Conv2D(32, 5, padding='same', activation='gelu')(inputs)
    x1 = Conv2D(32, 3, padding='same', activation='gelu')(inputs)
    
    x0 = Conv2D(64, 3, 2, padding='same')(x0)
    x0 = BatchNormalization()(x0)
    
    x1 = Conv2D(64, 3, 2, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    
    x0 = Conv2D(64, 3, padding='same', activation='relu')(x0)
    x1 = Conv2D(64, 3, padding='same', activation='relu')(x1)
    
    x = x0 + x1
    x = Conv2D(128, 2, 2, padding='same')(x)
    x = BatchNormalization()(x)
    
    for i in range(2):
        x0 = x
        x = Conv2D(128, 5, padding='same', activation='gelu')(x0)
        x = Conv2D(384, 1, padding='same', activation= None)(x)
        x = Conv2D(128, 1, padding='same', activation='relu')(x)
        x = LayerNormalization(epsilon=1e-5)(x) + x0
    
    x = Conv2D(256, 3, 2, padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv2D(384, 2, padding='same', activation='gelu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(361, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=x)

    return model

# def sp_attention(x, heads=8):
#     x0 = x
#     x = Conv2D(heads, 1, activation="relu")(x)
#     x = Conv2D(1, 1, activation="sigmoid")(x)
#     x = Multiply()([x0, x])
#     return  x0 * x + x0
    
# def sp_attention_(x, heads, dim):
#     h = x.shape[1]
#     w = x.shape[2]
#     x0 = x
#     x = Conv2D(heads, 1, activation="relu")(x) # [b, h, w, heads]
#     x = tf.reshape(x, [-1, h*w, heads])        # [b, hw, heads]
#     x = tf.transpose(x, [0,2,1])               # [b, heads, hw]
#     x = Dense(h*w*2, activation="gelu")(x)     # [b, heads, hw2] 
#     x = Dense(h*w, activation="sigmoid")(x)    # [b, heads, hw]
#     x = tf.transpose(x, [0,2,1])               # [b, hw, heads]
#     x = tf.reduce_mean(x, axis=-1, keepdims=True) # [b, hw, 1]
#     x = tf.reshape(x, [-1, h, w, 1])           # [b, h, w, 1]
#     x = tf.tile(x, [1,1,1,dim])
#     x = x0 + x0*x
#     return x

def ch_attention(x, dim):
    x0 = x
    x = GlobalAveragePooling2D(data_format=DATA_FORMAT)(x)
    x = Dense(dim//2, activation="relu")(x)
    x = Dense(dim, activation="sigmoid")(x)
    if DATA_FORMAT == "channels_first":
        x = tf.reshape(x, [-1,dim,1,1])
    else:
        x = tf.reshape(x, [-1,1,1,dim])

    return x0*x + x0

def conv_layer(x, dim, kernel_size=3, strides=1):
    x = Conv2D(dim, kernel_size, strides, padding='same', data_format=DATA_FORMAT)(x)
    x = BatchNormalization(axis=nor_axis, epsilon=1e-6)(x)
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

def kata_cnn(shape=(19,19,4), dim=32):
    inputs = Input(shape=shape)
    x = inputs
    if DATA_FORMAT == "channels_first":
        x = tf.transpose(inputs, [0,3,1,2])

    x0 = Conv2D(dim, 5, padding='same', activation='gelu', data_format=DATA_FORMAT)(x)
    x1 = Conv2D(dim, 3, padding='same', activation='gelu', data_format=DATA_FORMAT)(x)
    x3 = Conv2D(dim, 7, padding='same', activation='gelu', data_format=DATA_FORMAT)(x)
    
    x0 = conv_layer(x0, 64, 3)
    x1 = conv_layer(x1, 64, 3)
    x3 = conv_layer(x3, 64, 3)

    x0 = Conv2D(dim*2, 3, padding='same', activation='relu', data_format=DATA_FORMAT)(x0)
    x1 = Conv2D(dim*2, 3, padding='same', activation='relu', data_format=DATA_FORMAT)(x1)
    x3 = Conv2D(dim*2, 3, padding='same', activation='relu', data_format=DATA_FORMAT)(x3)
    
    x = x0 + x1 + x3

    x = BatchNormalization(axis=nor_axis, epsilon=1e-6)(x)

    x = Conv2D(dim*4, 3, 2, padding='same', activation='gelu', data_format=DATA_FORMAT)(x)
    
    for i, filter in enumerate([dim*4, dim*4, dim*8, dim*8, dim*16, dim*16]):
        if i in [2, 4]:
            x = conv_layer(x, filter, 3, 2)
        x = ch_attention(x, filter)
        # x = sp_attention(x)
        x = res_block(x, filter, 3)

    x = Conv2D(dim*16, 2, padding='same', activation='gelu', data_format=DATA_FORMAT)(x)
    x = GlobalAveragePooling2D(data_format=DATA_FORMAT)(x)
    x = Dense(dim*8, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(361, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=x)

    return model

def cnn_model_mix():
    inputs = Input(shape=(19,19,5))
    
    x0 = Conv2D(32, 5, padding='same', activation='gelu')(inputs)
    x1 = Conv2D(32, 3, padding='same', activation='gelu')(inputs)
    
    x0 = Conv2D(64, 3, 2, padding='same')(x0)
    x0 = BatchNormalization()(x0)
    
    x1 = Conv2D(64, 3, 2, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    
    x0 = Conv2D(64, 3, padding='same', activation='relu')(x0)
    x1 = Conv2D(64, 3, padding='same', activation='relu')(x1)
    
    x = x0 + x1
    x = Conv2D(128, 2, 2, padding='same')(x)
    x = BatchNormalization()(x)
    
    for i in range(2):
        x0 = x
        x = Conv2D(128, 5, padding='same', activation='gelu')(x0)
        x = Conv2D(384, 1, padding='same', activation= None)(x)
        x = Conv2D(128, 1, padding='same', activation='relu')(x)
        x = LayerNormalization(epsilon=1e-5)(x) + x0
    
    x = Conv2D(256, 3, 2, padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv2D(384, 2, padding='same', activation='gelu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(361, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=x)

    return model

def efficientnet_model():
    inputs = Input(shape=(19,19,4))
    x = tf.cast(inputs, tf.float32)
    base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights=None, input_tensor=x, pooling='avg')
    outputs = Dense(361, activation="softmax")(base_model.output)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def create_model():
    inputs = Input(shape=(19, 19, 4))
    outputs = tf.cast(inputs, tf.float32)
    # outputs = Flatten()(outputs)
    # outputs = Dense(361)(outputs)
    # outputs = Softmax()(outputs)
    # model = Model(inputs, outputs)
    outputs = Conv2D(kernel_size=7, filters=8, padding='same', activation='relu')(outputs)
    outputs = Conv2D(kernel_size=7, filters=8, padding='same', activation='relu')(outputs)
    outputs = Conv2D(kernel_size=3, filters=16, padding='same', activation='relu', strides=2)(outputs)
    outputs = BatchNormalization()(outputs)

    outputs = Conv2D(kernel_size=5, filters=16, padding='same', activation='relu')(outputs)
    outputs = Conv2D(kernel_size=5, filters=16, padding='same', activation='relu')(outputs)
    outputs = Conv2D(kernel_size=3, filters=32, padding='same', activation='relu', strides=2)(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Conv2D(kernel_size=3, filters=32, padding='same', activation='relu')(outputs)
    outputs = Conv2D(kernel_size=3, filters=32, padding='same', activation='relu')(outputs)

    # outputs = GlobalAveragePooling2D()(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(361)(outputs)
    outputs = Softmax()(outputs)
    model = Model(inputs, outputs)
    
    return model

def alpha_residual_block(x, filters, kernel_size=3, strides=1):
    input_tensor = x
    x = Conv2D(filters, kernel_size, strides, padding='same')(x)
    x = BatchNormalization(epsilon=1e-6)(x)
    x = GELU()(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization(epsilon=1e-6)(x + input_tensor)
    x = GELU()(x)
    return x

def alphago_zero_model(input_shape=(19,19,17), block_num=10, filters=256):
    inputs = Input(shape=input_shape)
    x = Conv2D(filters, 3, padding='same')(inputs)
    for _ in range(block_num):
        x = alpha_residual_block(x, filters)
    x = Conv2D(2, 1, padding='same')(x)
    x = BatchNormalization(epsilon=1e-6)(x)
    x = GELU()(x)
    x = Flatten()(x)
    output = Dense(361, activation="softmax")(x)
    model = Model(inputs, output)

    return model

class ConvNeXtBlock(Layer):
    def __init__(self, dim, drop_path=0, layer_scale_init_value=1e-6):
        super().__init__()
        
                                                            # Conv2D中自帶groups參數
        self.dwconv = Conv2D(dim, 7, 1, padding="same", groups=dim) # depthwise conv
        self.ln = LayerNormalization(epsilon=1e-6)
        self.pwconv1 = Conv2D(4*dim, 1, 1, activation="gelu") # pointwise/1x1 convs,
        self.pwconv2 = Conv2D(dim, 1, 1)
        #Layer Scale
        self.gamma = tf.Variable(layer_scale_init_value * tf.ones((dim)), trainable=True) if layer_scale_init_value > 0 else None
        
    def call(self, x):
        input = x
        x = self.dwconv(x)
        x = self.ln(x)
        x = self.pwconv1(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = input + x
        return x
class ConvNeXt(tf.keras.Model):
    def __init__(self, num_blocks, dim=16):
        super().__init__()
        self.n = num_blocks

        self.conv1 = Conv2D(dim, kernel_size=4, strides=4) #Changing stem to “Patchify”
        self.blocks1 =  [ConvNeXtBlock(dim) for i in range(1*self.n)] # n x (1:1:3:1)
        self.blocks2 =  [ConvNeXtBlock(dim*2) for i in range(1*self.n)]
        self.blocks3 =  [ConvNeXtBlock(dim*4) for i in range(3*self.n)]
        # self.blocks4 =  [ConvNeXtBlock(768) for i in range(1*self.n)]
  
        self.down2 = Conv2D(dim*2, 2, 2, activation="gelu")
        self.down3 = Conv2D(dim*4, 2, 2, activation="gelu")
        # self.down4 = layers.Conv2D(768, 2, 2, activation="gelu")
        
        self.ln = LayerNormalization(epsilon=1e-6)
        self.ln2 = LayerNormalization(epsilon=1e-6)
        self.ln3 = LayerNormalization(epsilon=1e-6)
        self.ln4 = LayerNormalization(epsilon=1e-6)
        # self.ln5 = layers.LayerNormalization(epsilon=1e-6)
        
        self.gavpool = GlobalAveragePooling2D()
        self.fc1 = Dense(1024, activation="gelu")
        self.fc = Dense(361, activation="softmax") #這邊就看你要練的資料有幾類
              
    def call(self, inputs):
   
        x = self.conv1(inputs) 
        x = self.ln(x)
        
        for i in range(self.n):
            x = self.blocks1[i](x)
        
        x = self.ln2(x)      # 下採樣: LN + conv2d 
        x = self.down2(x)   
        for i in range(self.n):
            x = self.blocks2[i](x)
        
        x = self.ln3(x)
        x = self.down3(x)
        for i in range(self.n*3):
            x = self.blocks3[i](x)
            
        # x = self.ln4(x)
        # x = self.down4(x)
        # for i in range(self.n):
        #     x = self.blocks4[i](x)
  
        x = self.gavpool(x)
        x = self.ln4(x)
        x = self.fc(x)
        return x

def mlp(x, hidden_units, dropout_rate=0.4, trainable=True):
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu, trainable=trainable)(x)
        x = Dropout(dropout_rate)(x)
    return x

def Transformer_layers(encoded_patches, dim, layers, trainable=True):
    for _ in range(layers):
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = MultiHeadAttention(
            num_heads=8, key_dim=dim, dropout=0.2, trainable=trainable,
        )(x1, x1)
        x2 = add([attention_output, encoded_patches])
     
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=[dim*4,dim], dropout_rate=0.2, trainable=trainable)
        encoded_patches = add([x3, x2])
        
    return encoded_patches

def TransFormer(input_shape=(19, 19, 6), dim = 32, layers=3, out_dim=361, act="softmax"):
    inputs = Input(shape=input_shape)
    input_transpose = tf.transpose(inputs, [0,3,1,2])
    input_reshape = tf.reshape(input_transpose, [-1, input_shape[-1], 19 * 19])
    input_embedding = Dense(dim*4, activation='gelu')(input_reshape)
    input_embedding = LayerNormalization()(input_embedding)
    input_embedding = Dense(dim, activation='gelu')(input_embedding)

    x = Transformer_layers(input_embedding, dim, layers)
    
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    features = mlp(x, hidden_units=[512*3, 768], dropout_rate=0.3, trainable=True)
    # Classify outputs.
    outs = Dense(out_dim, activation=act)(features)
    
    model = Model(inputs=inputs, outputs=outs)

    return model