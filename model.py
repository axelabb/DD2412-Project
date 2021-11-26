import tensorflow as tf
from tensorflow.keras.layers import Input,Permute, Reshape, Conv2D, BatchNormalization, Activation,Add,AveragePooling2D,Flatten,Dense


def basic_block(input,filters,strides,l_2):
    y = input
    x = BatchNormalization(momentum=0.9,epsilon=1e-5,beta_regularizer=l2(l_2),gamma_regularizer=l2(l_2))(input)
    x = Activation('relu')(x)
    x = Conv2D(filters,3,strides=strides,padding ='same',use_bias=False,kernel_initializer="he_normal",kernel_regularizer=l2(l_2))(x)
    x = BatchNormalization(momentum=0.9,epsilon=1e-5,beta_regularizer=l2(l_2),gamma_regularizer=l2(l_2))(x)
    x = Activation('relu')(x)
    x = Conv2D(filters,3,strides=1,padding ='same',use_bias=False,kernel_initializer="he_normal",kernel_regularizer=l2(l_2))(x)
    
    if not x.shape.is_compatible_with(y.shape):
        y = Conv2D(filters,1,strides=strides,padding ='same',use_bias=False,kernel_initializer="he_normal",kernel_regularizer=l2(l_2))(input)

    return Add()([x,y])


def res_group(input,filters,strides,n_blocks,l_2):
    x = basic_block(input,filters,strides,l_2)
    for _ in range(n_blocks-1):
        x = basic_block(x,filters,1,l_2)
    return x

def wide_resnet(input_shape,d,w_mult,n_classes,l_2=0):
    n_blocks = (d - 4) // 6
    input_shape = list(input_shape)
    ensemble_size = input_shape[0]

    input = Input(shape=input_shape)

    x = Permute([2,3,4,1])(input)

    # Reshape so that each subnetwork has 3 channels
    x = Reshape(input_shape[1:-1] + [input_shape[-1] * ensemble_size])(x)

    x = Conv2D(16,3,padding ='same',use_bias=False,kernel_initializer="he_normal",kernel_regularizer=l2(l_2))(x)

    for strides, filters in zip([1, 2, 2], [16, 32, 64]):
        x = res_group(x,filters*w_mult,strides,n_blocks,l_2)

    x = BatchNormalization(momentum=0.9,epsilon=1e-5,beta_regularizer=l2(l_2),gamma_regularizer=l2(l_2))(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    x = Flatten()(x)

    x = Dense(n_classes*ensemble_size,kernel_initializer='he_normal',activation=None,kernel_regularizer=l2(l_2),bias_regularizer=l2(l_2))(x)
    x = Reshape([ensemble_size,n_classes])(x)
    
    return tf.keras.Model(input,x)