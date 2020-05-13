from .imports import *

__all__ = ['residual_module', 'residual_transpose_module']


def residual_module(layer_in, n_filters):
    merge_input = layer_in
    if layer_in.shape[-1] != n_filters:
        merge_input = Conv2D(n_filters, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
    conv1 = Conv2D(n_filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
    conv2 = Conv2D(n_filters, (3,3), padding='same', activation='linear', kernel_initializer='he_normal')(conv1)
    layer_out = add([conv2, merge_input])
    layer_out = Activation('relu')(layer_out)
    return layer_out
  
def residual_transpose_module(layer_in, n_filters):
    merge_input = layer_in
    if layer_in.shape[-1] != n_filters:
        merge_input = Conv2DTranspose(n_filters, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
    conv1 = Conv2DTranspose(n_filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
    conv2 = Conv2DTranspose(n_filters, (3,3), padding='same', activation='linear', kernel_initializer='he_normal')(conv1)
    layer_out = add([conv2, merge_input])
    layer_out = Activation('relu')(layer_out)
    return layer_out

def vgg_block(x, n_filters, n_conv):
    # https://machinelearningmastery.com/how-to-implement-major-architecture-innovations-for-convolutional-neural-networks/
	for _ in range(n_conv):
		x = Conv2D(n_filters, (3,3), padding='same', activation='relu')(x)
	x = MaxPooling2D((2,2), strides=(2,2))(x)
	return x

def vgg_transpose_block(x, n_filters, n_conv):
    # https://machinelearningmastery.com/how-to-implement-major-architecture-innovations-for-convolutional-neural-networks/
	for _ in range(n_conv):
		x = Conv2DTranspose(n_filters, (3,3), padding='same', activation='relu')(x)
	x = UpSampling2D((2,2))(x)
	return x

# function for creating a projected inception module
def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
	# 1x1 conv
	conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)
	# 3x3 conv
	conv3 = Conv2D(f2_in, (1,1), padding='same', activation='relu')(layer_in)
	conv3 = Conv2D(f2_out, (3,3), padding='same', activation='relu')(conv3)
	# 5x5 conv
	conv5 = Conv2D(f3_in, (1,1), padding='same', activation='relu')(layer_in)
	conv5 = Conv2D(f3_out, (5,5), padding='same', activation='relu')(conv5)
	# 3x3 max pooling
	pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
	pool = Conv2D(f4_out, (1,1), padding='same', activation='relu')(pool)
	# concatenate filters, assumes filters/channels last
	layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
	return layer_out

# function for creating a projected inception module
def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
	# 1x1 conv
	conv1 = Conv2DTranspose(f1, (1,1), padding='same', activation='relu')(layer_in)
	# 3x3 conv
	conv3 = Conv2DTranspose(f2_in, (1,1), padding='same', activation='relu')(layer_in)
	conv3 = Conv2DTranspose(f2_out, (3,3), padding='same', activation='relu')(conv3)
	# 5x5 conv
	conv5 = Conv2DTranspose(f3_in, (1,1), padding='same', activation='relu')(layer_in)
	conv5 = Conv2DTranspose(f3_out, (5,5), padding='same', activation='relu')(conv5)
	# 3x3 max pooling
	pool = UpSampling2D((3,3), strides=(1,1), padding='same')(layer_in)
	pool = Conv2DTranspose(f4_out, (1,1), padding='same', activation='relu')(pool)
	# concatenate filters, assumes filters/channels last
	layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
	return layer_out