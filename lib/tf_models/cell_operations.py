##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import tensorflow as tf

__all__ = ['OPS', 'ResNetBasicblock', 'SearchSpaceNames']

OPS = {
  'none'        : lambda C_in, C_out, stride, affine: Zero(C_in, C_out, stride),
  'avg_pool_3x3': lambda C_in, C_out, stride, affine: POOLING(C_in, C_out, stride, 'avg', affine),
  'nor_conv_1x1': lambda C_in, C_out, stride, affine: ReLUConvBN(C_in, C_out, 1, stride, affine),
  'nor_conv_3x3': lambda C_in, C_out, stride, affine: ReLUConvBN(C_in, C_out, 3, stride, affine),
  'nor_conv_5x5': lambda C_in, C_out, stride, affine: ReLUConvBN(C_in, C_out, 5, stride, affine),
  'skip_connect': lambda C_in, C_out, stride, affine: Identity(C_in, C_out, stride) if stride == 1 and C_in == C_out else FactorizedReduce(C_in, C_out, stride, affine)
}

NAS_BENCH_201         = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']

SearchSpaceNames = {
                    'nas-bench-201': NAS_BENCH_201,
                   }


class POOLING(tf.keras.layers.Layer):

  def __init__(self, C_in, C_out, stride, mode, affine):
    super(POOLING, self).__init__()
    if C_in == C_out:
      self.preprocess = None
    else:
      self.preprocess = ReLUConvBN(C_in, C_out, 1, 1, affine)
    if mode == 'avg'  : self.op = tf.keras.layers.AvgPool2D((3,3), strides=stride, padding='same')
    elif mode == 'max': self.op = tf.keras.layers.MaxPool2D((3,3), strides=stride, padding='same')
    else              : raise ValueError('Invalid mode={:} in POOLING'.format(mode))

  def call(self, inputs, training):
    if self.preprocess: x = self.preprocess(inputs)
    else              : x = inputs
    return self.op(x)


class Identity(tf.keras.layers.Layer):
  def __init__(self, C_in, C_out, stride):
    super(Identity, self).__init__()
    if C_in != C_out or stride != 1:
      self.layer = tf.keras.layers.Conv2D(C_out, 3, stride, padding='same', use_bias=False)
    else:
      self.layer = None
  
  def call(self, inputs, training):
    x = inputs
    if self.layer is not None:
      x = self.layer(x)
    return x



class Zero(tf.keras.layers.Layer):
  def __init__(self, C_in, C_out, stride):
    super(Zero, self).__init__()
    if C_in != C_out:
      self.layer = tf.keras.layers.Conv2D(C_out, 1, stride, padding='same', use_bias=False)
    elif stride != 1:
      self.layer = tf.keras.layers.AvgPool2D((stride,stride), None, padding="same")
    else:
      self.layer = None
  
  def call(self, inputs, training):
    x = tf.zeros_like(inputs)
    if self.layer is not None:
      x = self.layer(x)
    return x


class ReLUConvBN(tf.keras.layers.Layer):
  def __init__(self, C_in, C_out, kernel_size, strides, affine):
    super(ReLUConvBN, self).__init__()
    self.C_in = C_in
    self.relu = tf.keras.activations.relu
    self.conv = tf.keras.layers.Conv2D(C_out, kernel_size, strides, padding='same', use_bias=False)
    self.bn   = tf.keras.layers.BatchNormalization(center=affine, scale=affine)
  
  def call(self, inputs, training):
    x = self.relu(inputs)
    x = self.conv(x)
    x = self.bn(x, training)
    return x


class FactorizedReduce(tf.keras.layers.Layer):
  def __init__(self, C_in, C_out, stride, affine):
    assert output_filters % 2 == 0, ('Need even number of filters when using this factorized reduction.')
    self.stride == stride
    self.relu   = tf.keras.activations.relu
    if stride == 1:
      self.layer = tf.keras.Sequential([
                          tf.keras.layers.Conv2D(C_out, 1, strides, padding='same', use_bias=False),
                          tf.keras.layers.BatchNormalization(center=affine, scale=affine)])
    elif stride == 2:
      stride_spec = [1, stride, stride, 1] # data_format == 'NHWC'
      self.layer1 = tf.keras.layers.Conv2D(C_out//2, 1, strides, padding='same', use_bias=False)
      self.layer2 = tf.keras.layers.Conv2D(C_out//2, 1, strides, padding='same', use_bias=False)
      self.bn     = tf.keras.layers.BatchNormalization(center=affine, scale=affine)
    else:
      raise ValueError('invalid stride={:}'.format(stride))

  def call(self, inputs, training):
    x = self.relu(inputs)
    if self.stride == 1:
      return self.layer(x, training)
    else:
      path1 = x
      path2 = tf.pad(x, [[0, 0], [0, 1], [0, 1], [0, 0]])[:, 1:, 1:, :] # data_format == 'NHWC'
      x1 = self.layer1(path1)
      x2 = self.layer2(path2)
      final_path = tf.concat(values=[x1, x2], axis=3)
      return self.bn(final_path)


class ResNetBasicblock(tf.keras.layers.Layer):

  def __init__(self, inplanes, planes, stride, affine=True):
    super(ResNetBasicblock, self).__init__()
    assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
    self.conv_a = ReLUConvBN(inplanes, planes, 3, stride, affine)
    self.conv_b = ReLUConvBN(  planes, planes, 3,      1, affine)
    if stride == 2:
      self.downsample = tf.keras.Sequential([
                                tf.keras.layers.AvgPool2D((stride,stride), None, padding="same"),
                                tf.keras.layers.Conv2D(planes, 1, 1, padding='same', use_bias=False)])
    elif inplanes != planes:
      self.downsample = ReLUConvBN(inplanes, planes, 1, stride, affine)
    else:
      self.downsample = None
    self.addition = tf.keras.layers.Add()
    self.in_dim  = inplanes
    self.out_dim = planes
    self.stride  = stride
    self.num_conv = 2

  def call(self, inputs, training):

    basicblock = self.conv_a(inputs, training)
    basicblock = self.conv_b(basicblock, training)

    if self.downsample is not None:
      residual = self.downsample(inputs)
    else:
      residual = inputs
    return self.addition([residual, basicblock])
