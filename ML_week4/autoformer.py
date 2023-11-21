"""
Python module implementing the necessary layers for an Autotransformer in Keras
"""

import tensorflow as tf
from tensorflow.math import conj
from tensorflow.signal import rfft, irfft

from tensorflow import keras
from keras.layers import AveragePooling1D

class DecompositionLayer(keras.layers.Layer):
  """
  Returns the trend and the seasonal parts of the time series.
  """

  def __init__(self, kernel_size):
    super(DecompositionLayer, self).__init__()
    self.kernel_size = kernel_size

  def build(self, input_shape):
    super(DecompositionLayer, self).build(input_shape)
    self.avg = AveragePooling1D(pool_size=self.kernel_size,
                                strides=1, padding="same")

  def call(self, inputs):
    """Input shape: Batch x Time x EMBED_DIM"""
    # padding on the both ends of time series
    num_of_pads = (self.kernel_size - 1) // 2
    front = keras.backend.repeat(inputs[:, 0:1, :], (1, num_of_pads, 1))
    end = keras.backend.repeat(inputs[:, -1:, :], (1, num_of_pads, 1))
    x_padded = keras.backend.concatenate([front, inputs, end], axis=1)

    # calculate the trend and seasonal part of the series
    x_trend = self.avg(keras.backend.permute_dimensions(x_padded, (0, 2, 1)))
    x_trend = keras.backend.permute_dimensions(x_trend, (0, 2, 1))
    x_seasonal = inputs - x_trend
    return x_seasonal, x_trend

def autocorrelation(query_states:tf.Tensor, key_states:tf.Tensor):
  """
  Computes autocorrelation(Q,K) using `keras.rfft`. 
  Think about it as a replacement for the QK^T in the self-attention.
  
  Assumption: states are resized to same shape of
  [batch_size, time_length, embedding_dim].
  """
  def channelwise_autocorrelation(query_states_channel,
                                  key_states_channel):
    query_states_fft = rfft(query_states_channel)
    key_states_fft = rfft(key_states_channel)
    attn_weights_fft = query_states_fft * conj(key_states_fft)

    return irfft(attn_weights_fft)

  return tf.concat(
    [channelwise_autocorrelation(query_states[:,:,idx],
                                 key_states[:,:,idx])
      for idx in range(query_states.get_shape()[2])], 2)

def time_delay_aggregation(attn_weights,
                           value_states,
                           autocorrelation_factor=2):
  """
  Computes aggregation as
  value_states.roll(delay) * top_k_autocorrelations(delay).
  The final result is the autocorrelation-attention output.
  Think about it as a replacement of the dot-product between attn_weights
  and value states.
  
  The autocorrelation_factor is used to find top k autocorrelations delays.
  Assumption: value_states and attn_weights shape:
  [batch_size, time_length, embedding_dim]
  """

  print("https://huggingface.co/blog/autoformer")
