"""
Python module implementing the necessary layers for a classic
Transformer in Keras
"""

from pydantic import Field, PositiveInt
from typing_extensions import Annotated

from tensorflow import keras
import numpy as np
import tensorflow as tf
from keras import layers

import math

UnitFloat = Annotated[float, Field(strict=True, ge=0., le=1.)]

def positional_encoding(length:PositiveInt, depth:PositiveInt):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1)

  return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(layers.Layer):
  """
  Custom implementation of the Positional embedding layer for fixed-length
  input sequences.
  """
  def __init__(self, length:PositiveInt, d_model:PositiveInt):
    super().__init__()
    self.length = length
    self.d_model = int(2*math.ceil(float(d_model)/2))
    self.pos_encoding = positional_encoding(
                            length=self.length,
                            depth=self.d_model
                          )[tf.newaxis, :, :]

  def call(self, inputs):
    # This factor sets the relative scale of the embedding and
    # positonal_encoding.
    inputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    inputs = inputs + self.pos_encoding
    return inputs

def transformer_encoder(inputs:keras.Input,
                        head_size:PositiveInt,
                        num_heads:PositiveInt,
                        ff_dim:PositiveInt,
                        dropout:UnitFloat=0.):
  # Positional Encoding
  x = PositionalEmbedding(inputs.shape[-1], head_size)(inputs)

  # Normalization and Attention
  x = layers.MultiHeadAttention(
      key_dim=head_size, num_heads=num_heads, dropout=dropout,
    )(x,x)
  res = layers.LayerNormalization(epsilon=1e-6, axis=(1,2))(x+inputs)

  # Feed-Forward
  x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
  x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
  x = layers.Dropout(dropout)(x)

  return layers.LayerNormalization(epsilon=1e-6, axis=(1,2))(x+res)

def transformer_decoder(inputs:keras.Input,
                        context:keras.Input,
                        head_size:PositiveInt,
                        num_heads:PositiveInt,
                        ff_dim:PositiveInt,
                        dropout:UnitFloat=0.):
  # Positional Encoding
  x = PositionalEmbedding(inputs.shape[-1], head_size)(inputs)

  # Normalization and Self-Attention
  x = layers.MultiHeadAttention(
      key_dim=head_size, num_heads=num_heads, dropout=dropout,
    )(x,x,use_causal_mask=True)
  res = layers.LayerNormalization(epsilon=1e-6, axis=(1,2))(x+inputs)

  # Normalization and Cross-Attention
  x = layers.MultiHeadAttention(
      key_dim=head_size, num_heads=num_heads, dropout=dropout,
    )(res,context)
  res = layers.LayerNormalization(epsilon=1e-6, axis=(1,2))(x+inputs)

  # Feed-Forward
  x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
  x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
  x = layers.Dropout(dropout)(x)

  return layers.LayerNormalization(epsilon=1e-6, axis=(1,2))(x+res)

def build_model(input_shape:tuple[PositiveInt,PositiveInt],
                head_size:PositiveInt,
                num_heads:PositiveInt,
                ff_dim:PositiveInt,
                num_transformer_blocks:PositiveInt,
                mlp_units:tuple[PositiveInt,...],
                dropout:UnitFloat=0.,
                mlp_dropout:UnitFloat=0.):
  context = keras.Input(shape=input_shape)
  inputs = keras.Input(shape=input_shape)

  x = context
  for _ in range(num_transformer_blocks):
    x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

  y = inputs
  for _ in range(num_transformer_blocks):
    y = transformer_decoder(y, x, head_size, num_heads, ff_dim, dropout)

  y = layers.GlobalAveragePooling1D(data_format="channels_first")(y)

  for dim in mlp_units:
    y = layers.Dense(dim, activation="relu")(y)
    y = layers.Dropout(mlp_dropout)(y)
  outputs = layers.Dense(1, activation="linear")(y)

  return keras.Model([context,inputs],outputs)
