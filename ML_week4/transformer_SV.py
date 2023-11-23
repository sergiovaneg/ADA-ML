"""
Python module implementing the necessary layers for a classic
Transformer in Keras
"""

from pydantic import Field, PositiveInt
from typing_extensions import Annotated

from tensorflow import keras
from keras import layers

UnitFloat = Annotated[float, Field(strict=True, ge=0., le=1.)]

def transformer_encoder(inputs:keras.Input,
                        head_size:PositiveInt,
                        num_heads:PositiveInt,
                        ff_dim:PositiveInt,
                        dropout:UnitFloat=0.):
  # Normalization and Attention
  x = layers.LayerNormalization(epsilon=1e-6)(inputs)
  x = layers.MultiHeadAttention(
      key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x,x)
  res = x + inputs

  # Feed-Forward
  x = layers.LayerNormalization(epsilon=1e-6)(res)
  x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
  x = layers.Dropout(dropout)(x)
  x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)

  return x+res

def build_model(input_shape:tuple[PositiveInt,PositiveInt],
                head_size:PositiveInt,
                num_heads:PositiveInt,
                ff_dim:PositiveInt,
                num_transformer_blocks:PositiveInt,
                mlp_units:tuple[PositiveInt,...],
                dropout:UnitFloat=0.,
                mlp_dropout:UnitFloat=0.):
  inputs = keras.Input(shape=input_shape)
  x = inputs
  for _ in range(num_transformer_blocks):
    x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

  x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
  for dim in mlp_units:
    x = layers.Dense(dim, activation="relu")(x)
    x = layers.Dropout(mlp_dropout)(x)
  outputs = layers.Dense(1, activation="linear")(x)

  return keras.Model(inputs,outputs)
