"""
Main grid search script.
"""

import os

import pandas as pd

from typing import Optional
from pydantic import NonNegativeInt, PositiveInt

import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras.backend import clear_session

from datetime import datetime
import transformer_sv

import matplotlib.pyplot as plt
# In[1]:

if not os.path.exists("./models/"):
  os.makedirs("./models/")
if not os.path.exists("./report/figures/"):
  os.makedirs("./report/figures/")


# # Data Import

# In[2]:

solar_energy_df = pd.read_csv("../ML_week3/solarenergy.csv",
                              delimiter=",",
                              index_col=0,
                              date_format="%d/%m/%Y %H:%M",
                              parse_dates=True).sort_index()

"""
solar_energy_df["Datetime"] = pd.to_datetime(solar_energy_df["Datetime"],
                                             format="%d/%m/%Y %H:%M")
solar_energy_df = solar_energy_df.set_index("Datetime").sort_index()
"""

solar_energy_df = solar_energy_df.dropna()
solar_energy_df = solar_energy_df.resample("1H").interpolate("linear")
solar_energy_df = \
  (solar_energy_df - solar_energy_df.mean()) / solar_energy_df.std()

training_ratio = 0.7

training_limit = \
  solar_energy_df.index[
      int(training_ratio*solar_energy_df.shape[0])
    ]

y_train_df = solar_energy_df.loc[:training_limit-pd.Timedelta(hours=1),
                                  "solar_mw"]
y_test_df = solar_energy_df.loc[training_limit:, "solar_mw"]


# # Tensor Creation

# In[3]:

n_feats = 1
input_memory=48

def tensor_memory_reshaper(in_np:np.ndarray, out_np:Optional[np.ndarray],
                           memory:NonNegativeInt,
                           n_feats_internal:PositiveInt = n_feats):
  in_np = in_np.reshape((-1,1,n_feats_internal))
  for _ in range(memory):
    next_np = in_np[1:,-1,:].reshape(((-1,1,n_feats_internal)))
    in_np = np.concatenate((in_np[:-1,:,:], next_np), axis=1)

  if out_np is not None:
    return (tf.convert_to_tensor(in_np),
            tf.convert_to_tensor(out_np[memory:]))
  else:
    return (tf.convert_to_tensor(in_np), None)


# # Model instantiation

# In[4]:
class StateResetCallback(keras.callbacks.Callback):
  def on_epoch_begin(self, epoch, logs=None):
    self.model.reset_states()

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss",patience=10,
                                  restore_best_weights=True),
    StateResetCallback()
  ]

model_dict = {}
date_format = "%Y%m%d%H%M%S"

count = 0
resume = 150

try:
  for input_memory in [72,48,24]:
    x_train_np = np.pad(
        y_train_df[:-1].to_numpy().reshape((-1,1)),
        ((input_memory+1,0),(0,0)),
        mode="edge"
      )
    y_train_np = np.pad(y_train_df.to_numpy(), ((input_memory,0),), mode="edge")

    x_train_tensor,y_train_tensor = \
      tensor_memory_reshaper(x_train_np, y_train_np, input_memory)

    for head_size in [8,4,2]:
      for num_heads in [8,4,2]:
        for ff_dim in [10,5,2]:
          for mlp_units in [32,16,8]:
            if count < resume:
              count += 1
              continue

            serial = "model_" + datetime.now().strftime(date_format)

            clear_session()

            transformer_model = transformer_sv.build_model(
                input_shape=x_train_tensor.shape[1:],
                head_size=head_size,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_transformer_blocks=1,
                mlp_units=[mlp_units],
                mlp_dropout=0.5,
                dropout=0.5
              )

            transformer_model.compile(
                loss="mse",
                optimizer="Adam"
              )

            transformer_model.fit(
                x=(x_train_tensor,x_train_tensor),
                y=y_train_tensor,
                validation_split=0.2,
                epochs=100,
                shuffle=True,
                callbacks=callbacks,
                verbose=0
              )
            transformer_model.save(f"./models/{serial}.keras")

            transformer_fit = transformer_model.predict(
                x=(x_train_tensor,x_train_tensor),
                verbose=0
              ).flatten()

            transformer_prediction = np.zeros(y_test_df.shape)
            buffer = y_train_np[-(input_memory+1):].copy()
            context,_ = tensor_memory_reshaper(buffer, None, input_memory, 1)
            for idx in range(len(y_test_df)):
              current_input,_ = tensor_memory_reshaper(buffer,
                                                       None, input_memory, 1)
              transformer_prediction[idx] = \
                transformer_model.predict((context,current_input),
                                          verbose=0).flatten()[-1]

              buffer[:-1] = buffer[1:]
              buffer[-1] = transformer_prediction[idx]

            st_rmse = np.sqrt(
                np.mean(
                    (transformer_prediction[:48]-y_test_df.to_numpy()[:48])**2
                  )
              )
            transformer_rmse = np.sqrt(
                np.mean((transformer_prediction-y_test_df.to_numpy())**2)
              )
            transformer_mae = np.abs(
                transformer_prediction-y_test_df.to_numpy()
              ).max()

            model_dict[serial] = {
              "serial": serial,
              "input_memory": input_memory,
              "head_size": head_size,
              "num_heads": num_heads,
              "ff_dim": ff_dim,
              "mlp_units": mlp_units,
              "ST_RMSE": st_rmse,
              "RMSE": transformer_rmse,
              "MAE": transformer_mae
            }
            print(f"Model: {model_dict[serial]}")

            plt.figure(figsize=(8,5))
            plt.plot(solar_energy_df["solar_mw"],
                    label="Measured")
            plt.plot(y_train_df.index, transformer_fit, label="Fitted")
            plt.plot(y_test_df.index, transformer_prediction, label="Estimated")
            plt.autoscale(True, "x", tight=True)
            plt.title(f"Classic Transformer - {serial}")
            plt.legend()
            plt.savefig(f"./report/figures/{serial}_fit.svg")
            plt.close("all")
finally:
  model_df = pd.DataFrame.from_dict(list(model_dict.values()))
  model_df.to_excel(f"./results_{datetime.now().strftime(date_format)}.xlsx")
