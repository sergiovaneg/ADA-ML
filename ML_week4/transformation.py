from gluonts.time_feature import (
    time_features_from_frequency_str,
    TimeFeature,
    get_lags_for_frequency,
)
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SelectFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
    RenameFields,
)

from transformers import PretrainedConfig

def create_transformation(freq: str,
                          config: PretrainedConfig) -> Transformation:
  remove_field_names = []
  if config.num_static_real_features == 0:
      remove_field_names.append(FieldName.FEAT_STATIC_REAL)
  if config.num_dynamic_real_features == 0:
      remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
  if config.num_static_categorical_features == 0:
      remove_field_names.append(FieldName.FEAT_STATIC_CAT)

  # a bit like torchvision.transforms.Compose
  return Chain(
      # step 1: remove static/dynamic fields if not specified
      [RemoveFields(field_names=remove_field_names)]
      # step 2: convert the data to NumPy (potentially not needed)
      + (
          [
              AsNumpyArray(
                  field=FieldName.FEAT_STATIC_CAT,
                  expected_ndim=1,
                  dtype=int,
              )
          ]
          if config.num_static_categorical_features > 0
          else []
      )
      + (
          [
              AsNumpyArray(
                  field=FieldName.FEAT_STATIC_REAL,
                  expected_ndim=1,
              )
          ]
          if config.num_static_real_features > 0
          else []
      )
      + [
          AsNumpyArray(
              field=FieldName.TARGET,
              # we expect an extra dim for the multivariate case:
              expected_ndim=1 if config.input_size == 1 else 2,
          ),
          # step 3: handle the NaN's by filling in the target with zero
          # and return the mask (which is in the observed values)
          # true for observed values, false for nan's
          # the decoder uses this mask (no loss is incurred for unobserved values)
          # see loss_weights inside the xxxForPrediction model
          AddObservedValuesIndicator(
              target_field=FieldName.TARGET,
              output_field=FieldName.OBSERVED_VALUES,
          ),
          # step 4: add temporal features based on freq of the dataset
          # month of year in the case when freq="M"
          # these serve as positional encodings
          AddTimeFeatures(
              start_field=FieldName.START,
              target_field=FieldName.TARGET,
              output_field=FieldName.FEAT_TIME,
              time_features=time_features_from_frequency_str(freq),
              pred_length=config.prediction_length,
          ),
          # step 5: add another temporal feature (just a single number)
          # tells the model where in its life the value of the time series is,
          # sort of a running counter
          AddAgeFeature(
              target_field=FieldName.TARGET,
              output_field=FieldName.FEAT_AGE,
              pred_length=config.prediction_length,
              log_scale=True,
          ),
          # step 6: vertically stack all the temporal features into the key FEAT_TIME
          VstackFeatures(
              output_field=FieldName.FEAT_TIME,
              input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
              + (
                  [FieldName.FEAT_DYNAMIC_REAL]
                  if config.num_dynamic_real_features > 0
                  else []
              ),
          ),
          # step 7: rename to match HuggingFace names
          RenameFields(
              mapping={
                  FieldName.FEAT_STATIC_CAT: "static_categorical_features",
                  FieldName.FEAT_STATIC_REAL: "static_real_features",
                  FieldName.FEAT_TIME: "time_features",
                  FieldName.TARGET: "values",
                  FieldName.OBSERVED_VALUES: "observed_mask",
              }
          ),
      ]
  )