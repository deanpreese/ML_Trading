import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import NBEATSModel, NHiTSModel
from darts.dataprocessing.encoders import NaNLabelEncoder
from darts.utils.data.preprocessing import (
    naive_freq_resort_encoded_univ,
    naive_freq_resort_encoded_multi,
)
from darts.metrics import mape, smape
from darts.utils.timeseries_generation import linear_timeseries
from darts.dataprocessing.transformers import MissingValueImputer, Scaler
from darts.utils.data.splitters import SingleThreadUnrolledSeriesHolder
from darts.utils.data.utils import sequential_lags, sequential_slices
from darts.utils.data.slices import SlicerUnrolledInvsObj, sliced_intersection
from sklearn.model_selection import train_test_split
from darts.wrappers import HierarchyWrapper, StaticHierarchyWrapper
from darts.utils.timeseries_generation import linear_timeseries
from darts.utils.likelihood_models import gaussian_likelihood_model, student_t_likelihood_model
from darts.utils.statistics import EmpiricalDistribution

# Load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna()
    feature_cols = [col for col in data.columns if col not in ['output', 'outputC']]
    target_cols = ['output', 'outputC']

    data_multi = data[feature_cols + target_cols].copy()
    encoders = NaNLabelEncoder().train_batch(data_multi[target_cols])
    data_multi[target_cols] = encoders.transform_batch(data_multi[target_cols])

    freq_resort_encoded_multi = naive_freq_resort_encoded_multi(data_multi, target_cols)
    freq_resort_encoded_multi = sequential_lags(freq_resort_encoded_multi, target_cols, 0)

    data_multi = TimeSeries.from_dataframe(freq_resort_encoded_multi)

    X, y = data_multi.dropna().split_apart(target_cols)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    return X_train, y_train, X_test, y_test

# Model Training
def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr, likelihood_model):
    unroller = SingleThreadUnrolledSeriesHolder(X_train)
    train_split = SlicerUnrolledInvsObj(indexes=unroller.get_slices(1.0)) >> sliced_intersection(y_train)

    unrollerVal = SingleThreadUnrolledSeriesHolder(X_val)
    val_split = SlicerUnrolledInvsObj(indexes=unrollerVal.get_slices(1.0)) >> sliced_intersection(y_val)

    imputer = MissingValueImputer(imputer="last")
    X_train_imp = imputer.flatten_and_impute(train_split.current_components)
    y_train_imp = imputer.flatten_and_impute(train_split.target)

    X_val_imp = imputer.flatten_and_impute(val_split.current_components)
    y_val_imp = imputer.flatten_and_impute(val_split.target)

    scaler = Scaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)

    model.fit(
        X_train_scaled,
        y_train_imp,
        val_series=y_val_imp,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        likelihood_model=likelihood_model,
    )

    return model

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    unrollerTest = SingleThreadUnrolledSeriesHolder(X_test)
    test_split = SlicerUnrolledInvsObj(indexes=unrollerTest.get_slices(1.0)) >> sliced_intersection(y_test)

    imputer = MissingValueImputer(imputer="last")
    X_test_imp = imputer.flatten_and_impute(test_split.current_components)
    y_test_imp = imputer.flatten_and_impute(test_split.target)

    scaler = Scaler()
    X_test_scaled = scaler.fit_transform(X_test_imp)

    y_pred = model.predict(X_test_scaled)

    mape_score = mape(y_test_imp, y_pred)
    smape_score = smape(y_test_imp, y_pred)

    print(f"MAPE: {mape_score:.4f}")
    print(f"SMAPE: {smape_score:.4f}")

# Load data
X_train, y_train, X_test, y_test = load_data('stock_data.csv')

# Likelihood models
gaussian_lm = gaussian_likelihood_model()
student_t_lm = student_t_likelihood_model()

# Train N-BEATS
nbeats_model = NBEATSModel(
    input_chunk_length=X_train.slice_nsample(0).shape[-1],
    output_chunk_length=y_train.slice_nsample(0).shape[-1],
    generic_architecture=True,
    num_stacks=3,
    num_blocks=3,
    num_layers=4,
    layer_widths=512,
)

nbeats_model = train_model(
    nbeats_model,
    X_train,
    y_train,
    X_test,
    y_test.sample(frac=0.2),
    epochs=100,
    batch_size=64,
    lr=1e-3,
    likelihood_model=gaussian_lm,
)

evaluate_model(nbeats_model, X_test, y_test)

# Train N-HiTS
nhits_model = NHiTSModel(
    input_chunk_length=X_train.slice_nsample(0).shape[-1],
    output_chunk_length=y_train.slice_nsample(0).shape[-1],
    generic_architecture=True,
    num_stacks=3,
    num_blocks=3,
    num_layers=4,
    layer_widths=512,
    num_levels=3,
)

nhits_model = train_model(
    nhits_model,
    X_train,
    y_train,
    X_test,
    y_test.sample(frac=0.2),
    epochs=100,
    batch_size=64,
    lr=1e-3,
    likelihood_model=student_t_lm,
)

evaluate_model(nhits_model, X_test, y_test)