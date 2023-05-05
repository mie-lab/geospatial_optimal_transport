from darts.models import (
    LinearRegressionModel,
    XGBModel,
    NHiTSModel,
    Croston,
    LightGBMModel,
)

TRAIN_CUTOFF = 0.9
TEST_SAMPLES = 50  # number of time points where we start a prediction
STEPS_AHEAD = 3
MAX_RENTALS = 1000  # how many rentals we expect maximally

model_class_dict = {
    "linear-5": (LinearRegressionModel, {"lags": 5}),
    "linear-lags": (
        LinearRegressionModel,
        {"lags": [-1, -2, -3, -4, -5, -24, -128]},
    ),
    "xgb-5": (XGBModel, {"lags": 5}),
    "xgb-lags": (XGBModel, {"lags": [-1, -2, -3, -4, -5, -24, -128]}),
    "nhits-50e-5i": (
        NHiTSModel,
        {"input_chunk_length": 5, "output_chunk_length": 3, "n_epochs": 50},
    ),
    "nhits-100e-5i": (
        NHiTSModel,
        {"input_chunk_length": 5, "output_chunk_length": 3, "n_epochs": 50},
    ),
    "lightgbm-5": (LightGBMModel, {"lags": 5}),
    "croston": (Croston, {}),
}
