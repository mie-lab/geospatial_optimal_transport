TRAINTEST_SPLIT = 0.9
TEST_SAMPLES = 50  # number of time points where we start a prediction
STEPS_AHEAD = 3
MAX_COUNT = 1000  # how many rentals we expect maximally

CONFIG = {
    "bikes": {
        "model": "nhits",
        "multi_vs_ind": "multi",
        "lags": 24,
        "output_chunk_length": 3,
        "num_stacks": 3,
        "lags_past_covariates": 1,
        "x_scale": 1,
        "reconcile": 0,
    },
    "charging": {
        "model": "linear",
        "multi_vs_ind": "ind",
        "lags": 48,
        "output_chunk_length": 3,
        "lags_past_covariates": 1,
        "x_scale": 1,
        "reconcile": 0,
    },
    "carsharing": {
        "model": "lightgbm",
        "multi_vs_ind": "multi",
        "lags": 12,
        "output_chunk_length": 6,
        "lags_past_covariates": 1,
        "x_scale": 1,
        "reconcile": 0,
    },
}
