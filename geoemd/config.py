TRAINTEST_SPLIT = 0.9
TEST_SAMPLES = 50  # number of time points where we start a prediction
STEPS_AHEAD = 3
MAX_COUNT = 1000  # how many rentals we expect maximally
TRAINVAL_SPLIT = 0.8

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
        # "model": "lightgbm",
        # "multi_vs_ind": "multi",
        # "lags": 12,
        # "output_chunk_length": 6,
        # "lags_past_covariates": 1,
        "model": "linear",
        "multi_vs_ind": "ind",
        "lags": 12,
        "output_chunk_length": 3,
        "lags_past_covariates": 0,
        "x_scale": 1,
        "reconcile": 0,
        "nhits_model_kwargs": {
            "input_chunk_length": 65,  # self.model_args["lags"],
            "n_epochs": 200,  # self.model_args["n_epochs"],
            "num_stacks": 8,  # self.model_args["num_stacks"],
            "optimizer_kwargs": {"lr": 0.000272},
            "dropout": 0.03988,
            "num_blocks": 1,
            "layer_widths": [256, 128, 128, 512, 128, 512, 512, 512],
            "num_layers": 4,
            "pooling_kernel_sizes": [[8], [8], [4], [4], [2], [2], [1], [1]],
            "n_freq_downsample": [[8], [8], [4], [4], [2], [2], [1], [1]],
            "activation": "ReLU",
            "MaxPool1d": True,
            "output_chunk_length": 48,
            "log_tensorboard": True,
        },
    },
}


STATION_PATH = {
    "bikes": "data/bikes/test_stations.csv",
    "bikes_2015": "data/bikes/stations_2015.csv",
    "carsharing": "data/carsharing/zurich_stations.csv",
    "charging": "data/charging/stations.csv",
}

DATA_PATH = {
    "bikes": "data/bikes/test_pickup.csv",
    "bikes_2015": "data/bikes/data_2015.csv",
    "carsharing": "data/carsharing/zurich_data.csv",
    "charging": "data/charging/test_data.csv",
}

FREQUENCY = {
    "bikes": "1h",
    "bikes_2015": "1h",
    "charging": "15min",
    "carsharing": "30min",
}

SPEED_FACTOR = {"bikes": 10, "bikes_2015": 10, "carsharing": 10, "charging": 10}
