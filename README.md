# Earth mover distance for spatiotemporal predictions


### Installation:

The code can be installed via pip in editable mode in a virtual environment with the following commands:

```
git clone https://github.com/mie-lab/geospatial_optimal_transport
cd  geospatial_optimal_transport
python -m venv env
source env/bin/activate
pip install -e .
```
This installs the package called `geoemd` in your virtual environment, together with all dependencies. 

### Data download and preprocessing:

Public bike sharing data from Montreal were downloaded[here](https://www.kaggle.com/datasets/aubertsigouin/biximtl). A script is provided to read all data and to convert it into the hourly number of bike pick-ups per station:

```
python preprocessing/bikes_montreal.py --path path/to/downloaded/folder
```

The script will output the preprocessed data into the same folder.

### Train and test a model

We use the `darts` library for time series prediction.

```
python train_test.py [-h] [-d DATA_PATH] [-s STATION_PATH] [-o OUT_PATH] [-c CONFIG] [-m MODEL] [--multi_vs_ind MULTI_VS_IND] [-r RECONCILE] [-x HIERARCHY] [-l LAGS] [--output_chunk_length OUTPUT_CHUNK_LENGTH] [--n_epochs N_EPOCHS] [--x_loss_function X_LOSS_FUNCTION] [--x_scale X_SCALE] [--num_stacks NUM_STACKS] [--lags_past_covariates LAGS_PAST_COVARIATES] [--y_clustermethod Y_CLUSTERMETHOD][--y_cluster_k Y_CLUSTER_K] [--model_path MODEL_PATH] [--load_model_name LOAD_MODEL_NAME] [--ordered_samples] [--optimize_optuna]
```


