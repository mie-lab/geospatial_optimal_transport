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

Public bike sharing data from Montreal were downloaded [here](https://www.kaggle.com/datasets/aubertsigouin/biximtl). A script is provided to read all data and to convert it into the hourly number of bike pick-ups per station:

```
python preprocessing/bikes_montreal.py --path path/to/downloaded/folder
```

The script will output the preprocessed data into the same folder.

Similarly, we provide a preprocessing script for the charging data (downloaded here: https://gitlab.com/smarter-mobility-data-challenge/tutorials); however, it builds up on the [notebook]( https://github.com/arthur-75/Smarter-Mobility-Data-Challenge/blob/main/notebook/cleaning.ipynb) from the winning team of the challenge:


```
python preprocessing/charging.py
```

### Train and test a model

We use the `darts` library for time series prediction.

```
python train_test.py [-h] [-d DATA_PATH] [-s STATION_PATH] [-o OUT_PATH] [-c CONFIG] [-m MODEL] [--multi_vs_ind MULTI_VS_IND] [-r RECONCILE] [-x HIERARCHY] [-l LAGS] [--output_chunk_length OUTPUT_CHUNK_LENGTH] [--n_epochs N_EPOCHS] [--x_loss_function X_LOSS_FUNCTION] [--x_scale X_SCALE] [--num_stacks NUM_STACKS] [--lags_past_covariates LAGS_PAST_COVARIATES] [--y_clustermethod Y_CLUSTERMETHOD][--y_cluster_k Y_CLUSTER_K] [--model_path MODEL_PATH] [--load_model_name LOAD_MODEL_NAME] [--ordered_samples] [--optimize_optuna]
```

E.g. we ran it with 

```
python scripts/train_test.py  -d data/bikes/test_pickup.csv     -s data/bikes/test_stations.csv     -o outputs/bikes     --model_path trained_models/bikes  --model nhits --n_epochs 100 --x_loss_function emdpartialspatial
```
This will train with a Sinkhorn loss (unbalanced OT) in the NHiTS model, and will save the model in the folder `trained_models/bikes`, and save the output in `outputs/bikes`.

### Evaluation

Our evaluation script is applied on a whole folder with the outputs from several models, and saves the results in a folder with the same name + "_plots", e.g., `outputs/bikes_plots`:

```
python scripts/evaluate.py -n bikes --redo 
```
