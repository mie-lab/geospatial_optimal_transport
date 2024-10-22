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
This installs the package called `geoemd` in your virtual environment, together with all dependencies. The code was tested on OS and Linux. 

### Data download and preprocessing:

We provide all preprocessed data in the [data_submission](data_submission/) folder. 

If you want to reproduce the preprocessing, please use the following instructions:

**1) Bike sharing data**

Public bike sharing data from Montreal were downloaded [here](https://www.kaggle.com/datasets/aubertsigouin/biximtl). A script is provided to read all data and to convert it into the hourly number of bike pick-ups per station:

```
python preprocessing/bikes_montreal.py --path path/to/downloaded/folder
```

The script will output the preprocessed data into the same folder.

**2) Charging occupancy data**

The charging station occupancy data were downloaded [here](here: https://gitlab.com/smarter-mobility-data-challenge/tutorials). It is taken from the `Smarter Mobility Data Challenge` (see [paper](https://arxiv.org/abs/2306.06142v1)). 

We first run the data through the [notebook]( https://github.com/arthur-75/Smarter-Mobility-Data-Challenge/blob/main/notebook/cleaning.ipynb) from the winning team of the challenge for preprocessing. We then saved the intermediate data and run it through the following script for further preprocessing steps (paths are hard-coded in the script):
```
python preprocessing/charging.py
```

**3) Traffic forecasting data**

Data was downloaded [here](https://github.com/MengzhangLI/STFGNN/tree/master/data/PEMS08). We save the data in a folder `data/traffic`. Then, we preprocess it through the following script:
```
python preprocessing/traffic.py
```

### Training models

-----------------
**NOTE**: The purpose of this repo is to demonstrate an *evaluation* framework. The goal is not to develop **new** superior models for bike sharing or other applications. Thus, the predictions themselves do not matter much and were only created to demonstrate the framework on new data. We provide our outputs [here](data_submission/predictions/). The following instructions are only necessary to retrain the models.

-----------------

We use the `darts` library for time series prediction.

```
python train_test.py [-h] [-d DATA_PATH] [-s STATION_PATH] [-o OUT_PATH] [-c CONFIG] [-m MODEL] [--multi_vs_ind MULTI_VS_IND] [-r RECONCILE] [-x HIERARCHY] [-l LAGS] [--output_chunk_length OUTPUT_CHUNK_LENGTH] [--n_epochs N_EPOCHS] [--x_loss_function X_LOSS_FUNCTION] [--x_scale X_SCALE] [--num_stacks NUM_STACKS] [--lags_past_covariates LAGS_PAST_COVARIATES] [--y_clustermethod Y_CLUSTERMETHOD][--y_cluster_k Y_CLUSTER_K] [--model_path MODEL_PATH] [--load_model_name LOAD_MODEL_NAME] [--ordered_samples] [--optimize_optuna]
```

E.g. we ran it with 

```
python scripts/train_test.py  -d data_submission/data_raw/bikes_data.csv     -s data_submission/data_raw/bikes_stations.csv    -o outputs/bikes     --model_path trained_models/bikes  --model nhits --n_epochs 100 --x_loss_function emdpartialspatial
```
This will train with a Sinkhorn loss (unbalanced OT) in the NHiTS model, and will save the model in the folder `trained_models/bikes`, and save the output in `outputs/bikes`.

### Evaluation

Our evaluation script is applied on a whole folder with the outputs from several models, and saves the results in a folder with the same name + "_plots", e.g., `outputs/bikes_plots`:

```
python scripts/evaluate.py -n bikes --redo 
```

## Reproduce results

We provide notebooks to reproduce all figures and tables from the manuscript:

* [synthetic](notebooks/synthetic_example.ipynb): This notebook provides the code for reproducing the experiments on synthetic data, including Figure 3 and Figure 4.
* [case study bike sharing](notebooks/bike_sharing_case_study.ipynb): Reproducing experiments on the application of the evaluation framework on bike sharing data (Figure 5, 6 and 7)
* [unpaired](notebooks/unpaired_ot.ipynb): This notebook reproduces the experiment in Appendix B (OT for unpaired data).
* [scales](notebooks/scales.ipynb): Reproduce the analysis on spatial and temporal scales (Figure 8, Table 2, Appendix F)
* [sinkhorn loss](notebooks/sinkhorn_loss.ipynb): Reproduce the results of training with the Sinkhorn loss with this notebook (Table 3, Figure 11) as well as the analysis in Appendix D
