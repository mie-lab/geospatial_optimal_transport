import numpy as np
import os
import optuna
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping

from darts.dataprocessing.transformers import Scaler
from darts.metrics import mse
from darts.models import NHiTSModel


class OptunaOptimizer:
    def __init__(self, model_path, model_name, **kwargs):
        self.study = optuna.create_study(direction="minimize")
        self.work_dir = os.path.join(model_path, model_name)
        self.loss_fn = kwargs.get("loss_fn", torch.nn.MSELoss())

    def __call__(self, train_series, val_series):
        self.train_series = train_series
        self.val_series = val_series
        self.study.optimize(
            self.objective, n_trials=100, callbacks=[print_callback]
        )

    # define objective function
    def objective(self, trial):
        # select input and output chunk lengths
        in_len = trial.suggest_int("in_len", 12, 36)
        out_len = trial.suggest_int("out_len", 1, in_len - 1)

        # Other hyperparameters
        num_stacks = trial.suggest_int("num_stacks", 3, 8)
        num_blocks = trial.suggest_int("num_blocks", 1, 3)
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
        include_past_covariates = trial.suggest_categorical(
            "past_covariates", [False, True]
        )

        # throughout training we'll monitor the validation loss for both
        # pruning and early stopping
        pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
        early_stopper = EarlyStopping(
            "val_loss", min_delta=0.00, patience=3, verbose=False
        )
        callbacks = [pruner, early_stopper]

        # detect if a GPU is available
        if torch.cuda.is_available():
            num_workers = 4
        else:
            num_workers = 0

        pl_trainer_kwargs = {
            "accelerator": "auto",
            "callbacks": callbacks,
        }

        # optionally also add the (scaled) year value as a past covariate
        if include_past_covariates:
            encoders = {}
            encoders["datetime_attribute"] = {
                "past": ["hour", "day", "weekday"]
            }
            # encoders["cyclic"] = {"past": ["hour", "day", "weekday"]}
            encoders["transformer"] = Scaler()
        else:
            encoders = None

        # reproducibility
        torch.manual_seed(42)

        # build the TCN model
        model = NHiTSModel(
            input_chunk_length=in_len,
            output_chunk_length=out_len,
            batch_size=32,
            n_epochs=100,
            nr_epochs_val_period=1,
            num_stacks=num_stacks,
            num_blocks=num_blocks,
            dropout=dropout,
            work_dir=self.work_dir,
            loss_fn=self.loss_fn,
            optimizer_kwargs={"lr": lr},
            add_encoders=encoders,
            pl_trainer_kwargs=pl_trainer_kwargs,
            model_name="nhits_model",
            force_reset=True,
            save_checkpoints=True,
        )

        # train the model
        model.fit(
            series=self.train_series,
            val_series=self.val_series,
            num_loader_workers=num_workers,
        )

        # reload best model over course of training
        model = NHiTSModel.load_from_checkpoint(
            "nhits_model", work_dir=self.work_dir
        )

        # Evaluate how good it is on the validation set, using sMAPE
        preds = model.predict(series=self.train_series, n=len(self.val_series))
        errors = mse(self.val_series, preds, n_jobs=-1, verbose=True)
        error_val = np.mean(errors)

        return error_val if error_val != np.nan else float("inf")


# for convenience, print some optimization trials information
def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(
        f"Best value: {study.best_value}, Best params: {study.best_trial.params}"
    )
