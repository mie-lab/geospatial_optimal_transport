import pandas as pd
from darts import TimeSeries, concatenate


class LastStepModel:
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, train_data, **kwargs):
        pass

    def predict(self, n, series, past_covariates=None, **kwargs):
        # repeat the last n entries and re-index with datetimes
        out_df = pd.concat([series[-1].pd_dataframe() for _ in range(n)])
        test = series.shift(n).time_index[-n:]
        out_df.index = test
        return TimeSeries.from_dataframe(out_df, freq=series.freq)

    def save(self, *args, **kwargs):
        pass
