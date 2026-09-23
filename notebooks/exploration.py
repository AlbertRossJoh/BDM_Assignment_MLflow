import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data Pipelines & Data Analytics Lifecycle
    # Forecasting the Wind Power Production in Orkney

    ## Introduction

    In this assignment you will play the role of a Data Scientist consultant for the Scottish and Southern Electricity Networks (SSEN). Among many other areas, the SSEN provides electricity for the Okrney islands, in Northern Scottland. This archipelago has significant wind and marine energy resources, and it generates over 100% of its
    net power from renewable sources, coming mainly from wind turbines situated across Orkney. This is good news for Orkney and the environment, but although wind farms provide emissions-free energy, they only generate electricity when the wind blows.

    Your job? To design and implement a training pipeline, including data preprocessing, evaluation and model storing, for a small wind energy forecasting system. The goal here is to use weather forecasting data to predict the energy production for Orkney. You will need all your data scientist skills for insights, and you will have
    some libraries at your disposal, such as Scikit-Learn and MLFlow (and as many others as you might need). This Jupyter Notebook will help you started.

    The steps you will complete in this project are:

    - align the datasets
    - create pipelines for preprocessing and prediction
    - compare your pipelines with MLflow Tracking
    - register your best model with MLflow Registry
    - serve the best model
    - package the project using MLProjects

    Above this cell you will find an *Outline* button. Click it and an outline of this notebook should appear on the left bottom corner of the screen. This will help you navigate to a specific section.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Dependencies**

    If you've just created this workspace, install mlflow by typing the command below in the Terminal. If you already did this in a previous session, there is no need to install it again.

    ```
    pip install mlflow
    ```

    Now let's import all libraries necessary for this project.
    The first time you will run a cell in this notebook, a dialogue box will appear asking if you want to Install/enable suggested extentions: python and jupyter. Go ahead and once that is finished select the created kernel.
    """)
    return


@app.cell
def _():
    # Cell tags: imports
    import mlflow
    from mlflow.tracking import MlflowClient

    # You will probably need these
    import pandas as pd
    import numpy as np
    import seaborn as sns
    from sklearn.pipeline import Pipeline

    # This are for example purposes. You may discard them if you don't use them.
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error
    from sklearn.base import clone
    from mlflow.models import infer_signature
    from urllib.parse import urlparse

    ### TODO -> HERE YOU CAN ADD ANY OTHER LIBRARIES YOU MAY NEED ###
    import polars as pl
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    from sklearn.base import clone
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import r2_score
    from sklearn.ensemble import HistGradientBoostingRegressor

    return (
        HistGradientBoostingRegressor,
        LinearRegression,
        MlflowClient,
        Pipeline,
        StandardScaler,
        TimeSeriesSplit,
        clone,
        mean_absolute_error,
        mlflow,
        np,
        pd,
        pl,
        plt,
        r2_score,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="data"></a>

    ## Data exploration

    You will work with three datasets:

    **Power generation data** (`power.csv`)

    * *Source:* Scottish and Southern Electricity Networks (SSEN)
    * *Granularity:* 1-minute sampling
    * *Target variable:* Total renewable power generation (MW)

    Key fields:

    * `time`: Timestamp of measurement
    * `Total`: Renewable power generation (MW)

    **Weather forecast data** (`weather.csv`)

    * *Source:* UK Met Office
    * *Granularity:* 3-hour intervals
    * Forecasts include a *source time* (forecast creation) and a *target time* (forecasted timestamp)

    Key fields:

    * `time`: Target time of the forecast
    * `Speed`: Wind speed (m/s)
    * `Direction`: Wind direction (categorical string, e.g. "NW")
    * `Source_time`: Forecast generation time
    * `Lead_hours`: Forecast horizon

    **Future forecasts** (`future.csv`)

    * Weather forecasts to be used for generating future power predictions using your model

    Let's first load the first 2 datasets.
    """)
    return


@app.cell
def _(pd, pl):
    # Cell tags: data
    power_df = pl.from_pandas(
        pd.read_csv("data/power.csv", parse_dates=["time"]),
        include_index=False,
    )
    wind_df = pl.from_pandas(
        pd.read_csv("data/weather.csv", parse_dates=["time"]),
        include_index=False,
    )
    wind_df = wind_df.with_columns(
        pl.from_epoch(pl.col("Source_time")).alias("Source_time")
    )
    return power_df, wind_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's take a look at the resulting dataframes:
    """)
    return


@app.cell
def _(pl, power_df):
    power_df.with_columns((pl.col("ANM") + pl.col("Non-ANM")).alias("Sum"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This table contains three columns, but closer inspection will reveal a clear relationship between those three. Can you spot that?

    We are interested in the total power generation, regardless of the source type. Modify the table such that only the column of interest is kept.
    """)
    return


@app.cell
def _(power_df):
    ### TODO -> REMOVE UNNECESSARY COLUMNS ###
    power_df_select = power_df.select(["time", "Total"])
    power_df_select
    return (power_df_select,)


@app.cell
def _(wind_df):
    wind_df_needed_columns = wind_df.select(["time", "Direction", "Speed"])
    return (wind_df_needed_columns,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This table contains four columns, but lead hours and source time are irrelevant here. Can you think why?

    It also contains a column named Direction, which shows the cardinal direction that the wind is blowing from. For example, the record containing ’SW’ in the wind direction field would indicate that the wind is blowing from the South-West (and to
    the North-East). Since our models can only handle numeric data, you will need to perform some transformation on this feature later on.

    Look at the table's index. Do both data sources contain the same intervals? And if not, what problems could arise when merging the data?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    >This table contains four columns, but lead hours and source time are irrelevant here. Can you think why?

    The source time is the time when the forecase was generated. The lead hours is how many hours before the event the forecast was made.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data alignment
    We have two dataframes, one with weather forecasts and one with power generation. To do some analysis on the relationship between these two datasets, it might be useful to align the datasets.

    **Example**
    """)
    return


@app.cell
def _():
    # Joining the data
    # joined_dfs = power_df.join(wind_df, how="inner")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Joining the two datasets with an inner join means keeping only those records that match their index. Although this will work, you may notice that most of our data is discarded due to the unmatching time intervals. You may want to explore other possible ways to merge the data; you can argue for using a specific method or experiment with both later on. For now, try out a different way to align the datasets.
    """)
    return


@app.cell
def _(pl, power_df_select, wind_df_needed_columns):
    ### TODO -> JOIN THE TWO DATASETS ###
    def resample(granularity="1h"):
        wind_df_upsampled = wind_df_needed_columns.upsample(
            time_column="time", every=granularity
        ).with_columns(
            pl.col("Direction").fill_null(strategy="forward"),
            pl.col("Speed").fill_null(strategy="forward"),
        )
        power_df_downsampled = power_df_select.with_columns(
            pl.col("time").dt.round(granularity)
        )
        return power_df_downsampled.join(
            wind_df_upsampled, how="inner", on="time"
        )

    joined_dfs = resample()
    return joined_dfs, resample


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Understanding the data: EDA

    It can be also useful to plot our datasets to see what relationships they might hold.
    """)
    return


@app.cell
def _(joined_dfs, np, pl, plt):
    # Calculate required rows for the last 7 days
    n_rows = int(7 * 24 / 3)

    # Subplots
    fig, ax = plt.subplots(1, 3, figsize=(25, 4))

    # 1. Speed and Power for the last 7 days
    tail_df = joined_dfs.tail(n_rows)
    # If your index/time axis was a column, pass that to ax[0].plot(x, y)
    ax[0].plot(tail_df["Speed"], label="Speed", color="blue")
    ax[0].plot(tail_df["Total"], label="Power", color="tab:red")
    ax[0].set_title("Windspeed & Power Generation over last 7 days")
    ax[0].set_xlabel("Time")
    ax[0].tick_params(axis="x", labelrotation=45)
    ax[0].set_ylabel("Windspeed [m/s], Power [MW]")
    ax[0].legend()

    # 2. Speed vs Total (Power Curve nature)
    ax[1].scatter(joined_dfs["Speed"], joined_dfs["Total"])

    # Group by Speed and get median Total (sort for clean line plotting)
    power_curve = (
        joined_dfs.group_by("Speed")
        .agg(pl.col("Total").median())
        .sort("Speed")
    )
    ax[1].plot(
        power_curve["Speed"], power_curve["Total"], "k:", label="Power Curve"
    )
    ax[1].legend()
    ax[1].set_title("Windspeed vs Power")
    ax[1].set_ylabel("Power [MW]")
    ax[1].set_xlabel("Windspeed [m/s]")

    # 3. Speed and Power per Wind Direction
    wind_grouped_by_direction = joined_dfs.group_by("Direction").agg(
        pl.col("Total").mean(), pl.col("Speed").mean()
    )

    bar_width = (
        0.35  # Adjusted slightly so side-by-side bars don't overlap completely
    )
    x = np.arange(len(wind_grouped_by_direction["Direction"]))

    ax[2].bar(
        x,
        wind_grouped_by_direction["Total"],
        width=bar_width,
        label="Power",
        color="tab:red",
    )
    ax[2].bar(
        x + bar_width,
        wind_grouped_by_direction["Speed"],
        width=bar_width,
        label="Speed",
        color="blue",
    )
    ax[2].legend()
    ax[2].set_xticks(x + bar_width / 2)
    ax[2].set_xticklabels(wind_grouped_by_direction["Direction"])
    ax[2].tick_params(axis="x", labelrotation=45)
    ax[2].set_title("Speed and Power per Direction")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    These plots should already give us an intuition of the different relationships between features. There is a clear relationship between the wind speed and the power generation from the turbines. But that relationship is not completely linear. Which plot shows that? Finally, it seems like the power generation also depends of where the wind is coming from. Maybe this could also be a useful feature.

    In order to plot the relationship between wind speed and power generation we have performed a very simple join with the two datasets. But since the intervals are not the same, a lot of data is discarded (can you spot where in the code this happens?).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    I already fixed the data
    """)
    return


@app.cell
def _():
    ### TODO -> DO SOME EXTRA EDA ON THE DATA ###
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pipeline and data transformations

    Now that we have our data, we need to construct the pipeline to process this data and pass it to our Machine Learning model. For this, use the Pipeline class from Scikit-Learn.

    This class applies a list of transformations to your data, and pass the final state to an estimator (your model). Intermediate steps of the pipeline must be ‘transforms’, that is, they must implement fit and transform methods. The final estimator only needs to implement fit.

    You can find more information about Scikit-Learn's Pipeline [here](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).

    **Example**
    """)
    return


@app.cell
def _(LinearRegression, Pipeline, StandardScaler):
    # A very basic pipeline
    pipeline_example = Pipeline(
        [
            # Transformations
            ("Scaler", StandardScaler()),
            # Estimator
            ("Linear Regression", LinearRegression()),
        ]
    )
    return (pipeline_example,)


@app.cell
def _(LinearRegression, Pipeline, StandardScaler):
    ### TODO -> CREATE YOUR OWN PIPELINE ###
    # Create your pipeline with the desired transformers
    pipeline = Pipeline(
        [
            ("Scaler", StandardScaler()),
            ("Linear Regression", LinearRegression()),
        ]
    )
    return (pipeline,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Evaluate your model

    Now that you have a preprocessing pipeline ready, along with the final estimator, you may want to know how well your model performs. Choose the method you prefer, with special attention to the selected metric.

    **Example**
    """)
    return


@app.cell
def _(joined_dfs):
    joined_dfs_1 = joined_dfs
    X = joined_dfs_1.select("Speed")
    y = joined_dfs_1.select("Total")
    return X, y


@app.cell
def _(X, mean_absolute_error, pipeline_example, train_test_split, y):
    # joined_dfs_1 = power_df.join(wind_df).dropna()
    def regular_train_test_split():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, shuffle=False
        )
        model = pipeline_example.fit(X_train, y_train)

        _mae = mean_absolute_error(pipeline_example.predict(X_test), y_test)
        print(_mae)
        model.score(X_test, y_test)

    regular_train_test_split()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's see how our predictions look compared to the true values.
    """)
    return


@app.cell
def _():
    # _predictions = pipeline.predict(X_test)
    # plt.figure(figsize=(15, 4))
    # plt.plot(np.arange(len(_predictions)), _predictions, label='Predictions')
    # plt.plot(np.arange(len(y_test)), y_test, label='Truth')
    # plt.legend()
    # plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    HINT: Pay special attention to this type of data: We are dealing with Time series data (i.e. data that is recorded over consistent intervals of time). It might be a good idea not to randomly split the data, since it wouldn't respect the temporal order and may cause data-leakage, unintentionally inferring the trend of future samples.
    """)
    return


@app.cell
def _():
    # Use your preferred method to evaluate your model
    ### TODO -> SPLIT THE DATA INTO TRAIN AND TEST SETS, AND EVALUATE YOUR MODEL ###
    return


@app.cell
def _(TimeSeriesSplit, X, mean_absolute_error, pipeline, y):
    def time_series_split():
        tscv = TimeSeriesSplit(n_splits=5)

        mae_scores = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            score = mean_absolute_error(y_test, y_pred)
            mae_scores.append(score)
            accurracy = pipeline.score(X_test, y_test)

            print(
                f"Fold {fold + 1} MAE: {score:.4f} Accurracy: {accurracy * 100:.2f}%"
            )

    # time_series_split()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tracking your experiments with MLFlow
    We have a working model with a certain accuracy. But wouldn't it be better to try different parameters and different models before deciding for one?

    This is exactly what we will do using the MLFlow library. MLflow is an open source platform to manage the ML lifecycle, including experimentation, reproducibility, deployment, and a central model registry. This will allow us for easy comparison of all our model experiments.

    Example
    When using MLFlow locally to log our experiments, we need to start a "local server". We can do this easily by running the following in our command line interface (in the Terminal):

    ```
    mlflow server --host 127.0.0.1 --port 5000
    ```

    Once you start the server you will see the following information in the Terminal: `Uvicorn running on http://127.0.0.1:5000`.

    The server is located at our localhost `127.0.0.1, port 5000`. We will use this information to indicate where our experiment logs will be shown.
    """)
    return


@app.cell
def _(mlflow):
    ## Start an MLflow run
    mlflow.sklearn.autolog()  # This is to help us track scikit learn metrics.
    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000"
    )  # We set the MLFlow UI to display in our local host.
    # experiment_name = 'LinearRegression-Example'
    ## Set the experiment and run name
    # run_name = 'Simple_regression'  # Think how to best organise experiments - for example by model type
    # mlflow.set_experiment(experiment_name)  # Give explicit names
    # with mlflow.start_run(run_name=run_name) as run:
    #    pipeline.fit(X_train, y_train)
    #    _predictions = pipeline.predict(X_test)
    #    _mae = mean_absolute_error(_predictions, y_test)
    #    mlflow.log_metric('MAE', _mae)  # Train our model  # Evaluate the model, using MAE as a metric
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the example above you can see that:
    - we are using `sklearn.autolog()` to automatically log `scikit-learn` specific metrics
    - we set an experiment_name; if the experiment doesn't exist it will create a new one, otherwise it will use the already existent experiment
    - we set a run_name; same as above
    - after model training we record the MAE to compare between runs

    Click on either of the links above to see your experiments. Have a look at the Overview, Model metrics, and Artifacts.

    Now have a look at the *Explorer* tab on the left and notice the newly created files and folders:
    - `mlflow.db` - a database containing metadata regarding experiments, runs, and parameters that you logged
    - `mlartifacts/` folders
        - each folder will correspond to a run
        - the `models` folder contains data about the logged models, such as the envinroment file, a .pkl file containing the model, and a MLmodel complete description of the model; have a look at each of those
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now try out different set-ups. You can experiment with different pre-processing steps, models, and parameters. If you need a tutorial on MLflow Tracking you can find one [here](https://mlflow.org/docs/latest/ml/tracking/).
    """)
    return


@app.cell
def _(mlflow):
    def already_run(experiment_name, run_name):
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return False
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
        )
        return not runs.empty

    return (already_run,)


@app.cell
def _(
    X,
    already_run,
    mean_absolute_error,
    mlflow,
    pipeline,
    train_test_split,
    y,
):
    def regular_split():
        experiment_name = "Regular split experiment"
        run_name = "Simple_regression"
        if already_run(experiment_name, run_name):
            return
        mlflow.set_experiment(experiment_name)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, shuffle=False
        )

        with mlflow.start_run(run_name=f"Regular split") as run:
            model = pipeline.fit(X_train, y_train)

            _mae = mean_absolute_error(pipeline.predict(X_test), y_test)
            accurracy = model.score(X_test, y_test)
            mlflow.log_metric("MAE", _mae)
            mlflow.log_metric("accurracy", accurracy)

    regular_split()
    return


@app.cell
def _(
    MlflowClient,
    TimeSeriesSplit,
    X,
    already_run,
    clone,
    mean_absolute_error,
    mlflow,
    np,
    pd,
    pipeline,
    plt,
    r2_score,
    y,
):
    def _index(arr, idx):
        """Positional row selection that works for ndarrays, DataFrames, and Series."""
        return arr.iloc[idx] if hasattr(arr, "iloc") else arr[idx]

    def _flat_hyperparams(estimator):
        """Only scalar/JSON-friendly hyperparameters — drop nested estimator/transformer objects."""
        return {
            k: v
            for k, v in estimator.get_params(deep=True).items()
            if not hasattr(v, "get_params")
        }

    def mlflow_timeseriessplit(
        X,
        y,
        pipeline,
        experiment,
        run_name,
        *,
        train_transform=None,
        n_splits=5,
        registered_model_name=None,
    ):
        if already_run(experiment, run_name):
            return

        mlflow.set_experiment(experiment)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        client = MlflowClient()
        registered_model_name = registered_model_name or f"{run_name}_pipeline"

        fold_results = []

        with mlflow.start_run(run_name=run_name) as parent_run:
            mlflow.log_param("n_splits", n_splits)
            mlflow.log_params(_flat_hyperparams(pipeline))

            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_train, y_train = _index(X, train_idx), _index(y, train_idx)
                X_test, y_test = _index(X, test_idx), _index(y, test_idx)

                if train_transform is not None:
                    X_train, y_train = train_transform(X_train, y_train)

                fold_pipeline = clone(pipeline)

                with mlflow.start_run(
                    run_name=f"fold_{fold}", nested=True
                ) as run:
                    mlflow.set_tags(
                        {"fold": fold, "cv_strategy": "TimeSeriesSplit"}
                    )

                    fold_pipeline.fit(X_train, y_train)
                    y_pred = fold_pipeline.predict(X_test)

                    y_true_np = np.asarray(y_test).ravel()
                    y_pred_np = np.asarray(y_pred).ravel()

                    fig, ax = plt.subplots(figsize=(15, 4))
                    ax.plot(
                        np.arange(len(y_pred_np)),
                        y_pred_np,
                        label="Predictions",
                    )
                    ax.plot(
                        np.arange(len(y_true_np)), y_true_np, label="Truth"
                    )
                    ax.legend()
                    ax.set_title(f"Fold {fold}: Predictions vs Truth")

                    mlflow.log_figure(
                        fig, f"predictions_vs_truth_fold_{fold}.png"
                    )
                    plt.close(fig)

                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    mlflow.log_metrics({"mae": mae, "r2": r2})
                    # also log on the parent run, keyed by step, for a fold-over-fold chart
                    mlflow.log_metric("fold_mae", mae, step=fold)
                    mlflow.log_metric("fold_r2", r2, step=fold)

                    mlflow.sklearn.log_model(
                        fold_pipeline,
                        "model",
                        serialization_format="cloudpickle",
                    )
                    print(f"Fold {fold + 1} MAE: {mae:.4f}  R²: {r2:.4f}")

                    fold_results.append((fold, mae, r2, run.info.run_id))

            mae_scores = [r[1] for r in fold_results]
            r2_scores = [r[2] for r in fold_results]

            for prefix, scores in [("mae", mae_scores), ("r2", r2_scores)]:
                stats = pd.Series(scores).describe()
                mlflow.log_metrics(
                    {
                        f"{prefix}_{name.replace('%', 'pct')}": value
                        for name, value in stats.items()
                        if name != "count"
                    }
                )

            best_fold, best_mae, best_r2, best_run_id = min(
                fold_results, key=lambda r: r[1]
            )
            mlflow.log_metric("mae_best", best_mae)
            mlflow.set_tag("best_fold", best_fold)

            model_info = mlflow.register_model(
                model_uri=f"runs:/{best_run_id}/model",
                name=registered_model_name,
            )
            client.update_model_version(
                name=registered_model_name,
                version=model_info.version,
                description=f"Best fold {best_fold} — MAE={best_mae:.4f}, R²={best_r2:.4f}",
            )
            client.set_model_version_tag(
                registered_model_name,
                model_info.version,
                "fold",
                str(best_fold),
            )
            client.set_model_version_tag(
                registered_model_name,
                model_info.version,
                "source_run_id",
                best_run_id,
            )

        return mae_scores, best_fold, best_mae

    mlflow_timeseriessplit(
        X, y, pipeline, "Time Series Split Experiment", "TSCV_parent"
    )
    return (mlflow_timeseriessplit,)


@app.cell
def _(np):
    def remove_outliers_tukey(X_train, y_train, k: float = 1.5):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        Q1 = np.percentile(X_train, 25, axis=0)
        Q3 = np.percentile(X_train, 75, axis=0)
        IQR = Q3 - Q1
        lower, upper = Q1 - k * IQR, Q3 + k * IQR

        mask = ((X_train >= lower) & (X_train <= upper)).all(axis=1)
        return X_train[mask], y_train[mask]

    return (remove_outliers_tukey,)


@app.cell
def _(X, mlflow_timeseriessplit, pipeline, remove_outliers_tukey, y):
    mlflow_timeseriessplit(
        X,
        y,
        pipeline,
        "Remove Outliers Experiment",
        "Outliers parent",
        train_transform=remove_outliers_tukey,
    )
    return


@app.cell
def _(
    LinearRegression,
    Pipeline,
    X,
    mlflow_timeseriessplit,
    remove_outliers_tukey,
    y,
):
    from sklearn.preprocessing import MaxAbsScaler

    pipeline_maxabsscaler = Pipeline(
        [
            ("MaxScaling", MaxAbsScaler()),
            ("Linear Regression", LinearRegression()),
        ]
    )

    mlflow_timeseriessplit(
        X,
        y,
        pipeline_maxabsscaler,
        "Max abs scaling",
        "Max scaling",
        train_transform=remove_outliers_tukey,
    )
    return (MaxAbsScaler,)


@app.cell
def _(LinearRegression, Pipeline, X, mlflow_timeseriessplit, y):
    from sklearn.preprocessing import RobustScaler

    pipeline_robust = Pipeline(
        [
            ("MaxScaling", RobustScaler()),
            ("Linear Regression", LinearRegression()),
        ]
    )

    mlflow_timeseriessplit(
        X, y, pipeline_robust, "Robust scaling", "Robust scaling"
    )
    return (RobustScaler,)


@app.cell
def _(
    LinearRegression,
    Pipeline,
    StandardScaler,
    joined_dfs,
    mlflow_timeseriessplit,
    pl,
    y,
):
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.compose import ColumnTransformer

    COMPASS_16 = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    ]
    DIR_TO_DEG = {d: i * 22.5 for i, d in enumerate(COMPASS_16)}

    def add_cyclic_direction(df, col="Direction"):
        deg = df[col].replace(DIR_TO_DEG).cast(pl.Float64)
        rad = deg.radians()
        return df.with_columns(
            direction_sin=rad.sin(),
            direction_cos=rad.cos(),
        )

    X_with_dir = add_cyclic_direction(joined_dfs.select(["Speed", "Direction"])).drop("Direction")

    pipeline_with_dir = Pipeline(
        [
            ("Scaling", StandardScaler()),
            ("Linear Regression", LinearRegression()),
        ]
    )

    mlflow_timeseriessplit(
        X_with_dir,
        y,
        pipeline_with_dir,
        "Direction encoding",
        "Linear",
    )
    return X_with_dir, add_cyclic_direction


@app.cell
def _(
    HistGradientBoostingRegressor,
    Pipeline,
    StandardScaler,
    X_with_dir,
    add_cyclic_direction,
    joined_dfs,
    mlflow_timeseriessplit,
    y,
):
    pipeline_gradient_boosting = Pipeline(
        [
            ("Scaling", StandardScaler()),
            ("Gradient boosting", HistGradientBoostingRegressor()),
        ]
    )

    mlflow_timeseriessplit(
        X_with_dir,
        y,
        pipeline_gradient_boosting,
        "Direction encoding",
        "Gradient boosting",
    )

    pipeline_gradient_boosting_categorical = Pipeline(
        [
            #("Scaling", StandardScaler()),
            ("Gradient boosting categorical", HistGradientBoostingRegressor(categorical_features=["Direction"])),
        ]
    )

    mlflow_timeseriessplit(
        add_cyclic_direction(joined_dfs.select(["Speed", "Direction"])),
        y,
        pipeline_gradient_boosting_categorical,
        "Direction encoding",
        "Gradient boosting categorical",
    )

    mlflow_timeseriessplit(
        joined_dfs.select(["Speed", "Direction"]),
        y,
        pipeline_gradient_boosting_categorical,
        "Direction encoding",
        "Gradient boosting categorical no sin and cos",
    )
    return pipeline_gradient_boosting, pipeline_gradient_boosting_categorical


@app.cell
def _(Pipeline, StandardScaler, X_with_dir, mlflow_timeseriessplit, y):
    from sklearn.ensemble import RandomForestRegressor

    pipeline_random_forest = Pipeline(
        [
            ("Scaling", StandardScaler()),
            ("Random forest", RandomForestRegressor()),
        ]
    )

    mlflow_timeseriessplit(
        X_with_dir,
        y,
        pipeline_random_forest,
        "Direction encoding",
        "Random forest",
    )
    return


@app.cell
def _(
    HistGradientBoostingRegressor,
    Pipeline,
    StandardScaler,
    X_with_dir,
    mlflow_timeseriessplit,
    y,
):
    pipeline_gradient_boosting_tuned = Pipeline(
        [
            ("Scaling", StandardScaler()),
            ("Gradient boosting", HistGradientBoostingRegressor()),
        ]
    )

    mlflow_timeseriessplit(
        X_with_dir,
        y,
        pipeline_gradient_boosting_tuned,
        "Direction encoding",
        "Gradient boosting tuned",
    )
    return


@app.cell
def _(
    X_with_dir,
    add_cyclic_direction,
    mlflow_timeseriessplit,
    pipeline_gradient_boosting,
    resample,
    y,
):
    joined_dfs_minute = resample("1m")
    y_min = joined_dfs_minute.select("Total")
    X_with_dir_min = joined_dfs_minute.select(["Speed", "Direction"])
    X_with_dir_min = add_cyclic_direction(X_with_dir_min)

    mlflow_timeseriessplit(
        X_with_dir,
        y,
        pipeline_gradient_boosting,
        "Minute granularity vs Hour granularity",
        "Minute",
    )
    mlflow_timeseriessplit(
        X_with_dir_min,
        y_min,
        pipeline_gradient_boosting,
        "Minute granularity vs Hour granularity",
        "Hour",
    )
    return X_with_dir_min, y_min


@app.cell
def _(
    HistGradientBoostingRegressor,
    MaxAbsScaler,
    Pipeline,
    RobustScaler,
    StandardScaler,
    X_with_dir_min,
    mlflow_timeseriessplit,
    y_min,
):
    from sklearn.preprocessing import (
        Normalizer,
        QuantileTransformer,
        PowerTransformer,
        MinMaxScaler,
    )

    pipeline_robust_scaler = Pipeline(
        [
            ("scaling", RobustScaler()),
            ("pred", HistGradientBoostingRegressor()),
        ]
    )
    pipeline_normalized = Pipeline(
        [("norm", Normalizer()), ("pred", HistGradientBoostingRegressor())]
    )
    pipeline_quantile = Pipeline(
        [
            ("quantile", QuantileTransformer()),
            ("pred", HistGradientBoostingRegressor()),
        ]
    )
    transformers = [
        RobustScaler(),
        Normalizer(),
        QuantileTransformer(),
        PowerTransformer(),
        MaxAbsScaler(),
        MinMaxScaler(),
        StandardScaler(),
    ]
    for transformer in transformers:
        pipe = Pipeline([
            ("transform", transformer),
            ("pred", HistGradientBoostingRegressor())
        ])
        mlflow_timeseriessplit(X_with_dir_min, y_min, pipe, "Scaler testing", transformer.__class__.__name__)
    return


@app.cell
def _(
    joined_dfs,
    mlflow_timeseriessplit,
    pipeline_gradient_boosting_categorical,
    pl,
    y,
):
    mlflow_timeseriessplit(
        joined_dfs.select(["time", "Speed", "Direction"]),
        y,
        pipeline_gradient_boosting_categorical,
        "Time feature engineering",
        "Regular time",
    )

    feature_engineering = joined_dfs.select(["time", "Speed", "Direction"]).with_columns(
        pl.col("time").dt.hour().alias("hour"),
        pl.col("time").dt.ordinal_day().alias("doy"),
    ).drop("time")
    mlflow_timeseriessplit(
        feature_engineering,
        y,
        pipeline_gradient_boosting_categorical,
        "Time feature engineering",
        "Time split into features",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparing models

    Suppose you have ran some experiments, trying different pre-processing steps, models, and parameters. To easily compare the results from our experiments, we can use the MLFlow interface.

    **Example.**
    We have been logging our experiments in our local server. We can access the UI by opening the localhost address in any browser. You should see a list of all the experiments under *Experiments*. When logging multiple runs with different parameters/metrics, you will be able to easily compare them using the "Chart View". Make sure to try this out.

    ## Registering the best model

    Now suppose you have tried many different models with different parameters, you might want to register the best one. To do this, we can use again the UI. You can follow the instructions below or this [visual guide](register_model_guide.pdf) which you can open in the repository:

    - In the Experiments list, select the name of the experiment you want.
    - Click Models in the left panel.
    - Click on the model you want to use, in this case it's named "model"
    - Then "Register Model".
    - First "Select a model", click "Create New Model" and name it "LinearRegression"

    Now, if you go to the Homepage, on the "Models" tab, you will see that you registered the model.

    In case of doubt, you can check the [documentation](https://mlflow.org/docs/latest/ml/model-registry/tutorial/).

    **Using a registered model**

    Now that we have registered our best model, we can use it for future predictions. We can retrieve the weather forecasts for the next days, and make predictions of power generation.

    **Example.**
    First we need the new forecast data:
    """)
    return


@app.cell
def _(pd):
    future_df = pd.read_csv("data/future.csv")
    future_df["time"] = pd.to_datetime(future_df["time"], utc=True)
    future_df["Source_time"] = pd.to_datetime(
        future_df["Source_time"], unit="s", utc=True
    )
    return (future_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Then, we can retrieve any saved model and use it to predict on the new data:
    """)
    return


@app.cell
def _(future_df, mlflow):
    model_name = "LinearRegression"
    model_version = 1
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri=model_uri)
    X_forecast = future_df[["Speed"]]
    _predictions = model.predict(X_forecast)
    predictions_df = future_df[["Speed"]].copy()
    predictions_df["Predicted_Power"] = _predictions
    print(predictions_df.head())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Serving your model

    So far, we have trained several models and registered the best one. Often times, we will want to deploy the best model in such a way that other people can take advantage of it, by sending a request with data and getting back predictions. In this project you will only serve the model locally, a step usually done before deployment to ensure that the model works as expected.

    Serving a model refers to the process of making a trained machine learning model available to receive input data and provide predictions or inferences based on that data. In other words, when you serve a model, you set it up in a way that it can be queried with new data, and it will produce predictions or outputs based on the patterns it learned during training.

    Serving a model is a critical step in the machine learning lifecycle, as it allows you to leverage the model's predictive capabilities in real-world applications. When a model is served, it becomes accessible to applications, websites, or other systems that need to utilize its predictions.

    You have tried serving the model in [one of the previous exercises](https://github.com/LSDA-BDM/exercise-mlflow-caiso). Run the same command to serve your current model.

    > Note. The current configuration is slightly different that in the exercise. You won't have to call `mlflow run .` as in the exercise. Do you know why? However, before serving the model, you will have to set the tracking URI in the terminal with:
    > ```bash
    > export MLFLOW_TRACKING_URI=http://localhost:5000
    > ```

    Serve the model and then use `curl` to test that it worked.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Packaging your project
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Lastly, we want to make sure that others are able to run the project and do further development. As opposed to just making the model available as before (with serving) now you will package the project in order to make it reproducible.

    Reproducibility refers to the ability of researchers, developers, and practitioners to recreate the exact same results, analyses, and outcomes from a given set of source code and data. In that sense, MLflow Projects is a standardized way to encapsulate an experiment in a reproducible
    manner, so colleagues or other stakeholders can reproduce your code. This is done by specifying all the dependencies using a 'pip' environment file, a 'conda' environment file, or a docker environment. Here we will take the ‘pip‘ approach.

    An MLflow project consists (at least) of:

    - The code you want to package, including the data for your experiments (if needed). Jupyter Notebooks are not supported, so you will have to more the relevant code in `.py` file; you can check the `script.py` file provided for an example.
    - A requirements.txt file specifying the dependencies for the code (in this case a text file with ‘pip‘-installable dependencies).
    - An MLproject file specifying which environment file to use, parameters accepted, and the entry points to the code.

    First, try to run the script `script.py` with

    ```bash
    python script.py
    ```
    > Note: For this to run you must have started an mlflow server as shown in Tracking your experiments.

    Then, create a similar script containing your experiments; also create the other files necessary for an MLProject. Have a look at [this exercise](https://github.com/LSDA-BDM/exercise-polynomial/) that explains how MLflow projects work.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Notes and hints

    Here are a few resources you can use to inform your design decisions for your experiments:

    **Joining the data.** How do you join and align the data will affect the amount of data you end up with for training your model. Make sure you explain your decision and reflect upon its pros and cons. Some resources for pandas: [Join](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.join.html), [Merge](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html), [Resample](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html).

    **Wind direction encoding.** You will most likely need to transform the wind direction into a numerical feature. Some possibilities to consider, make sure you reflect on their advantages and disadvantages:
    - [OneHotEncoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
    - Transforming to degrees or radians with a [custom transformer](https://towardsdatascience.com/pipelines-custom-transformers-in-scikit-learn-ef792bbb3260/)
    - Converting speed and direction to a [2-dimensional vector](https://www.tensorflow.org/tutorials/structured_data/time_series#wind)

    **Scaling.** Think what is the range of your features. Is your model affected by this? Should you scale your features? You should explain why would you do it, or why it's not necessary. Check [Scikit-Learn's Preprocessing and Normalization](https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling) or this [article](https://www.geeksforgeeks.org/ml-feature-scaling-part-2/).

    **Feature transformations.** If you want to apply different transformations to different columns (e.g. categorical and numerical), you can use the [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html).

    **Missing values.** Check [Scikit-Learn’s imputers](https://scikit-learn.org/stable/modules/impute.html) to see different options for dealing with missing values.

    **Splitting and shuffling.** You are working with time series data. Is it acceptable to do a random shuffle on the data when splitting for training? If so, why? If not, how can you split the data? Make sure to argue your decision. Check [Scikit-Learn’s TimeSeries Split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html#sklearn.model_selection.TimeSeriesSplit) or this [article](https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
