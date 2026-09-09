#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Utilizing weather forecasts for power output prediction],
  abstract: [

  ],
  authors: (
    (
      name: "Albert Juel Ross",
      organization: [ITU],
      location: [Copenhagen, Denmark],
      email: "alrj@itu.dk",
    ),
  ),
  index-terms: (),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

= Introduction
In 2020 the Orkney islands generated 128% of their electricity needs from renewables @orkney where bulk of this power generation, came from wind turbines @orkney-renewables. Orkney currently has limitations with their grid, as export is constrained to the capacity of two subsea cables @orkney. For this reason, short-term forecasts of power generated from wind-energy could be desirable. The main focus of this report is building a reproducible machine learning pipeline that predicts total power output, given a forecast of wind speed and direction.

The data in question is a 90 day frame of wind speed and direction with another dataset showing the amount of power generated through wind turbines.
== Overview
= Methodology <sec:methods>

Since ML and MLOps is such an established field, multiple frameworks and libraries exist in order to make the life of the engineer easier. All runtime dependencies are version-pinned @requirements, which `python_env.yaml` installs into the isolated environment that `mlflow run` builds, so the pipeline reproduces from a clean checkout on another machine.

Most of the libraries are for convenience, however, we should specifically highlight the use of `sklearn` @scikit-learn, a very common ML library, and the usage of `mlflow` @mlflow, a tool for comparing and versioning models.

The main discovery process has been in a python notebook #footnote([Specifically a marimo notebook]), with models and model metrics being logged to `mlflow`. This initial discovery process was for previewing different models on a simple dataset. After this initial discovery process, the main focus was improve the best model. This was both done through feature engineering and hyperparameter tuning. The tuning was done through `optuna` @akiba2019optuna, which is a hyperparameter tuning framework.

Both feature transformations/engineering, and the model itself, is placed in a single pipeline. This means that model fitting and predictions are reproducible, and simple.


= Data alignment <sec:alignment>
//Data was loaded in using using the `read_csv` function from `polars` which is a popular dataframe library.
One of the main issues with the two datasets is the different cardinality and resolution. This being because the power dataset is sampled every minute, but the wind dataset is sampled every three hours. There are multiple ways of addressing this; either downsample the power dataset to match the 3 hour interval, or upsample the wind dataset to match the minute interval. Or do something in-between. The main benefit of downsampling is that we match the data to the least common denominator, i.e. we're not creating any data or making any assumptions about its shape. This however comes at the cost of the amount of training data as-well as the granularity of the predictions. Upsampling is the opposite of this, we need to make assumptions about the data's shape, however we end up getting more data and predictions can be made with higher granularity #footnote([Granularity here being the precision of the prediction, if the model has been trained on hourly data, it will be hard to predict what will happen the next second.]).

The approach which seems to maintain most of the variance of the dataset, is upsampling the wind data to minute granularity. The sum of the standard deviations of wind speed an total power output, is in the original dataset $~15.90$, with the sum of the re-sampled standard deviations being $~15.85$. A more balanced approach is to re-sample the data to hour granularity, however this does not maintain as much of the variance $~15.65$.

The upsampling is done by interpolating between actual data points, which assumes that windspeed changes smoothly, direction is also interpolated, using a simple algorithm which just chooses the midpoint between two categorical directions.

The downsampling is done by grouping the power dataset by hours and taking the mean of the total power output.

The main reason for upsampling the wind data to such an extreme is that the variance did not take a bit hit, and the model predictions got better.




= Data preprocessing <sec:preprocessing>
// The sklearn Pipeline / ColumnTransformer that wraps every step below.

== Train/test split <sec:split>
// Splitting strategy for time-series data and why (TimeSeriesSplit, no shuffling).

== Missing values <sec:missing>
// How gaps are handled (imputation strategy, native NaN handling) and why.

== Wind direction encoding <sec:direction>
// Compass -> numeric representation (sin/cos vector form) and why over label/one-hot.

== Feature scaling <sec:scaling>
// Which features are scaled, which estimators need it, and why.

= Model training and evaluation <sec:modeling>

== Models <sec:models>
// The >= 2 regression models chosen (RidgeCV, HistGradientBoostingRegressor) and why.

== Evaluation metrics <sec:metrics>
// Regression metrics reported (R2, RMSE, MAE, MSE) and how they are aggregated over folds.

== Future predictions <sec:future>
// Using future.csv to generate forecasts and confirm the model runs on unseen data.

= Experiment tracking with MLflow <sec:tracking>
// Parameters, metrics and artifacts logged; how experiments and runs are organised.

== Model comparison and selection <sec:selection>
// Comparing model variants in the MLflow UI and selecting the best model.

== Results <sec:results>
// Results pulled from the MLflow interface (table / screenshots of the comparison).

= Model serving <sec:serving>
// Registering the selected model in the MLflow Model format.
// Serving it and exposing a prediction endpoint that takes weather inputs.
// Screen capture of a successful serving request.

= Reproducibility with MLflow Projects <sec:reproducibility>
// Packaging training as an MLProject (entry points, MLproject file).
// Environment file (python_env.yaml + requirements.txt, @requirements).
// Demonstrating a from-scratch run on another machine via `mlflow run`.

= Reflection <sec:reflection>

== Training data window size <sec:window>
// Discussion of the training window (e.g. 90 days) and its effect.

== Limitations and improvements <sec:limitations>
// Limitations of the approach and concrete potential improvements.

= Conclusion <sec:conclusion>

= Use of generative AI <sec:ai>
Anthropic's Claude @claude was used during this project as a coding assistant and a sounding board. Concretely it helped scaffold and refactor parts of the pipeline code (the MLflow Project entry points, and the shared plotting and experiment helpers), and discuss design trade-offs such as the temporal alignment strategy and the cross-validation setup. All generated code and text were reviewed, tested, and edited by the author, who takes full responsibility for the final submission. Model selection, feature-engineering decisions, and interpretation of the results are the author's own.
== Example of code refactor
One of the refactors made by Clause is from a function into a pipeline transformer.
```Python
COMPASS_16 = [
  "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]
DIR_TO_DEG = {d: i * 22.5 for i, d in enumerate(COMPASS_16)}

def add_cyclic_direction(df, col="Direction"):
    deg = df[col].replace(DIR_TO_DEG).cast(pl.Float64)
    rad = deg.radians()
    return df.with_columns(
        direction_sin=rad.sin(),
        direction_cos=rad.cos(),
    )
```

refactored into

```Python
class DirectionSinCos(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        angle = (
            pl.col("Direction").replace_strict(
                DIR_TO_IDX, return_dtype=pl.Float64
            )
            * 22.5
        ).radians()
        return X.select(
            angle.sin().alias("direction_sin"),
            angle.cos().alias("direction_cos"),
        )

    def get_feature_names_out(self, input_features=None):
        return ["direction_sin", "direction_cos"]
```
== Example of code generation
In `train.py` the main function takes 19 variables, for each of these we have to manually annotate a click option and a default value. This is very time consuming to do by hand without any clear benefit. Thus a set of hyperparameters were supplied to claude after which the function was scaffolded.

The plotting code was completely built by claude.
