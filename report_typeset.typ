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
A common problem within the space of machine learning is the need to predict time series data. Time series data can take many forms, however in this report we will focus on weather data. Specifically, given a set of training data, how can we use common tooling, feature engineering and multiple machine learning models to reach an acceptable fit.

The data in question is a 90 day frame of wind speed and direction with another dataset showing the amount of power generated through wind turbines. The goal is to use MLOps to streamline model selection and comparison.
== Overview
= Methodology <sec:methods>

Since ML and MLOps is such an established field, multiple frameworks and libraries exist in order to make the life of the engineer easier. All runtime dependencies are version-pinned @requirements, which `python_env.yaml` installs into the isolated environment that `mlflow run` builds, so the pipeline reproduces from a clean checkout on another machine.

Most of the libraries are for convenience, however, we should specifically highlight the use of `sklearn` @scikit-learn, a very common ML library, and the usage of `mlflow` @mlflow, a tool for comparing and versioning models.

The main discovery process has been in a python notebook #footnote([Specifically a marimo notebook]), with models and model metrics being logged to `mlflow`. This initial discovery process was for previewing different models on a simple dataset. After this initial discovery process, the main focus was improve the best model. This was both done through feature engineering and hyperparameter tuning. The tuning was done through `optuna` @akiba2019optuna, which is a hyperparameter tuning framework.

// something about the pipeline handling all transformations


= Data alignment <sec:alignment>
// Loading power generation + weather forecast data.
// Temporal alignment strategy (resampling / joins / interpolation) and *why*.
// Implications of the chosen strategy (e.g. interpolated rows, mask of real vs synthetic).

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
