#import "@preview/charged-ieee:0.1.4": ieee
#import "@preview/wordometer:0.1.5": total-words, word-count

#show: word-count.with(exclude: (raw,))
#set table.hline(stroke: 0.5pt)

#show: ieee.with(
  title: [Utilizing weather forecasts for power output prediction],
  abstract: [
    #total-words words
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
  figure-supplement: [Figure],
)

= Introduction
In 2020 the Orkney islands generated 128% of their electricity needs from renewables @orkney where bulk of this power generation, came from wind turbines @orkney-renewables. Orkney currently has limitations with their grid, as export is constrained to the capacity of two subsea cables @orkney. For this reason, short-term forecasts of power generated from wind-energy could be desirable. The main focus of this report is building a reproducible machine learning pipeline that predicts total power output, given a forecast of wind speed and direction.

The data in question is a 90 day frame of wind speed and direction with another dataset showing the amount of power generated through wind turbines.

== Overview
= Methodology <sec:methods>

Since ML and MLOps is such an established field, multiple frameworks and libraries exist in order to make the life of the engineer easier. All runtime dependencies are version-pinned @requirements #footnote([The `uv` package manager was actually used for development, but MLFlow does not seem to support this in the MLproject file, thus a seperate requirements file exists.]), which `python_env.yaml` installs into the isolated environment that `mlflow run` builds, so the pipeline reproduces from a clean checkout on another machine.

Most of the libraries are for convenience, however, we should specifically highlight the use of `sklearn` @scikit-learn, a very common ML library, and the usage of `mlflow` @mlflow, a tool for comparing and versioning models.

The main discovery process has been in a python notebook, with models and model metrics being logged to `mlflow`. The process included building a meta-framework for automating experiments, and for building "immutable" pipelines#footnote([The pipeline itself is not immutable, nothing in python is. The intended interaction with the builder is done in such a way that changes to the current workflow is immutable. This is especially useful for multiprocessing]). The main purpose of this meta-framework was to (1) make pipeline updates simple and immutable, (2) making experiments easy and exhaustive. As it can be hard to figure out whats the best combination of tranformers/regressors, the main method employed in this project is doing exhaustive search on pipelines. Each of these runs are recorded to mlflow, where we can view the properties of individual runs.
//This initial discovery process was for previewing different models on a simple dataset. After this initial discovery process, the main focus was improve the best model. This was both done through feature engineering and hyperparameter tuning. The tuning was done through `optuna` @akiba2019optuna, which is a hyperparameter tuning framework.

Both feature transformations/engineering, and the model itself, is placed in a single pipeline. This means that model fitting and predictions are reproducible, and simple. The only step which is not in the pipeline is the data-re-sampling#footnote([The `sklearn.Pipeline` is not made for re-sampling in general, for this the `imblearn.Pipeline` could be used, however I found this out of scope for the project.]).


= Data alignment <sec:alignment>
//Data was loaded in using using the `read_csv` function from `polars` which is a popular dataframe library.
One of the main issues with the two datasets is the different cardinality and resolution. This being because the power dataset is sampled every minute, but the wind dataset is sampled every three hours. There are multiple ways of addressing this; either downsample the power dataset to match the 3 hour interval, or upsample the wind dataset to match the minute interval. Or do something in-between. The main benefit of downsampling is that we match the data to the least common denominator, i.e. we're not creating any data or making any assumptions about its shape. This however comes at the cost of the amount of training data as-well as the granularity of the predictions. Upsampling is the opposite of this, we need to make assumptions about the data's shape, however we end up getting more data and predictions can be made with higher granularity #footnote([Granularity here being the precision of the prediction, if the model has been trained on hourly data, it will be hard to predict what will happen the next second.]).

The approach that best balances dataset distributions is to re-sample to 1-hour intervals. Upsampling is done by interpolating between actual data points, assuming wind speed changes smoothly, while direction is interpolated via forward filling. Downsampling is done by grouping the power dataset by hour and taking the mean of total power output.
#figure(
  image("./figures/power-distribution.png", width: 70%),
  caption: [Total power output distribution (Gaussian KDE) before and after down-sampling to a 1-hour interval (Mean over 1-hour frame)],
)
As we see, this down-sampling causes an increase in the 30 MW range and in the 15 MW range. This is most likely some large values affecting the mean #footnote([This can be corrected for using the median, however this minimally worsens the R2 score of the models.]).

#figure(
  image("./figures/wind-distribution.png", width: 70%),
  caption: [Wind speed distribution (Gaussian KDE) before and after re-sampling to a 1-hour interval (Linear interpolation)],
)

The wind speed seems minimally affected by the up-sampling, only causing some smoothing in the distribution, this makes sense as we're doing linear interpolation.
#figure(
  image("./figures/direction-distribution.png", width: 70%),
  caption: [Direction class counts before and after re-sampling to a 1-hour interval (forward fill)],
)

Directions are forward filled, thus there is almost no difference between the raw dataset and the upsampled one. The main reason for the small difference is that there exists two gaps which are larger than 3 hours.
#figure(
  table(
    columns: 2,
    stroke: none,
    table.header(
      table.hline(),
      [Measurement gaps], [count],
      table.hline(),
    ),
    [3h], [713],
    [6h], [1],
    [12h], [1],
  ),
)

Thus this causes some over-representation of some classes, we avoid this problem by ignoring it#footnote([The over-representation is minimal, I conjecture that this problem is not worth chasing.]).

//The approach which maintains most of the variance of the dataset, is upsampling the wind data to minute granularity. The sum of the standard deviations of wind speed an total power output, is in the original dataset $~15.90$, with the sum of the re-sampled standard deviations being $~15.85$. A more balanced approach is to re-sample the data to hour granularity, however this does not maintain as much of the variance $~15.65$.

= Data preprocessing <sec:preprocessing>
Due to the main methodology of doing exhaustive pipeline runs, multiple approaches have been taken, which may or may not end up in the optimal pipeline. This section will describe the approaches taken.

== Train/test split <sec:split>
The data-splitting was done using Cross-Validation (CV). When using CV the number of folds chosen were 5. The main reason for this is that this is what `TimeSeriesSplit` defaults to. Other splits were tested, such as `ShuffleSplit` and `GroupShuffleSplit`, however model performance got suspiciously good, suggesting data leakage. Using a time series split is also representational of the real world usage on the model, i.e. we use previous data points to predict future data. One issue with doing interpolation in combination with CV is that the split might land on interpolated features, these are auto-correlated, meaning that the model was actually trained on data which is in the test set.

== Missing values <sec:missing>
The dataset does not have any null values, any introduced are by doing joins and re-sampling, which have been described in section @sec:alignment.

== Wind direction encoding <sec:direction>
Two methods of direction encoding have been employed and tested. The first method is using a `OneHotEncoder`, which encodes categorical data by pivoting the categories to binary columns. The other method is encoding the directions to degrees and extracting sin and cos from those degrees. Since the compass is split into 16 classes, we have $360/16=22.5 degree$. This just accumulates the further we go around the compass.

//The need to handle direction encoding, differs by the model which is chosen. For a linear regression model, which does not have built in support for categorical data, we can encode the direction into sinus and cosinus. The main problem here being that linear models does not handle non-linear data well. I chose to go with boosted trees for my main model, specifically the `sklearn.ensemble.HistGradientBoostRegressor`. This has native support for categorical data. However the categories does not really represent the circular dependency of the data, i.e. which directions are close to each other. I therefore complimented it with the direction encoding, however, for boosted trees, it did not seem to make much of a difference.
== Feature scaling <sec:scaling>
With linear models it can make sense to scale features as some feature might skew the MSE, causing other features to be undermined. To do this a `StandardScaler` was applied. A `MinMaxScaler` was also tried. The main difference between the two is that the `StandardScaler` and the `MinMaxScaler` is that the standard scaler removes the mean such that the dataset has unit variance @standardscaler (i.e the z-score is calculated) $z=(x-mu)/sigma$. The `MinMaxScaler` scales each feature such that it is in the range between 0 and 1 @minmaxscaler.
//With a linear regression pipeline, it can make a lot of sense to to scale features, as is changes the properties of how the curve is fitted. However since I ended up using boosted trees, this benefit is no longer there. The reason is that boosted trees makes a series of "splits", meaning that monotonic transformations (such as scaling) have no real impact.


= Model training and evaluation <sec:modeling>

== Models <sec:models>
// The >= 2 regression models chosen (RidgeCV, HistGradientBoostingRegressor) and why.

Four models were evaluated against each other during the exhaustive runs.
- A linear regressions model, `LinearRegression`
- A tree based model, `HistGradientBoostingRegressor`
- A support vector machine based model, `SVR`
- A multi layer perceptron model, `MLPRegressor`

The choice of these models are a bit arbitrary but they try the main types of models on the data. An important note is that training `SVR` on large datasets is painfully slow, meaning that it is impractical to use when upsampling data to minute granularity. Thus for this specific instance the `LinearSVR` was chosen.

== Evaluation metrics <sec:metrics>
The main metrics used to evaluate these models is the mean $R^2$ score of 5 CV folds. Other supporting metrics are mean _RMSE_ and mean _MAE_. As well as best and worst of each respectively. $R^2$ explains the proportion of variance which is explained by the model. An $R^2$ score of 0 is no better than just using the mean, 1 is a perfect fit, and a negative value means the model is worse than just using the mean.

== Future predictions <sec:future>
// Using future.csv to generate forecasts and confirm the model runs on unseen data.
#figure(
  image("./figures/total-forecast.png"),
  caption: [Forecast for future total power output (MW) based on weather forecast data],
)

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
Anthropic's Claude @claude was used during this project as a coding assistant and a sounding board. Concretely it helped scaffold, refactor and debug parts of the pipeline code#footnote([Specifically used for `matplotlib` plotting, refactoring some of the Experiment builder code and miscellaneous debugging]). All generated code and text were reviewed, tested, and edited by the author, who takes full responsibility for the final submission. Model selection, feature-engineering decisions, and interpretation of the results are the author's own.
== Example of code refactor
One of the code refactors done by Claude, is that the experiment builder was first responsible for building the pipeline. Later I, after building the `ColumnTransformerBuilder`, realized that the current architecture was not sound. Thus it refactored the existing building logic into the `PipelineBuilder`.
== Example of code generation
The plotting logic with in was completely built by Claude
