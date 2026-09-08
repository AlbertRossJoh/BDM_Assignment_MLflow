
#set page(header: context {
  [
    Albert Juel Ross #h(1fr) BDM ITU 2026 Handin 1#h(1fr) Page #counter(page).display()
    #line(length: 100%, stroke: 0.5pt)
  ]
})

#set document(title: [Utilizing weather forecasts for power output prediction])

#title()

= Data alignment
Data was loaded in using using the `read_csv` function from `polars` which is a popular dataframe library.

One of the main issues with the two datasets is the different cardinality and resolution. This being because the power dataset is sampled every minute, but the wind dataset is sampled every three hours.There are multiple ways of addressing this; either downsample the power dataset to match the 3 hour interval, or upsample the wind dataset to match the minute interval. Or do something in-between. The main benefit of downsampling is that we match the data to the least common denominator, i.e. we're not creating any data or making any assumptions about its shape. This however comes at the cost of the amount of training data as-well as the granularity of the predictions. Upsampling is the opposite of this, we need to make assumptions about the data's shape, however we end up getting more data and predictions can be made with higher granularity #footnote([Granularity here being the precision of the prediction, if the model has been trained on hourly data, it will be hard to predict what will happen the next second.]).

The approach which seems to maintain most of the variance of the dataset, is upsampling the wind data to minute granularity. The sum of the standard deviations of wind speed an total power output, is in the original dataset $~15.90$, with the sum of the re-sampled standard deviations being $~15.76$.

The upsampling is done by interpolating between actual data points, which assumes that windspeed changes smoothly, direction is also interpolated, using a simple algorithm which just chooses the midpoint between two categorical directions.

The downsampling is done by grouping the power dataset by hours and taking the mean of the total power output.

The main reason for upsampling the wind data to such an extreme is that the variance did not take a bit hit, and the model predictions got better.


//- Load power generation and weather forecast data
//- Align the two data sources temporally (e.g. resampling, joins, or interpolation)
//- Clearly justify your alignment strategy and discuss its implications

= Data preprocessing with pipelines

The data-splitting was done within the objective function using Cross-Validation (CV). When using CV the number of folds chosen were 3, 5 and 8; however when optimizing the hyperparameters, the 3 splits were chosen, while the two other splits are metrics logged. The reason for this split is a bit arbitrary, but when testing it was a nice balance between having large sets to train the model on, while also having sufficient testing data. The method used is the `TimeSeriesSplit` which trains the model in increasing order.

The dataset does not have any null values, any introduced are by doing joins and re-sampling, which have been described above.

The need to handle direction encoding, differs by the model which is chosen. For a linear regression model, which does not have built in support for categorical data, we can encode the direction into sinus and cosinus. The main problem here being that linear models does not handle non-linear data well. I chose to go with boosted trees for my main model, specifically the `sklearn.ensemble.HistGradientBoostRegressor`. This has native support for categorical data. However the categories does not really represent the circular dependency of the data, i.e. which directions are close to each other. I therefore complimented it with the direction encoding, however, for boosted trees, it did not seem to make much of a difference.

With a linear regression pipeline, it can make a lot of sense to to scale features, as is changes the properties of how the curve is fitted. However since I ended up using boosted trees, this benefit is no longer there. The reason is that boosted trees makes a series of "splits", meaning that monotonic transformations (such as scaling) have no real impact.
//- Apply a suitable data-splitting strategy for train and test
//- Handle missing values
//- Transform wind direction into a numeric representation (e.g. encoding, radians, vector form)
//- Scale numerical features where appropriate


= Model training and evaluation

- Train at least 2 regression models of your choice
- Use evaluation metrics appropriate for regression
- Use the `future.csv` file to generate future predictions (to check that your model works on new data)

= Experiment tracking with MLflow

- Log parameters, metrics, and artifacts using MLflow Tracking
- Organize experiments and runs clearly
- Use MLflow to compare model variants and select a best model
- You need to include in a report the results of your experiments from the MLflow interface

= Model serving

- Register the selected model using the MLflow Model format
- Serve the model
- Expose a prediction endpoint that accepts weather inputs and returns power forecasts
- You need to include in a report a screen capture showing successful serving of the model

= Reproducibility with MLflow Projects

- Package your training code as an MLProject
- Specify dependencies using an environment file
- Ensure the project can be executed from scratch on another machine

= Reflection

- Discussion of training data window size (e.g. 90 days)
- Mention limitations of your approach and potential improvements

