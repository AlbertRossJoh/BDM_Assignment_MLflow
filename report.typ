
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

One of the main issues with the two datasets is the different cardinality and resolution. There are multiple ways of addressing this; however I ended up taking the following approach.
- Load power generation and weather forecast data
- Align the two data sources temporally (e.g. resampling, joins, or interpolation)
- Clearly justify your alignment strategy and discuss its implications

= Data preprocessing with pipelines

- Apply a suitable data-splitting strategy for train and test
- Handle missing values
- Transform wind direction into a numeric representation (e.g. encoding, radians, vector form)
- Scale numerical features where appropriate


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

