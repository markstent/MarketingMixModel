# Marketing Mix Modeling (MMM) Analysis

![header](mmmAnalysis.jpg)

This project provides a framework for conducting Marketing Mix Modeling (MMM) analysis. It includes functionalities for preprocessing data, fitting different types of regression/XGBoost models, calculating ROI, forecasting sales, and visualizing feature importance and media mix contributions.

A Jupyter notebook example of implementation can be found [here](mmmAnalysis_example.ipynb).

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Class and Method Descriptions](#class-and-method-descriptions)
- [License](#license)

## Installation

### Step 1: Clone the Repository

```
git git@github.com:markstent/MarketingMixModel.git
```

### Step 2: Create a Virtual Environment

Create a virtual environment to manage dependencies.

Using venv

```
python3 -m venv environment_name
```

### Step 3: Activate the Virtual Environment

On macOS and Linux
```
source environment_name/bin/activate
```

On Windows
```
environment_namev\Scripts\activate
```

### Step 4: Install Dependencies

Install the required dependencies using the requirements.txt file.

```
pip install -r requirements.txt
```

### Usage

#### Import the Class and Load Data

```
import pandas as pd
from mmm_analysis import MMMAnalysis

# Load your dataset
data = pd.read_csv('path_to_your_data.csv')

# Define target and features
target = 'sales'
features = ['TV', 'radio', 'newspaper']

# Initialize the MMMAnalysis class
mmm = MMMAnalysis(data, target, features, model_type='ridge')

```

#### Perform EDA
```
mmm.perform_eda()
```
#### Preprocess Data
```
mmm.preprocess_data()
```
#### Fit the Model
```
mmm.fit_model()
print(f"R-squared value: {mmm.r_squared}")
```
#### Calculate ROI
```
roi_df = mmm.calculate_roi()
mmm.plot_roi(roi_df)
```
#### Forecast Sales

Define spending scenarios and forecast sales.

```
scenarios = [
    {'TV': 300, 'radio': 100, 'newspaper': 150},
    {'TV': 100, 'radio': 150, 'newspaper': 100},
    {'TV': 400, 'radio': 100, 'newspaper': 50},
]

try:
    forecast_results = mmm.forecast_sales(scenarios)
    forecast_df = pd.DataFrame({
        'Scenario': [f'Scenario {i+1}' for i in range(len(scenarios))],
        'Predicted Sales': forecast_results
    })
    mmm.plot_forecast(forecast_df)
    mmm.plot_media_mix()
except ScenarioFeatureMismatchError as e:
    print(e)
except ValueError as e:
    print(f"Error: {e}")
```

### Class and Method Descriptions

#### MMMAnalysis

A class to perform Marketing Mix Modeling (MMM) analysis.

#### Methods
- ```__init__(self, data, target, features, date_column=None, date_format=None, model_type='linear', corr_threshold=0.9)```: Initializes the class with data, target, features, and model type ('linear', 'ridge', 'lasso', 'xgboost')
- ```preprocess_data(self)```: Preprocesses the data, checks for multicollinearity, and standardises the features.
- ```fit_model(self)```: Fits the chosen regression/XGBoost model and calculates the R-squared value and feature importance.
- ```calculate_roi(self)```: Calculates ROI for the top 5 features and returns a DataFrame.
- ```forecast_sales(self, spending_splits)```: Forecasts sales based on provided spending scenarios.
- ```plot_feature_importance(self)```: Plots the feature importance.
- ```plot_roi(self, roi_df)```: Plots the ROI by marketing channel.
- ```plot_forecast(self, forecast_df)```: Plots the forecasted sales for different spending scenarios.
- ```plot_media_mix(self)```: Plots the media mix contribution to sales.
- ```perform_eda(self)```: Performs exploratory data analysis including pair plots and correlation heatmaps.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
