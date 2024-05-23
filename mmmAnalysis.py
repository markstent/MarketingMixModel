import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# Custom error handling for the scenario planner

class ScenarioFeatureMismatchError(Exception):
    def __init__(self, missing_features):
        self.missing_features = missing_features
        self.message = f"Forecast scenarios are missing the following features: {', '.join(missing_features)}"
        super().__init__(self.message)

class MMMAnalysis:
    def __init__(self, data, target, features, date_column=None, date_format=None, model_type='linear', corr_threshold=0.9):
        """
        Initialize the MMMAnalysis class.
        
        Parameters:
        - data: DataFrame containing the dataset.
        - target: String, name of the target variable column.
        - features: List of strings, names of the feature columns.
        - date_column: String, name of the date column (optional).
        - date_format: String, format of the date column (optional).
        - model_type: String, type of model to use ('linear', 'ridge', 'lasso', 'xgboost').
        - corr_threshold: Float, threshold for removing highly correlated features.
        """
        self.data = data
        self.target = target
        self.features = features
        self.date_column = date_column
        self.date_format = date_format
        self.model_type = model_type
        self.corr_threshold = corr_threshold
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.r_squared = None
        
        if self.date_column:
            self.data[self.date_column] = pd.to_datetime(self.data[self.date_column], format=self.date_format)
            self.data.set_index(self.date_column, inplace=True)
        
    def preprocess_data(self):
        # Check for multicollinearity and remove highly correlated features
        corr_matrix = self.data[self.features].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.corr_threshold)]
        self.features = [feature for feature in self.features if feature not in to_drop]
        
        # Debugging output
        print("Features after multicollinearity check:", self.features)
        
        # Standardize features
        self.data[self.features] = self.scaler.fit_transform(self.data[self.features])
        
    def fit_model(self):
        X = self.data[self.features]
        y = self.data[self.target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if self.model_type == 'linear':
            self.model = LinearRegression()
            self.model.fit(X_train, y_train)
        elif self.model_type == 'ridge':
            self.model = Ridge()
            param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
            grid_search = GridSearchCV(self.model, param_grid, cv=3, scoring='r2')
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
        elif self.model_type == 'lasso':
            self.model = Lasso(max_iter=10000)  # Increase max_iter to ensure convergence
            param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
            grid_search = GridSearchCV(self.model, param_grid, cv=3, scoring='r2')
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(objective='reg:squarederror')
            # Hyperparameter tuning using GridSearchCV
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'subsample': [0.8, 0.9, 1.0]
            }
            grid_search = GridSearchCV(self.model, param_grid, cv=3, scoring='r2')
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
        else:
            raise ValueError("Invalid model type. Choose 'linear', 'ridge', 'lasso', or 'xgboost'.")
        
        # Predictions and R-squared value
        y_pred = self.model.predict(X_test)
        self.r_squared = r2_score(y_test, y_pred)
        
        # Feature importance
        if self.model_type in ['linear', 'ridge', 'lasso']:
            self.feature_importance = np.abs(self.model.coef_)
        elif self.model_type == 'xgboost':
            self.feature_importance = self.model.feature_importances_
        
        # Debugging output
        print("Feature Importance:")
        print(self.feature_importance)
        
        # Check if all feature importances are zero
        if np.all(self.feature_importance == 0):
            print("Warning: All feature importances are zero. The model may not fit the data well.")
            print("Consider trying a different model type, such as 'ridge' or 'xgboost'.")
        
    def calculate_roi(self):
        if self.feature_importance is not None:
            mean_sales = self.data[self.target].mean()
            roi = (self.feature_importance[:5] * self.data[self.features[:5]].mean().values) / mean_sales  # calculates roi on top 5 features
            roi_df = pd.DataFrame({'Channel': self.features[:5], 'ROI': roi})
            return roi_df
        else:
            print("Feature importance is not calculated. Fit the model first.")
            return pd.DataFrame()
        
    def forecast_sales(self, spending_splits):
        if self.model is not None:
            forecast_df = pd.DataFrame(spending_splits)
            # Check if scenario variables match dataset variables
            missing_features = [feature for feature in self.features if feature not in forecast_df.columns]
            if missing_features:
                raise ScenarioFeatureMismatchError(missing_features)

            extra_features = [feature for feature in forecast_df.columns if feature not in self.features]
            if extra_features:
                raise ValueError(f"Forecast scenarios contain extra features not present in the model: {extra_features}")

            # Ensure all features are included in the forecast dataframe
            for feature in self.features:
                if feature not in forecast_df.columns:
                    forecast_df[feature] = self.data[feature].mean()
                    
            forecast_scaled = self.scaler.transform(forecast_df)
            predicted_sales = self.model.predict(forecast_scaled)
            return predicted_sales
        else:
            print("Model is not fitted yet. Fit the model first.")
            return np.array([])
    
    def plot_feature_importance(self):
        if self.feature_importance is not None and len(self.feature_importance) > 0:
            importance_df = pd.DataFrame({'Feature': self.features, 'Importance': self.feature_importance})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
            plt.figure(figsize=(12, 6))
            sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
            plt.title('Feature Importance')
            plt.show()
        else:
            print("Feature importance is not available.")

    def plot_roi(self, roi_df):
        if not roi_df.empty:
            plt.figure(figsize=(12, 6))
            sns.barplot(data=roi_df, x='Channel', y='ROI', palette='viridis')
            plt.title('Return on Investment (ROI) by Marketing Channel')
            plt.show()
        else:
            print("ROI data is not available.")
        
    def plot_forecast(self, forecast_df):
        if not forecast_df.empty:
            plt.figure(figsize=(12, 6))
            sns.barplot(data=forecast_df, x='Scenario', y='Predicted Sales', palette='viridis')
            plt.title('Forecasted Sales for Different Spending Scenarios')
            plt.show()
        else:
            print("Forecast data is not available.")
    
    def plot_media_mix(self):
        if self.feature_importance is not None:
            contributions = self.feature_importance[:5] * self.data[self.features[:5]].mean().values
            contributions_df = pd.DataFrame({'Channel': self.features[:5], 'Contribution': contributions})
            contributions_df['Contribution'] = contributions_df['Contribution'].abs()  # Use absolute values to avoid negative values
            contributions_df = contributions_df.dropna()  # Drop any rows with NaN values
            contributions_df = contributions_df[contributions_df['Contribution'] > 0]  # Remove zero contributions

            if not contributions_df.empty:
                plt.figure(figsize=(10, 6))
                plt.pie(contributions_df['Contribution'], labels=contributions_df['Channel'], autopct='%1.1f%%', colors=sns.color_palette('viridis', len(contributions_df)))
                plt.title('Marketing Mix Contribution to Sales')
                plt.show()
            else:
                print("Media mix data is not available.")
        else:
            print("Feature importance is not calculated. Fit the model first.")
        
    def perform_eda(self):
        if not self.data.empty:
            # Plot pairplot to see the relationships between features and target
            sns.pairplot(self.data, diag_kind='kde')
            plt.show()

            # Plot correlation heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm')
            plt.title('Correlation Heatmap')
            plt.show()
        else:
            print("Data is not available for EDA.")
