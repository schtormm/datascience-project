import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
import statsmodels.api as sm
from pmdarima.model_selection import train_test_split
from scipy import stats
from sklearn.metrics import (mean_absolute_percentage_error,
                             mean_squared_error, root_mean_squared_error)
from statsmodels.graphics.api import qqplot
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.arima.model import ARIMA

dataset = 'cleaned_V11.csv'
# mute sklearn warnings
import warnings

warnings.filterwarnings("ignore")

def get_parameters():
    filename = 'testset.json'
    try:
        with open(filename, 'r') as f:
            parameters = json.load(f)
    except FileNotFoundError:
        print(f"Parameter file {filename} not found. Using default parameters.")
        parameters = {'country_codes': ['USA', 'DNK', 'NLD'], 
                      'test_period': [1950, 2000],
                      'forecast_horizon': 10,
                      'model': 'ARIMA',
                      "parameters": {
                        "orders": [[2, 1, 2], [2, 1, 3], [3, 1, 2]]
                    }}       
    return parameters

def create_dataset(countries, test_period, dataset):
    df = pd.read_csv(dataset)
    # Create a dictionary of dataframes for each country with the test and train split
    country_dfs = {}
    print("Creating datasets for countries:", countries)
    for country in countries:
        country_data = df[df['countrycode'] == country].set_index('year')['rgdpe']
        train = country_data[(country_data.index >= test_period[0]) & (country_data.index < test_period[1])]
        test = country_data[(country_data.index >= test_period[1])]
        country_dfs[country] = {'train': train, 'test': test}
    return country_dfs

def create_models(country_dfs, model, arima_order = None):
    # For each country, create and fit an ARIMA model and save it in a dictionary
    country_models = {}
    print(f"Creating models with order {arima_order} for countries:", list(country_dfs.keys()))
    for country, data in country_dfs.items():
        print(f"Fitting ARIMA model for {country} with order {arima_order}")
        model = ARIMA(data['train'], order=(2, 1, 1)).fit()
        country_models[country] = model

    return country_models

def evaluate_models(country_dfs, model):
    # get MSE, RMSE and MAPE for each model on its test set using crossvalidation
    # should we possibly also use the different ARIMA forms here @Wesley1701?

    # note for MAPE from sklearn: 
    # Note that we are not using the common “percentage” definition: the percentage in the range [0, 100] is converted to a relative value in the range [0, 1] by dividing by 100. 
    # Thus, an error of 200% corresponds to a relative error of 2.
    evaluation_metrics = {}
    for country, data in country_dfs.items():
        test = data['test']
        model_fit = model[country]
        forecast = model_fit.predict(start=test.index[0], end=test.index[-1])
        mape = mean_absolute_percentage_error(test, forecast)
        mse = mean_squared_error(test, forecast)
        rmse = root_mean_squared_error(test, forecast)
        evaluation_metrics[country] = {'MSE': mse, 'RMSE': rmse, 'MAPE': mape}

    eval_df = pd.DataFrame.from_dict(evaluation_metrics, orient='index')
    eval_df.to_csv('evaluation_metrics.csv')


def evaluate_models_cross(country_dfs, model):
    # get MSE, RMSE and MAPE for each model on its test set using crossvalidation
    # same notes as above
    evaluation_metrics_cross = {}
    for country, data in country_dfs.items():
        series = data['train']._append(data['test'])
        model_fit = model[country]
        n_splits = 5
        mse_list = []
        rmse_list = []
        mape_list = []
        split_size = len(series) // n_splits
        for i in range(n_splits):
            train_end = (i + 1) * split_size
            train = series[:train_end]
            test = series[train_end:train_end + split_size]
            if len(test) == 0:
                continue
            model_cv = ARIMA(train, order=(2, 1, 1)).fit()
            forecast = model_cv.predict(start=test.index[0], end=test.index[-1])
            mse_list.append(mean_squared_error(test, forecast))
            rmse_list.append(root_mean_squared_error(test, forecast))
            mape_list.append(mean_absolute_percentage_error(test, forecast))
        evaluation_metrics_cross[country] = {
            'MSE': np.mean(mse_list),
            'RMSE': np.mean(rmse_list),
            'MAPE': np.mean(mape_list)
        }
   
    eval_df = pd.DataFrame.from_dict(evaluation_metrics_cross, orient='index')
    eval_df.to_csv('evaluation_metrics.csv')



def create_plots(models, country_dfs, timerange, forecast_horizon):
    print(f'Models: {json.dumps(models, indent=4, default=str)}')
    for country in models.keys():
        plt.figure(figsize=(10, 6))
        model = models[country]
        plot_predict(model, start=timerange[1], end=(timerange[1] + forecast_horizon), ax=None)
        country_dfs[country]['train'].plot(label='Observed')
        # make sure plot only shows from 2000 onwards
        plt.xlim([timerange[0], timerange[1] + forecast_horizon])
        # get rid of 1e6 notation on y-axis
        plt.ticklabel_format(style='plain', axis='y')
        # add title and labels
        plt.title(f'ARIMA(2,1,1) Forecast of rgdpe for {country}')
        plt.xlabel('Year')
        plt.ylabel('Real GDP (in millions)')
        plt.legend()
        # Save the plot to a file
        plot_name = f'plots_{country}.png'
        plt.savefig(plot_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved for {country}: {plot_name}")  

if __name__ == "__main__":
    parameters = get_parameters()
    print(f"Parameters: {parameters}")
    country_dfs = create_dataset(parameters['country_codes'], parameters['test_period'], dataset)
    arima_order = parameters['parameters']['orders'][0]  # Example: using the first ARIMA order
    country_models = create_models(country_dfs, parameters['model'], arima_order)
    create_plots(country_models, country_dfs, parameters['test_period'], parameters['forecast_horizon'])
    evaluate_models(country_dfs, country_models)
    evaluate_models_cross(country_dfs, country_models)
 