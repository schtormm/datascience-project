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
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.api import qqplot
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.arima.model import ARIMA

dataset = 'cleaned_V11.csv'
# mute sklearn warnings
import warnings

warnings.filterwarnings("ignore")

def get_parameters(filename = 'parameters.json'):
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

def create_models_ARIMA(country_dfs, parameters):
    arima_orders = parameters['orders']  # Example: using the first ARIMA order
    # For each country, create and fit an ARIMA model and save it in a dictionary
    country_models = {}
    print("Creating models for countries:", list(country_dfs.keys()))
    for country, data in country_dfs.items():
        country_models[country] = []
        for order in arima_orders:
            print(f"Fitting ARIMA{order} model for {country}")
            model_fit = ARIMA(data['train'], order=order).fit()
            country_models[country].append(model_fit)

    return country_models

def create_models_linear(country_dfs, parameters):
    exp = parameters['is_exponential']
    country_models = {}
    print("Creating models for countries:", list(country_dfs.keys()))
    for country, data in country_dfs.items():
        X_train = data['train'].index.to_numpy().reshape(-1, 1)
        if exp:
            y_train = np.log(data['train'].values)            
        else:
            y_train = data['train'].values
        print(f"Fitting {'exponential' if exp else 'linear'} data of {country} in to the model")
        linear_model = LinearRegression().fit(X_train, y_train)
        country_models[country] = linear_model
    return country_models

def get_predictions(country_dfs, models):
    predictions = {}
    for country, data in country_dfs.items():
        models_to_evaluate = models[country]
        test = data['test']
        if type(models_to_evaluate) == list:
            predictions[country] = {}
            for model_fit in models_to_evaluate:
                forecast = model_fit.predict(start=test.index[0], end=test.index[-1])
                order = model_fit.model.order
                predictions[country][order] = forecast
        else:
            x = test.index.to_numpy()
            predictions[country] = models_to_evaluate.predict(x.reshape(-1, 1))
    return predictions

def evaluate_models(country_dfs, predictions, models, used_model = None):
    # get MSE, RMSE and MAPE for each model on its test set using crossvalidation

    # note for MAPE from sklearn: 
    # Note that we are not using the common “percentage” definition: the percentage in the range [0, 100] is converted to a relative value in the range [0, 1] by dividing by 100. 
    # Thus, an error of 200% corresponds to a relative error of 2.
    evaluation_metrics = {}
    for country, data in country_dfs.items():
        evaluation_metrics[country] = []
        test = data['test']
        predictions_for_country = predictions[country]
        if type(predictions_for_country) == dict:
            for key in predictions_for_country:
                preds = predictions_for_country[key]
                mape = mean_absolute_percentage_error(test, preds)
                mse = mean_squared_error(test, preds)
                rmse = root_mean_squared_error(test, preds)
                if used_model == 'ARIMA':
                    evaluation_metrics[country].append({'ARIMA Order': key, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape})
                else:
                    evaluation_metrics[country].append({'MSE': mse, 'RMSE': rmse, 'MAPE': mape})
        else:
            mape = mean_absolute_percentage_error(test, predictions_for_country)
            mse = mean_squared_error(test, predictions_for_country)
            rmse = root_mean_squared_error(test, predictions_for_country)
            evaluation_metrics[country] = {'MSE': mse, 'RMSE': rmse, 'MAPE': mape}

    eval_df = pd.DataFrame.from_dict(evaluation_metrics, orient='index')
    eval_df.to_csv('evaluation_metrics.csv')


def evaluate_models_cross_ARIMA(country_dfs, models):
    # get MSE, RMSE and MAPE for each model on its test set using crossvalidation
    # same notes as above
    evaluation_metrics_cross = {}
    for country, data in country_dfs.items():
        evaluation_metrics_cross[country] = []
        series = data['train']._append(data['test'])
        n_splits = 5
        split_size = len(series) // n_splits
        for model in models[country]:
            mse_list = []
            rmse_list = []
            mape_list = []
            for i in range(n_splits):
                train_end = (i + 1) * split_size
                train = series[:train_end]
                test = series[train_end:train_end + split_size]
                if len(test) == 0:
                    continue
                
                try:
                    model_cv = ARIMA(train, order=model.model.order).fit()
                    forecast = model_cv.predict(start=test.index[0], end=test.index[-1])
                    mse_list.append(mean_squared_error(test, forecast))
                    rmse_list.append(root_mean_squared_error(test, forecast))
                    mape_list.append(mean_absolute_percentage_error(test, forecast))
                except Exception as e:
                    print(f"Warning: ARIMA model cross validation failed for {country} with order {model.model.order} on split {i}: {e}")
                    continue
            
            if mse_list:  # Only add metrics if we have valid results
                evaluation_metrics_cross[country].append ({
                    "ARIMA Order": model.model.order,
                    'MSE (mean of all splits)': np.mean(mse_list),
                    'RMSE (mean of all splits)': np.mean(rmse_list),
                    'MAPE (mean of all splits)': np.mean(mape_list)
                })
   
    eval_df = pd.DataFrame.from_dict(evaluation_metrics_cross, orient='index')
    eval_df.to_csv('evaluation_metrics_cross.csv')



def create_plots_arima(models, country_dfs, timerange, forecast_horizon):
    
    for country in models.keys():
        fig, axs = plt.subplots(3, 1, figsize=(10, 20))
        model = models[country]
        # create one plot
        i = 0
        for m in model:
            # put plot in the figure
            forecast = m.get_forecast(steps=forecast_horizon)
            forecast_index = np.arange(timerange[1], timerange[1] + forecast_horizon)
            # subplot for each model
            axs[i].plot(forecast_index, forecast.predicted_mean, label=f'Forecast ARIMA{m.model.order}')
            country_dfs[country]['train'].plot(ax=axs[i], label='Observed')
            # make sure plot only shows from 2000 onwards
            axs[i].set_xlim([timerange[0], timerange[1] + forecast_horizon])
            # get rid of 1e6 notation on y-axis
            axs[i].ticklabel_format(style='plain', axis='y')
            # add title and labels
            axs[i].set_title(f'ARIMA{m.model.order} Forecast of rgdpe for {country}')
            axs[i].set_xlabel('Year')
            axs[i].set_ylabel('Real GDP (in millions)')
            axs[i].legend()
            i +=1
            # Save the plot to a file
            plot_name = f'plots_{country}.png'
        fig.savefig(plot_name, dpi=300, bbox_inches='tight')
        
        print(f"Plot saved for {country}: {plot_name}")  

if __name__ == "__main__":
    parameters = get_parameters('parameters_linear.json')
    print(f"Parameters: {parameters}")
    country_dfs = create_dataset(parameters['country_codes'], parameters['test_period'], dataset)

    #maybe move this into the switch statement
    match parameters['model']:
        case 'ARIMA':
            country_models = create_models_ARIMA(country_dfs, parameters['parameters'])
            predictions = get_predictions(country_dfs, country_models)
            # print(predictions)
            create_plots_arima(country_models, country_dfs, parameters['test_period'], parameters['forecast_horizon'])
            evaluate_models(country_dfs, predictions, country_models, 'ARIMA')
            evaluate_models_cross_ARIMA(country_dfs, country_models)
        case 'Linear Regression':
            country_models = create_models_linear(country_dfs, parameters['parameters'])
            predictions = get_predictions(country_dfs, country_models)
            evaluate_models(country_dfs, predictions, country_models)
        case _:
            raise ValueError(f"Model {parameters['model']} not recognized.")
    
    


 