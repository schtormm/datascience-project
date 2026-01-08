import json
import os
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
import statsmodels.api as sm
from pmdarima.model_selection import train_test_split
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_percentage_error,
                             mean_squared_error, root_mean_squared_error)
from sklearn.model_selection import GridSearchCV
from statsmodels.graphics.api import qqplot
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.arima.model import ARIMA

dataset = 'cleaned_V11_with_synthetic.csv'

# mute sklearn warnings
import warnings

warnings.filterwarnings("ignore")

def output_folder_setup(parameters_filename='parameters.json'):
    # Create nested structure
    date_folder = datetime.now().strftime("%Y-%m-%d")
    base_path = f"experiments/{date_folder}"
    os.makedirs(base_path, exist_ok=True)

    # Find the next experiment number
    experiment_num = 1
    while os.path.exists(f"{base_path}/experiment_{experiment_num}"):
        experiment_num += 1

    output_folder = f"{base_path}/experiment_{experiment_num}"
    os.makedirs(output_folder, exist_ok=True)

    # Save parameters to the output folder
    try:
        with open(parameters_filename, 'r') as f:
            parameters = json.load(f)
        with open(f"{output_folder}/used_parameters.json", 'w') as f_out:
            json.dump(parameters, f_out, indent=4)
    except FileNotFoundError:
        print(f"Parameter file {parameters_filename} not found. No parameters saved.")

    return output_folder

def get_parameters(filename = 'parameters.json'):
    try:
        with open(filename, 'r') as f:
            parameters = json.load(f)
    except FileNotFoundError:
        print(f"Parameter file {filename} not found. Using default parameters.")
        parameters = {'country_codes': ['USA', 'DNK', 'NLD', "FDA"], 
                      'test_period': [1950, 2000],
                      'forecast_horizon': 10,
                      'model': 'ARIMA',
                      "parameters": {
                        "orders": [[2, 1, 2], [2, 1, 3], [3, 1, 2], [3, 1, 3], [1, 2, 1]]
                    }}       
    return parameters

def create_dataset(countries, test_period, forecast_horizon, dataset):
    df = pd.read_csv(dataset)
    # Create a dictionary of dataframes for each country with the test and train split
    country_dfs = {}
    print("Creating datasets for countries:", countries)
    for country in countries:
        country_data = df[df['countrycode'] == country].set_index('year')['rgdpe']
        train = country_data[(country_data.index >= test_period[0]) & (country_data.index < test_period[1])]
        test = country_data[(country_data.index >= test_period[1]) & (country_data.index < test_period[1] + forecast_horizon)]
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

def get_predictions(country_dfs, models, forecast_horizon = None, is_exponential = False):
    predictions = {}
    for country, data in country_dfs.items():
        models_to_evaluate = models[country]
        test = data['test']
        if type(models_to_evaluate) == list:
            predictions[country] = {}
            for model_fit in models_to_evaluate:
                forecast = model_fit.get_forecast(steps=forecast_horizon)
                order = model_fit.model.order
                predictions[country][order] = forecast.predicted_mean
        else:
            x = test.index.to_numpy()
            predictions[country] = models_to_evaluate.predict(x.reshape(-1, 1))
            if is_exponential:
                predictions[country] = np.exp(predictions[country])
    return predictions

def evaluate_models(country_dfs, predictions, used_model = None, output_folder='experiments'):
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
            is_dict = True
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
            is_dict = False
            mape = mean_absolute_percentage_error(test, predictions_for_country)
            mse = mean_squared_error(test, predictions_for_country)
            rmse = root_mean_squared_error(test, predictions_for_country)
            evaluation_metrics[country] = {'MSE': mse, 'RMSE': rmse, 'MAPE': mape}

    eval_df = pd.DataFrame.from_dict(evaluation_metrics, orient='index')
    
    # Calculate mean metrics across all countries
    if is_dict == False:
        # When metrics are directly stored (not lists)
        mean_metrics = eval_df.mean()
        mean_row = pd.DataFrame([mean_metrics], index=['Mean'])
        eval_df = pd.concat([eval_df, mean_row])
    else:
        # Group metrics by ARIMA order
        metrics_by_order = defaultdict(lambda: {'MSE': [], 'RMSE': [], 'MAPE': []})

        for country in evaluation_metrics:
            for model in evaluation_metrics[country]:
                order = model['ARIMA Order']
                metrics_by_order[order]['MSE'].append(model['MSE'])
                metrics_by_order[order]['RMSE'].append(model['RMSE'])
                metrics_by_order[order]['MAPE'].append(model['MAPE'])

        # Calculate means for each ARIMA order
        mean_metrics = {}
        for order, metrics in metrics_by_order.items():
            mean_metrics[order] = {
                'MSE': format(np.mean(metrics['MSE']), '.4f'),
                'RMSE': format(np.mean(metrics['RMSE']), '.4f'),
                'MAPE': format(np.mean(metrics['MAPE']), '.4f'),
        }
        mean_row = pd.DataFrame([mean_metrics], index=['Mean'])
        eval_df = pd.concat([eval_df, mean_row])
            
    eval_df.to_csv(f'{output_folder}/evaluation_metrics.csv')


def evaluate_models_cross_ARIMA(country_dfs, models, output_folder='experiments'):
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
                    'MSE (mean of all splits)': format(np.mean(mse_list), '.4f'),
                    'RMSE (mean of all splits)': format(np.mean(rmse_list), '.4f'),
                    'MAPE (mean of all splits)': format(np.mean(mape_list), '.4f')
                })
   
    eval_df = pd.DataFrame.from_dict(evaluation_metrics_cross, orient='index')
    eval_df.to_csv(f'{output_folder}/evaluation_metrics_cross.csv')

def evaluate_models_cross_linear(country_dfs, models, output_folder='experiments'):
    evaluation_metrics_cross_linear = {}
    for country, data in country_dfs.items():
        evaluation_metrics_cross_linear[country] = []
        full_series = data['train']._append(data['test'])
        n_splits = 5
        split_size = len(full_series) // n_splits

        for i in range(n_splits):
            train_end = (i + 1) * split_size
            train = full_series[:train_end]
            test = full_series[train_end:train_end + split_size]
            if len(test) == 0:
                continue
            
            X_train = train.index.to_numpy().reshape(-1, 1)
            y_train = train.values
            X_test = test.index.to_numpy().reshape(-1, 1)
            y_test = test.values

            try:
                linear_model_cv = LinearRegression().fit(X_train, y_train)
                predictions = linear_model_cv.predict(X_test)
                mse = mean_squared_error(y_test, predictions)
                rmse = root_mean_squared_error(y_test, predictions)
                mape = mean_absolute_percentage_error(y_test, predictions)
       
                evaluation_metrics_cross_linear[country].append ({
                    'MSE (split {})'.format(i+1): mse,
                    'RMSE (split {})'.format(i+1): rmse,
                    'MAPE (split {})'.format(i+1): mape
                })
            except Exception as e:
                print(f"Warning: Linear Regression model cross validation failed for {country} on split {i}: {e}")
                continue
        # add mean metrics across splits
        if evaluation_metrics_cross_linear[country]:
            # try to unscrew this stuff, i should be formatting here but i do some hacky shit with enumerate for the mean,
            # because we also include it  per split
            mse_mean = np.mean([m['MSE (split {})'.format(i+1)] for i, m in enumerate(evaluation_metrics_cross_linear[country])])
            rmse_mean = np.mean([m['RMSE (split {})'.format(i+1)] for i, m in enumerate(evaluation_metrics_cross_linear[country])])
            mape_mean = np.mean([m['MAPE (split {})'.format(i+1)] for i, m in enumerate(evaluation_metrics_cross_linear[country])])
            evaluation_metrics_cross_linear[country].append ({
                'MSE (mean of all splits)': format(mse_mean, '.4f'),
                'RMSE (mean of all splits)': format(rmse_mean, '.4f'),
                'MAPE (mean of all splits)': format(mape_mean, '.4f')
            })
    
    eval_df = pd.DataFrame.from_dict(evaluation_metrics_cross_linear, orient='index')
    eval_df.to_csv(f'{output_folder}/evaluation_metrics_cross_linear.csv')


def create_plots_arima(predictions, country_dfs, timerange, forecast_horizon, output_folder='experiments'):  
    for country in predictions.keys():
        orders = predictions[country].keys()
        num_models = len(orders)
        
        fig, axs = plt.subplots(num_models, 1, figsize=(10, 5 * num_models))
        
        if num_models == 1:
            axs = [axs]

        # create one plot
        i = 0
        for order in orders:
            # put plot in the figure
            forecast = predictions[country][order]
            forecast_index = np.arange(timerange[1], timerange[1] + forecast_horizon)
            
            # Plot forecast and observed data
            axs[i].plot(forecast_index, forecast, label=f'Forecast ARIMA{order}', linestyle='--')
            # Concatenate train and test for continuous line
            observed_data = pd.concat([country_dfs[country]['train'], country_dfs[country]['test']])
            observed_data.plot(ax=axs[i], label='Observed Data', color='blue')
            # Highlight test portion
            country_dfs[country]['test'].plot(ax=axs[i], label='Actual Data', color='red', linewidth=2)

            # make sure plot only shows from 2000 onwards
            axs[i].set_xlim([timerange[0], timerange[1] + forecast_horizon])
            # get rid of 1e6 notation on y-axis
            axs[i].ticklabel_format(style='plain', axis='y')
            # add title and labels
            axs[i].set_title(f'ARIMA{order} Forecast of rgdpe for {country}')
            axs[i].set_xlabel('Year')
            axs[i].set_ylabel('Real GDP (in millions)')
            axs[i].legend()
            i +=1
        # Save the plot to a file
        fig.savefig(f'{output_folder}/plots_{country}.png', dpi=300, bbox_inches='tight')
        print(f"Plots saved for {country}")
        plt.close(fig)

    # Plot 2: All countries with all their models
    num_countries = len(predictions.keys())
    fig, axs = plt.subplots(num_countries, 1, figsize=(12, 6 * num_countries))

    # Handle case where there's only one country
    if num_countries == 1:
        axs = [axs]

    for idx, country in enumerate(predictions.keys()):
        orders = predictions[country].keys()
        
        # Plot observed data
        observed_data = pd.concat([country_dfs[country]['train'], country_dfs[country]['test']])
        observed_data.plot(ax=axs[idx], label='Observed Data', linewidth=2, color='blue')
        # Highlight test portion
        country_dfs[country]['test'].plot(ax=axs[idx], label='Actual Data', color='red', linewidth=2)
        
        # Plot all models for this country
        for order in orders:
            forecast = predictions[country][order]
            forecast_index = np.arange(timerange[1], timerange[1] + forecast_horizon)
            axs[idx].plot(forecast_index, forecast, 
                        label=f'Forecast ARIMA{order}', linestyle='--')
        
        # Set x-axis limits and formatting
        axs[idx].set_xlim([timerange[0], timerange[1] + forecast_horizon])
        axs[idx].ticklabel_format(style='plain', axis='y')
        
        # Add title and labels
        axs[idx].set_title(f'All ARIMA Model Forecasts for {country}')
        axs[idx].set_xlabel('Year')
        axs[idx].set_ylabel('Real GDP (in millions)')
        axs[idx].legend()

    # Save the combined plot
    fig.savefig(f'{output_folder}/plots_all_countries.png', dpi=300, bbox_inches='tight')
    print("Combined plot saved for all countries: plots_all_countries.png")
    plt.close(fig)    

def calculate_recession_chances(country_dfs, country_models, output_folder='experiments'):
    recession_chances = {}
    for country, data in country_dfs.items():
        model_fits = country_models[country]
        recession_chances[country] = {}
        for model_fit in model_fits:
            order = model_fit.model.order
            # because we said we'd do 5 year forecasts
            forecast = model_fit.get_forecast(steps=5)
            forecast_values = forecast.predicted_mean
            # check for negative pct change
            recession_prob = np.mean(forecast_values.pct_change().dropna() < 0)
            recession_chances[country][order] = format(recession_prob, '.4f')
    
    recession_df = pd.DataFrame.from_dict(recession_chances, orient='index')
    recession_df.to_csv(f'{output_folder}/recession_chances.csv')


def calculate_recession_chances_linear(country_dfs, country_models, output_folder='experiments'):
    recession_chances = {}
    for country, data in country_dfs.items():
        print(f"Calculating recession chances for {country}")
        #print(country_models[country])
        model = country_models[country]
        # because we said we'd do 5 year forecasts
        last_year = data['train'].index[-1]
        forecast_years = np.arange(last_year + 1, last_year + 6).reshape(-1, 1)
        forecast_values = model.predict(forecast_years)
        # check for negative pct change
        recession_prob = np.mean(pd.Series(forecast_values).pct_change().dropna() < 0)
        recession_chances[country] = format(recession_prob, '.4f')  
    
    recession_df = pd.DataFrame.from_dict(recession_chances, orient='index', columns=['Recession Probability'])
    recession_df.to_csv(f'{output_folder}/recession_chances_linear.csv')

def create_plots(predictions, country_dfs, timerange, forecast_horizon, output_folder='experiments'):
    for country in predictions.keys():
        plt.figure(figsize=(10, 6))

        # Plot observed data
        observed_data = pd.concat([country_dfs[country]['train'], country_dfs[country]['test']])
        observed_data.plot(label='Observed Data', color='blue')
        # Highlight test portion
        country_dfs[country]['test'].plot(label='Actual Data', color='red', linewidth=2)

        forecast_index = np.arange(timerange[1], timerange[1] + forecast_horizon)
        plt.plot(forecast_index, predictions[country], label='Forecast Linear Regression', linestyle='--')
        plt.xlim([timerange[0], timerange[1] + forecast_horizon])
        plt.ticklabel_format(style='plain', axis='y')
        plt.title(f'Linear Regression Forecast of rgdpe for {country}')
        plt.xlabel('Year')
        plt.ylabel('Real GDP (in millions)')
        plt.legend()
        plt.savefig(f'{output_folder}/plots_{country}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved for {country}")


# def grid_search_best_ARIMA(country_dfs):
#     p_values = [0, 1, 2, 4, 6, 8, 10]
#     d_values = range(0, 3)
#     q_values = range(0, 3)
#     best_orders = {}
#    # use gridsearchcv from sklearn to find best arima order for each country
#     for country, data in country_dfs.items():
#         series = data['train']._append(data['test'])
#         best_aic = float("inf")
#         best_order = None
#         print(f"Starting grid search for {country}")
#         # use gridsearchcv from sklearn to find best arima order for each country
#         for p in p_values:
#             for d in d_values:
#                 for q in q_values:
#                     order = (p, d, q)
#                     try:
#                         model = ARIMA(series, order=order)
#                         model_fit = model.fit()
#                         aic = model_fit.aic
#                         if aic < best_aic:
#                             best_aic = aic
#                             best_order = order
#                     except Exception as e:
#                         print(f"Warning: ARIMA{order} model fitting failed for {country}: {e}")
#                         continue
#         best_orders[country] = {'Best ARIMA Order': best_order, 'Best AIC': format(best_aic, '.4f')}
#     best_orders_df = pd.DataFrame.from_dict(best_orders, orient='index')
#     best_orders_df.to_csv('best_arima_orders.csv')

if __name__ == "__main__":
    parameters_filename = 'parameters_from_gridsearch.json'  # Change this to switch parameter files

    output_folder = output_folder_setup(parameters_filename)
    parameters = get_parameters(parameters_filename)
    print(f"Parameters: {parameters}")
    country_dfs = create_dataset(parameters['country_codes'], parameters['test_period'], parameters['forecast_horizon'], dataset)

    match parameters['model']:
        case 'ARIMA':
            country_models = create_models_ARIMA(country_dfs, parameters['parameters'])
            predictions = get_predictions(country_dfs, country_models, forecast_horizon=parameters['forecast_horizon'])
            # print(predictions)
            create_plots_arima(predictions, country_dfs, parameters['test_period'], parameters['forecast_horizon'], output_folder=output_folder)
            # grid_search_best_ARIMA(country_dfs)
            evaluate_models(country_dfs, predictions, 'ARIMA', output_folder=output_folder)
            evaluate_models_cross_ARIMA(country_dfs, country_models, output_folder=output_folder)
            calculate_recession_chances(country_dfs, country_models, output_folder=output_folder)
        case 'Linear Regression':
            country_models = create_models_linear(country_dfs, parameters['parameters'])
            predictions = get_predictions(country_dfs, country_models, is_exponential=parameters['parameters']['is_exponential'])
            evaluate_models(country_dfs, predictions, country_models, output_folder=output_folder)
            evaluate_models_cross_linear(country_dfs, country_models, output_folder=output_folder)
            create_plots(predictions, country_dfs, parameters['test_period'], parameters['forecast_horizon'], output_folder=output_folder)
            calculate_recession_chances_linear(country_dfs, country_models, output_folder=output_folder)
        case _:
            raise ValueError(f"Model {parameters['model']} not recognized.")
    
    


 