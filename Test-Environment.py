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
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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
                      'gdp_used': 'original',
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
        country_data = df[df['countrycode'] == country].set_index('year')
        if parameters['gdp_used'] == 'log':
            country_data = np.log(country_data['rgdpe'])  # use log differenced data
        else:
            country_data = country_data['rgdpe']  # use original data
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


def diagnostic_analysis_pre_prediction(country_dfs, gdp_used, output_folder='experiments'):
   # create diagnostics subfolder under output_folder
    os.makedirs(f'{output_folder}/diagnostics', exist_ok=True)
    
    for country, data in country_dfs.items():
        # Combine train and test data
        full_data = data['train']._append(data['test'])
        
        # Create diagnostic plots
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # 1. Time series plot
        axes[0].plot(full_data.index, full_data.values)
        axes[0].set_title(f'Time Series - {country}')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel(f"{'log of Real GDP' if gdp_used == 'log' else 'Real GDP'}")
        
        # 2. QQ-plot
        qqplot(full_data.values, line='s', ax=axes[1])
        axes[1].set_title(f'Q-Q Plot - {country}')
        
        # 3. Histogram
        axes[2].hist(full_data.values, bins=20, edgecolor='black')
        axes[2].set_title(f'Distribution - {country}')
        axes[2].set_xlabel(f"{'log of Real GDP' if gdp_used == 'log' else 'Real GDP'}")
        axes[2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f'{output_folder}/diagnostics/raw_data_diagnostics_{country}.png', dpi=300)
        plt.close()
        
        # Shapiro-Wilk test for normality
        shapiro_test = stats.shapiro(full_data.values)
        with open(f'{output_folder}/diagnostics/shapiro_wilk_{country}.txt', 'w') as f:
            f.write(f"Shapiro-Wilk test results for {country}:\n")
            f.write(f"Statistic: {shapiro_test.statistic}, p-value: {shapiro_test.pvalue}\n")
            f.close()

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
                predictions[country][order] = {
                    "forecast": forecast,
                    "mean": forecast.predicted_mean,
                }
        else:
            x = test.index.to_numpy()
            predictions[country] = models_to_evaluate.predict(x.reshape(-1, 1))
            if is_exponential:
                predictions[country] = np.exp(predictions[country])
    return predictions

def evaluate_models(country_dfs, predictions, models, used_model = None, output_folder='experiments'):
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
                mape = mean_absolute_percentage_error(test, preds["mean"])
                mse = mean_squared_error(test, preds["mean"])
                rmse = root_mean_squared_error(test, preds["mean"])
                if used_model == 'ARIMA' and models:
                    # Find the corresponding model to get AIC and BIC
                    model_fit = next(m for m in models[country] if m.model.order == key)
                    aic = float(model_fit.aic) #np.float64, convert to 4 decimal places
                    bic = float(model_fit.bic)
                    evaluation_metrics[country].append({'ARIMA Order': key, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'AIC': aic, 'BIC': bic})
                elif used_model == 'ARIMA':
                    evaluation_metrics[country].append({'ARIMA Order': key, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape})
                else:
                    evaluation_metrics[country].append({'MSE': format(mse, '.4f'), 'RMSE': format(rmse, '.4f'), 'MAPE': format(mape, '.4f')})
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
        metrics_by_order = defaultdict(lambda: {'MSE': [], 'RMSE': [], 'MAPE': [], 'AIC': [], 'BIC': []})

        for country in evaluation_metrics:
            for model in evaluation_metrics[country]:
                order = model['ARIMA Order']
                metrics_by_order[order]['MSE'].append(model['MSE'])
                metrics_by_order[order]['RMSE'].append(model['RMSE'])
                metrics_by_order[order]['MAPE'].append(model['MAPE'])
                if 'AIC' in model:
                    metrics_by_order[order]['AIC'].append(model['AIC'])
                if 'BIC' in model:
                    metrics_by_order[order]['BIC'].append(model['BIC'])

        # Calculate means for each ARIMA order
        mean_metrics = {}
        for order, metrics in metrics_by_order.items():
            mean_metrics[order] = {
                'MSE (mean of all countries)' : format(np.mean(metrics['MSE']), '.4f'),
                'RMSE (mean of all countries)': format(np.mean(metrics['RMSE']), '.4f'),
                'MAPE (mean of all countries)': format(np.mean(metrics['MAPE']), '.4f'),
            }
            if metrics['AIC']:
                mean_metrics[order]['AIC (mean of all countries)'] = format(np.mean(metrics['AIC']), '.4f')
            if metrics['BIC']:
                mean_metrics[order]['BIC (mean of all countries)'] = format(np.mean(metrics['BIC']), '.4f')
        mean_row = pd.DataFrame([mean_metrics], index=['Mean'])
        eval_df = pd.concat([eval_df, mean_row])
        # make a nice table of mean metrics for each ARIMA order: per country, and make sure the ARIMA order has "ARIMA order" as column name
        # include some | formatting for better readability
        mean_metrics_df = pd.DataFrame.from_dict(mean_metrics, orient='index')
        mean_metrics_df.index.name = "ARIMA Order"
        print("Mean metrics for each ARIMA order:")
        # save print to markdown file
        with open(f'{output_folder}/mean_metrics_per_arima_order.md', 'w') as f:
            f.write(mean_metrics_df.to_markdown())
            f.close()

        for country in evaluation_metrics:
            # format metrics to 4 decimal places
            metrics = evaluation_metrics[country]
            
            for model in metrics:
                for key in model:
                    if key != 'ARIMA Order':
                        model[key] = format(model[key], '.4f')
            
                        
            with open(f'{output_folder}/metrics_per_arima_order_{country}.md', 'w') as f:
                country_metrics_df = pd.DataFrame(metrics)
                country_metrics_df = country_metrics_df.set_index('ARIMA Order')
                f.write(country_metrics_df.to_markdown())
                f.close()
    
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
                    'MAPE (mean of all splits)': format(np.mean(mape_list), '.4f'),
                    #format to the same number of decimals
                    "RMSE (splits)": [format(rmse, '.4f') for rmse in rmse_list],
                    "MAPE (splits)": [format(mape, '.4f') for mape in mape_list]
                })
   
    
    eval_df = pd.DataFrame.from_dict(evaluation_metrics_cross, orient='index')
    eval_df.to_csv(f'{output_folder}/evaluation_metrics_cross.csv')
    return evaluation_metrics_cross

def diagnostic_analysis_ARIMA(country_dfs, country_models, output_folder='experiments'):
   # create diagnostics subfolder under output_folder
    os.makedirs(f'{output_folder}/diagnostics', exist_ok=True)
    
    for country, models in country_models.items():
        for model_fit in models:
            order = model_fit.model.order
            
            # Extract residuals
            residuals = model_fit.resid
            
            # Create diagnostic plots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # 1. Residuals over time
            axes[0, 0].plot(residuals)
            axes[0, 0].set_title(f'Residuals - ARIMA{order} - {country}')
            axes[0, 0].axhline(y=0, linestyle='--', color='red')
            
            # 2. QQ-plot
            qqplot(residuals, line='s', ax=axes[0, 1])
            axes[0, 1].set_title(f'Q-Q Plot - ARIMA{order} - {country}')
            
            # 3. Histogram of residuals
            axes[1, 0].hist(residuals, bins=20, edgecolor='black')
            axes[1, 0].set_title(f'Residual Distribution - ARIMA{order} - {country}')
            
            # 4. ACF of residuals
            plot_acf(residuals, ax=axes[1, 1], lags=20)
            axes[1, 1].set_title(f'ACF of Residuals - ARIMA{order} - {country}')
            
            plt.tight_layout()
            plt.savefig(f'{output_folder}/diagnostics/diagnostics_{country}_ARIMA{order}.png', dpi=300)
            plt.close()

            # 5. Ljung-Box test
            lb_test = sm.stats.acorr_ljungbox(residuals, lags=[10], return_df=True)
            with open(f'{output_folder}/diagnostics/ljung_box_{country}_ARIMA{order}.txt', 'w') as f:
                f.write(f"Ljung-Box test results for {country} ARIMA{order}:\n")
                f.write(lb_test.to_string())
                f.close()
            
            # 6. Shapiro-Wilk test for normality
            shapiro_test = stats.shapiro(residuals)
            with open(f'{output_folder}/diagnostics/shapiro_wilk_{country}_ARIMA{order}.txt', 'w') as f:
                f.write(f"Shapiro-Wilk test results for {country} ARIMA{order}:\n")
                f.write(f"Statistic: {shapiro_test.statistic}, p-value: {shapiro_test.pvalue}\n")
                f.close()

            

def plot_cross_validation_metrics(evaluations, country_selected, output_folder='experiments'):
    orders = [model['ARIMA Order'] for model in evaluations[country_selected]]
    # rmse per split
    rmse_per_split = [list(map(float, model['RMSE (splits)'])) for model in evaluations[country_selected]]
    # mape * 100 to get actual percentage
    mape_per_split = [list(map(lambda x: float(x) * 100, model['MAPE (splits)'])) for model in evaluations[country_selected]]


    # Plot RMSE per split
    # 4 plots: 1 per ARIMA order
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    # scale to 1.2 times the max value for better visibility
    max_rmse = max([max(rmse_list) for rmse_list in rmse_per_split])
    axs = axs.flatten()
    for i, order in enumerate(orders):
        axs[i].bar(range(1, len(rmse_per_split[i]) + 1), rmse_per_split[i])
        axs[i].set_title(f'RMSE per Split for ARIMA{order} - {country_selected}')
        axs[i].set_xlabel('Split Number')
        axs[i].set_ylabel('RMSE')
        # turn off scientific notation, with formatter set_scientific(False)
        axs[i].yaxis.get_major_formatter().set_scientific(False)
        axs[i].set_ylim(0, max_rmse * 1.2)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/rmse_per_split_{country_selected}.png')
    plt.close()

    # Plot MAPE per split
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()
    max_mape = max([max(mape_list) for mape_list in mape_per_split])
    for i, order in enumerate(orders):
        axs[i].bar(range(1, len(mape_per_split[i]) + 1), mape_per_split[i])
        axs[i].set_title(f'MAPE per Split for ARIMA{order} - {country_selected}')
        axs[i].set_xlabel('Split Number')
        axs[i].yaxis.get_major_formatter().set_scientific(False)
        axs[i].set_ylabel('MAPE')
        axs[i].set_ylim(0, max_mape * 1.2)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/mape_per_split_{country_selected}.png')
    plt.close()


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
            forecast = predictions[country][order]["forecast"]
            forecast_mean = predictions[country][order]["mean"]
            forecast_index = np.arange(timerange[1], timerange[1] + forecast_horizon)
            
            # Plot forecast and observed data
            # 95% confidence intervals
            conf_int = forecast.conf_int(alpha=0.05)
            axs[i].fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='gray', alpha=0.3, label='95% Confidence Interval')
            axs[i].plot(forecast_index, forecast_mean, label=f'Forecast ARIMA{order}', linestyle='--')
            
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
            axs[i].set_title(f'ARIMA{order} Forecast of {parameters["gdp_used"]} rgdpe for {country}')
            axs[i].set_xlabel('Year')
            axs[i].set_ylabel(f'{"log of" if parameters["gdp_used"] == "log" else ""} Real GDP (in millions)')
            axs[i].legend()
            i +=1
        # Save the plot to a file
        fig.savefig(f'{output_folder}/plots_{country}.png', dpi=300, bbox_inches='tight')
        print(f"Plots saved for {country}")
        plt.close(fig)

        fig2, axs2 = plt.subplots(num_models, 1, figsize=(10, 5 * num_models))

        if num_models == 1:
            axs2 = [axs2]
        
        # create one plot
        j = 0 
        for order in orders:
            # make sure plot shows test period - 20 years until forecast horizon
            forecast_mean = predictions[country][order]["mean"]
            forecast_index = np.arange(timerange[1], timerange[1] + forecast_horizon)
            start_year = max(timerange[0], timerange[1] - 20)
            end_year = timerange[1] + forecast_horizon
            print(f"Start year: {start_year}, End year: {end_year}")
      
            # Plot forecast and observed data
            axs2[j].fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='gray', alpha=0.3, label='95% Confidence Interval')
            axs2[j].plot(forecast_index, forecast_mean, label=f'Forecast ARIMA{order}', linestyle='--')
            # Concatenate train and test for continuous line
            observed_data = pd.concat([country_dfs[country]['train'], country_dfs[country]['test']])
            observed_data.plot(ax=axs2[j], label='Observed Data', color='blue')
            # Highlight test portion
            country_dfs[country]['test'].plot(ax=axs2[j], label='Actual Data', color='red', linewidth=2)
            # Set zoom window
            observed_subset = observed_data[(observed_data.index >= start_year) & (observed_data.index <= end_year)]
            axs2[j].set_ylim([observed_subset.min() * 0.95, observed_subset.max() * 1.05])
            axs2[j].set_xlim([start_year, end_year])
            # make sure plot only shows from start_year until forecast horizon
            # get rid of 1e6 notation on y-axis
            axs2[j].ticklabel_format(style='plain', axis='y')
            # add title and labels
            axs2[j].set_title(f'ARIMA{order} Forecast of {parameters["gdp_used"]} rgdpe for {country} (Zoomed In)')
            axs2[j].set_xlabel('Year')
            axs2[j].set_ylabel(f'{"log of" if parameters["gdp_used"] == "log" else ""} Real GDP (in millions)')
            axs2[j].legend()
            j +=1
        # Save the plot to a file
        fig2.savefig(f'{output_folder}/plots_{country}_zoomed.png', dpi=300, bbox_inches='tight')
        print(f"Zoomed plots saved for {country}")
        plt.close(fig2)
        


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
            forecast = predictions[country][order]['forecast']
            forecast_mean = predictions[country][order]['mean']
            forecast_index = np.arange(timerange[1], timerange[1] + forecast_horizon)

            # plot forecast and forecast mean
            conf_int = forecast.conf_int(alpha=0.05)
            axs[idx].fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], alpha=0.2)
            axs[idx].plot(forecast_index, forecast_mean, 
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

def calculate_recession_chances(country_dfs, country_models, timerange, forecast_horizon, output_folder='experiments'):
    os.makedirs(f'{output_folder}/recession-chances', exist_ok=True)
    recession_chances = {}
    # for country, data in country_dfs.items():
    #     model_fits = country_models[country]
    #     recession_chances[country] = {}
    #     for model_fit in model_fits:
    #         order = model_fit.model.order
    #         # because we said we'd do 5 year forecasts
    #         forecast = model_fit.get_forecast(steps=5)
    #         forecast_values = forecast.predicted_mean
    #         # check for negative pct change
    #         recession_prob = np.mean(forecast_values.pct_change().dropna() < 0)
    #         recession_chances[country][order] = format(recession_prob, '.4f')

    # calulate recession chances based on the best model per country (based on AIC)
    # create output folder for recesson prediction
    
    start_year = timerange[1]       
    end_year = timerange[1] + forecast_horizon


    for country, data in country_dfs.items():
        recession_chances[country] = []
        model_fits = country_models[country]
        for model in model_fits:
            order = model.model.order
            print(f"Evaluating ARIMA{order} model for recession chances in {country}")
            # because we said we'd do 5 year forecasts
            forecast = model.get_forecast(steps=5)
            # based on "shape" / slope of the confidence interval, maybe we can determine recession chances?
            conf_int = forecast.conf_int(alpha=0.05)
            # get slope of upper bound
            confidence_upper_bound_slope = (conf_int.iloc[-1, 1] - conf_int.iloc[0, 1]) / 5  # over 5 years
            # get slope of lower bound
            confidence_lower_bound_slope = (conf_int.iloc[-1, 0] - conf_int.iloc[0, 0]) / 5  # over 5 years
            confidence_total_slope = (confidence_upper_bound_slope + confidence_lower_bound_slope) / 2
            print(f"Confidence interval slopes for ARIMA{order} in {country}: upper={confidence_upper_bound_slope}, lower={confidence_lower_bound_slope}, total={confidence_total_slope}")
            # if the total slope is negative, we can say there's a high chance of recession
            # this is needed because confidence_total_slope is float and you cannot do == 0 on floats (at least reliably)
            if np.isclose(confidence_total_slope, 0):
                recession_prob = "No recession predicted (total slope is zero)"
            elif confidence_total_slope < 0:
                recession_prob = "High chance (predicted shrinkage)"
            elif confidence_total_slope > 0:
                recession_prob = "Low chance (predicted growth)"
            
            recession_chances[country].append({
                'ARIMA Order': order,
                'Slope of Confidence Interval (upper bound)': format(confidence_upper_bound_slope, '.4f'),
                'Slope of Confidence Interval (lower bound)': format(confidence_lower_bound_slope, '.4f'),
                'Total Slope of Confidence Interval': format(confidence_total_slope, '.4f'),
                'Recession Probability': recession_prob
            })

    for country in recession_chances:
        recession_df = pd.DataFrame(recession_chances[country])
        recession_df = recession_df.set_index('ARIMA Order')
        with open(f'{output_folder}/recession-chances/{country}.md', 'w') as f:
            # write period of forecast
            f.write(f"# Recession Chances for {country}\n")
            f.write(f"## Forecast Period: {start_year} to {end_year}\n\n")
            f.write(recession_df.to_markdown())
            f.close()
            

  


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
            diagnostic_analysis_pre_prediction(country_dfs, output_folder=output_folder)
            predictions = get_predictions(country_dfs, country_models, forecast_horizon=parameters['forecast_horizon'])
            # print(predictions)
            create_plots_arima(predictions, country_dfs, parameters['test_period'], parameters['forecast_horizon'], output_folder=output_folder)
            # grid_search_best_ARIMA(country_dfs)
            evaluate_models(country_dfs, predictions, models=country_models, used_model='ARIMA', output_folder=output_folder)
            evaluations = evaluate_models_cross_ARIMA(country_dfs, country_models, output_folder=output_folder)
            diagnostic_analysis_ARIMA(country_dfs, country_models, output_folder=output_folder)
            plot_cross_validation_metrics(evaluations, parameters['country_codes'][0], output_folder=output_folder)
            calculate_recession_chances(country_dfs, country_models, parameters["test_period"], parameters["forecast_horizon"], output_folder=output_folder)
        case 'Linear Regression':
            country_models = create_models_linear(country_dfs, parameters['parameters'])
            predictions = get_predictions(country_dfs, country_models, is_exponential=parameters['parameters']['is_exponential'])
            evaluate_models(country_dfs, predictions, country_models, output_folder=output_folder)
            evaluate_models_cross_linear(country_dfs, country_models, output_folder=output_folder)
            create_plots(predictions, country_dfs, parameters['test_period'], parameters['forecast_horizon'], output_folder=output_folder)
            calculate_recession_chances_linear(country_dfs, country_models, output_folder=output_folder)
        case _:
            raise ValueError(f"Model {parameters['model']} not recognized.")




