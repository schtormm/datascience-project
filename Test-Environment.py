import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict

from statsmodels.graphics.api import qqplot

import pmdarima as pm
from pmdarima.model_selection import train_test_split

dataset = 'cleaned_V11.csv'
parameters = {'country': ['USA', 'DNK', 'NLD'], 
              'test_period': [1950, 2000],
              'forecast_horizon': 10,
              'model': 'ARIMA',
              'arima_orders': [(2, 1, 1), (2, 1, 3), (3, 1, 2)]}

def get_parameters():
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

def evaluate_models():
    pass

def create_plots(models, country_dfs, timerange, forecast_horizon):
    country = 'USA'  # Example: using the first country
    model = models[country]
    plot_predict(model, start=timerange[1], end=(timerange[1] + forecast_horizon), ax=None)
    country_dfs[country]['train'].plot(label='Observed')
    # make sure plot only shows from 2000 onwards
    plt.xlim([timerange[0], timerange[1] + forecast_horizon])
    # get rid of 1e6 notation on y-axis
    plt.ticklabel_format(style='plain', axis='y')
    # add title and labels
    plt.title(f'ARIMA(3,0,1) Forecast of rgdpe for {country}')
    plt.xlabel('Year')
    plt.ylabel('Real GDP (in millions)')
    plt.legend()
    plt.show()  



if __name__ == "__main__":
    country_dfs = create_dataset(parameters['country'], parameters['test_period'], dataset)
    arima_order = parameters['arima_orders'][0]  # Example: using the first ARIMA order
    country_models = create_models(country_dfs, parameters['model'], arima_order)
    create_plots(country_models, country_dfs, parameters['test_period'], parameters['forecast_horizon'])