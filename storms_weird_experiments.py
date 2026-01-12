import os

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import statsmodels as sm
import statsmodels.api as sm_api
from ssalib import SingularSpectrumAnalysis
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
from pmdarima.arima import auto_arima
from pmdarima.model_selection import train_test_split
from statsmodels.tsa.stattools import kpss


df = pd.read_csv('cleaned_V11.csv')
# create folder "storms_plots" if it doesn't exist
# create file structure: "storms_plots/[country]/raw/ and "storms_plots/[country]/arima/"

countries =  ["Netherlands", "United States"]
[country.lower() for country in countries]
print(countries)

for country in countries:
    if not os.path.exists(f'storms_plots/{country}/raw/'):
        os.makedirs(f'storms_plots/{country}/raw/')
    if not os.path.exists(f'storms_plots/{country}/transformations/'):
        os.makedirs(f'storms_plots/{country}/transformations/')
    if not os.path.exists(f'storms_plots/{country}/arima/'):
        os.makedirs(f'storms_plots/{country}/arima/')
    if not os.path.exists('storms_plots/fake_data_plots/'):
        os.makedirs('storms_plots/fake_data_plots/')
    if not os.path.exists(f'storms_plots/{country}/pacf_acf'):
        os.makedirs(f'storms_plots/{country}/pacf_acf')
    



df["rgdpe_diff"] = df.groupby('country')['rgdpe'].diff()
df["rgdpe_diff2"] = df.groupby('country')['rgdpe_diff'].diff() 
df['rgdpe_log'] = np.log(df['rgdpe'])
df['rgdpe_log_diff'] = df.groupby('country')['rgdpe_log'].diff()





# plot rdgpe for Netherlands
netherlands_data = df[(df['country'] == 'Netherlands') & (df['year'] > 1950)][['year', 'rgdpe', 'rgdpe_diff', 'rgdpe_diff2', 'rgdpe_log', 'rgdpe_log_diff']]
# convert rdgpe to 2021 USD instead of 2021 USD millions
plt.figure(figsize=(10,6))
plt.plot(netherlands_data['year'], netherlands_data['rgdpe'], label='RDGPE (in trillions of 2021 USD)', color='blue')
plt.title('RGDPE for the Netherlands, 1950-2023')
plt.xlabel('Year')
# mark 2007-2009 financial crisis
plt.axvspan(2007, 2009, color='red', alpha=0.3, label='2008-2009 Financial Crisis')
plt.ylabel('RGDPE (in trillions of 2021 USD)')
plt.savefig('storms_plots/netherlands/raw/netherlands_rdgpe.png')
plt.close()


# do adf test on raw data for Netherlands
adf_result = adfuller(netherlands_data['rgdpe'])
print('ADF Statistic for Netherlands rdgpe: %f' % adf_result[0])
print('p-value: %f' % adf_result[1])
print('Critical Values:')
for key, value in adf_result[4].items():
    print('\t%s: %.3f' % (key, value))
if adf_result[1] < 0.05:
    print("ADF test indicates the series is stationary. (p-value < 0.05)")
elif adf_result[1] >= 0.05:
    print("ADF test indicates the series is non-stationary. (p-value >= 0.05)")

kpss_result = kpss(netherlands_data['rgdpe'], regression='c')
print('KPSS Statistic for Netherlands rdgpe: %f' % kpss_result[0])
print('p-value: %f' % kpss_result[1])
print('Critical Values:')
for key, value in kpss_result[3].items():
    print('\t%s: %.3f' % (key, value))
if kpss_result[1] < 0.05:
    print("KPSS test indicates the series is non-stationary. (p-value < 0.05)")
elif kpss_result[1] >= 0.05:
    print("KPSS test indicates the series is stationary. (p-value >= 0.05)")


fig, ax = plt.subplots(2,1, figsize=(10,8))
plot_acf(netherlands_data['rgdpe'], ax=ax[0], lags=30)
plot_pacf(netherlands_data['rgdpe'], ax=ax[1], lags=30)
plt.suptitle('ACF and PACF of rdgpe for Netherlands', fontsize=16)
plt.savefig('storms_plots/Netherlands/pacf_acf/rdgpe_acf.png')
plt.close()


# plot differenced data for Netherlands
netherlands_diff = netherlands_data['rgdpe_diff']
# print netherlands diff in 2023
print("Netherlands differenced rdgpe in 2023:", netherlands_diff[netherlands_data['year'] == 2023].values)
plt.figure(figsize=(10,6))
# scale down to -80000 on y axis
plt.ylim(-80000, 80000)
plt.plot(netherlands_data['year'], netherlands_diff, label='Netherlands Differenced')
plt.title('Differenced rdgpe for Netherlands')
plt.xlabel('Year')
plt.ylabel('Differenced rdgpe')
plt.savefig('storms_plots/netherlands/transformations/netherlands_rdgpe_differenced.png')
plt.close()

# do adf test on differenced data for Netherlands
adf_result_diff = adfuller(netherlands_diff)
print('ADF Statistic for Netherlands Differenced rdgpe: %f' % adf_result_diff[0])
print('p-value: %f' % adf_result_diff[1])
print('Critical Values:')
for key, value in adf_result_diff[4].items():
    print('\t%s: %.3f' % (key, value))
if adf_result_diff[1] < 0.05:
    print("ADF test indicates the series is stationary. (p-value < 0.05)")
elif adf_result_diff[1] >= 0.05:
    print("ADF test indicates the series is non-stationary. (p-value >= 0.05)")

# also do kpss test

kpss_result_diff = kpss(netherlands_diff.dropna(), regression='c')
print('KPSS Statistic for Netherlands Differenced rdgpe: %f' % kpss_result_diff[0])
print('p-value: %f' % kpss_result_diff[1])
print('Critical Values:')
for key, value in kpss_result_diff[3].items():
    print('\t%s: %.3f' % (key, value))
if kpss_result_diff[1] < 0.05:
    print("KPSS test indicates the series is non-stationary. (p-value < 0.05)")
elif kpss_result_diff[1] >= 0.05:
    print("KPSS test indicates the series is stationary. (p-value >= 0.05)")


# # plot log data for Netherlands
netherlands_log = netherlands_data['rgdpe_log']
plt.figure(figsize=(10,6))
plt.plot(netherlands_data['year'], netherlands_log, label='Netherlands Log rdgpe')
plt.title('Log rdgpe for Netherlands')
plt.xlabel('Year')
plt.ylabel('Log rdgpe')
plt.savefig('storms_plots/netherlands/transformations/rdgpe_log.png')
plt.close()


# plot log differenced data for Netherlands
netherlands_log_diff = netherlands_data['rgdpe_log_diff']
plt.figure(figsize=(10,6))
plt.ylim(-0.15, 0.15)
plt.plot(netherlands_data['year'], netherlands_log_diff, label='Netherlands Log Differenced')
plt.title('Log Differenced rdgpe for Netherlands')
plt.xlabel('Year')
plt.ylabel('Log Differenced rdgpe')
plt.savefig('storms_plots/netherlands/transformations/rdgpe_log_differenced.png')

# do ADF test on log differenced data for Netherlands
adf_result_log = adfuller(netherlands_log_diff)
print('ADF Statistic for Netherlands Log Differenced rdgpe: %f' % adf_result_log[0])
print('p-value: %f' % adf_result_log[1])
print('Critical Values:')
for key, value in adf_result_log[4].items():
    print('\t%s: %.3f' % (key, value))
if adf_result_log[1] < 0.05:
    print("ADF test indicates the series is stationary. (p-value < 0.05)")
elif adf_result_log[1] >= 0.05:
    print("ADF test indicates the series is non-stationary. (p-value >= 0.05)")

kpss_result_log_diff = kpss(netherlands_log_diff.dropna(), regression='c')
print('KPSS Statistic for Netherlands Log Differenced rdgpe: %f' % kpss_result_log_diff[0])
print('p-value: %f' % kpss_result_log_diff[1])
print('Critical Values:')
for key, value in kpss_result_diff[3].items():
    print('\t%s: %.3f' % (key, value))
if kpss_result_log_diff[1] < 0.05:
    print("KPSS test indicates the series is non-stationary. (p-value < 0.05)")
elif kpss_result_log_diff[1] >= 0.05:
    print("KPSS test indicates the series is stationary. (p-value >= 0.05)")

#plot acf and pacf for log data for Netherlands
fig, ax = plt.subplots(2,1, figsize=(10,8))
plot_acf(netherlands_data['rgdpe_log'], ax=ax[0], lags=30)
plot_pacf(netherlands_data['rgdpe_log'], ax=ax[1], lags=30)
plt.suptitle('ACF and PACF of Log rdgpe for Netherlands', fontsize=16)
plt.savefig('storms_plots/Netherlands/pacf_acf/rdgpe_log_acf_pacf.png')
plt.close()


# plot acf and pacf for log diff data for Netherlands
fig, ax  = plt.subplots(2,1, figsize=(10,8))
plot_acf(netherlands_log_diff, ax=ax[0], lags=30)
plot_pacf(netherlands_log_diff, ax=ax[1], lags=30)
plt.suptitle('ACF and PACF of Log Differenced rdgpe for Netherlands', fontsize=16)
plt.savefig('storms_plots/Netherlands/pacf_acf/rdgpe_log_differenced_acf_pacf.png')
plt.close()




train, test = train_test_split(netherlands_data['rgdpe_log_diff'], test_size=0.2)
model = ARIMA(order=(1,2,1))
print(model.summary())
preds, conf_int = model.predict(n_periods=test.shape[0], return_conf_int=True)

print(f"RMSE on test data: %.3f" % np.sqrt(mean_squared_error(test, preds)))
print(preds)
# show prediction overall
plt.figure(figsize=(10,6))
plt.plot(netherlands_data['year'], netherlands_data['rgdpe'], label='Actual rdgpe', color='blue')
# make sure test uses the correct years (not just index 0,1,2...)
plt.plot(netherlands_data['year'].iloc[-test.shape[0]:], preds, label='ARIMA Predictions', color='red', linestyle='--')
plt.fill_between(netherlands_data['year'].iloc[-test.shape[0]:], conf_int[:,0], conf_int[:,1], color='pink', alpha=0.3, label='Confidence Interval')
plt.title('ARIMA Model Predictions vs Actual rdgpe for Netherlands')
plt.xlabel('Year')
plt.gca().yaxis.get_major_formatter().set_scientific(False)
plt.ylabel('rdgpe')
plt.legend()
plt.show()



# do adf test on second order differenced data for Netherlands

adf_result2 = adfuller(netherlands_data['rgdpe_diff2'].dropna())
print('ADF Statistic for Netherlands 2nd Order Differenced rdgpe: %f' % adf_result2[0])
print('p-value: %f' % adf_result2[1])
print('Critical Values:')
for key, value in adf_result2[4].items():
    print('\t%s: %.3f' % (key, value))
if adf_result2[1] < 0.05:
    print("ADF test indicates the series is stationary. (p-value < 0.05)")
elif adf_result2[1] >= 0.05:
    print("ADF test indicates the series is non-stationary. (p-value >= 0.05)")

# do kpss test on second order differenced data for Netherlands
kpss_result2 = kpss(netherlands_data['rgdpe_diff2'].dropna(), regression='c')
print('KPSS Statistic for Netherlands 2nd Order Differenced rdgpe: %f' % kpss_result2[0])
print('p-value: %f' % kpss_result2[1])
print('Critical Values:')
for key, value in kpss_result2[3].items():
    print('\t%s: %.3f' % (key, value))
if kpss_result2[1] < 0.05:
    print("KPSS test indicates the series is non-stationary. (p-value < 0.05)")
elif kpss_result2[1] >= 0.05:
    print("KPSS test indicates the series is stationary. (p-value >= 0.05)")

netherlands_diff2 = netherlands_data[['year', 'rgdpe_diff2']]
netherlands_diff2_clean = netherlands_diff2['rgdpe_diff2'].dropna()
plt.figure(figsize=(10,6))
plt.plot(netherlands_diff2['year'], netherlands_diff2['rgdpe_diff2'], label='Netherlands 2nd Order Differenced')
plt.title('2nd Order Differenced rdgpe for Netherlands')
plt.xlabel('Year')
plt.ylabel('2nd Order Differenced rdgpe')
plt.xlabel('Year')
plt.ylabel('Differenced rdgpe')
plt.savefig('storms_plots/Netherlands/transformations/rdgpe_2ndorder_differenced.png')
plt.close()



#plot acf and pacf of second order differenced data for Netherlands
fig, ax = plt.subplots(2,1, figsize=(10,8))
plot_acf(netherlands_diff2_clean, ax=ax[0], lags=30)
plot_pacf(netherlands_diff2_clean, ax=ax[1], lags=30)
plt.suptitle('ACF and PACF of 2nd Order Differenced rdgpe for Netherlands', fontsize=16)
plt.savefig('storms_plots/Netherlands/pacf_acf/rdgpe_2ndorder_differenced_acf_pacf.png')
plt.close()



# # plot predictions vs actual
# plt.figure(figsize=(10,6))
# plt.plot(netherlands_data_test_x, netherlands_data_test_y, label='Actual rdgpe', color='blue')
# plt.plot(netherlands_data_test_x, predictions, label='ARIMA Predictions', color='red', linestyle='--')
# plt.title('ARIMA Model Predictions vs Actual rdgpe for Netherlands')
# plt.xlabel('Year')
# plt.gca().yaxis.get_major_formatter().set_scientific(False)
# plt.ylabel('rdgpe')
# plt.legend()
# plt.savefig('storms_plots/netherlands/arima/netherlands_rdgpe_arima_predictions.png')
# plt.close()


# # try to fit ARIMA on differenced data for Netherlands
# netherlands_diff_clean = netherlands_diff.dropna()
# netherlands_years_clean = netherlands_data['year'][netherlands_diff_clean.index]




# # qq plot of residuals
# sm_api.qqplot(residuals, line='s')
# plt.title('QQ Plot of Residuals of ARIMA model on Differenced rdgpe for Netherlands')
# plt.savefig('storms_plots/netherlands_rdgpe_differenced_arima_qqplot.png')


# # try so see if differencing works for USA as well
# usa_diff = df[df['country'] == 'United States'][['year', 'rgdpe']]
# usa_diff['rgdpe_diff'] = usa_diff['rgdpe'].diff()
# plt.figure(figsize=(10,6))
# plt.plot(usa_diff['year'], usa_diff['rgdpe_diff'], label='USA Differenced', color='green')
# plt.title('Differenced rdgpe for USA')
# plt.xlabel('Year')
# plt.ylabel('Differenced rdgpe')
# # mark 2007-2009 financial crisis
# plt.axvspan(2007, 2009, color='red', alpha=0.3, label='2008-2009 Financial Crisis')
# # mark oil crisis 1973
# plt.axvspan(1973, 1974, color='orange', alpha=0.3, label='1973 Oil Crisis')
# # mark dotcom bubble 2000-2001
# plt.axvspan(2000, 2001, color='green', alpha=0.3, label='2000-2001 Dotcom Bubble')
# plt.legend()
# # mark covid-19 recession 2020-onwards
# plt.axvspan(2020, 2023, color='purple', alpha=0.3, label='2020 Covid-19 Recession')
# plt.savefig('storms_plots/USA_rdgpe_differenced.png')

# # try second order differencing for Netherlands


    
# #do adf test on 2nd order log differenced data for Netherlands
# df['rgdpe_log_diff2'] = df.groupby('country')['rgdpe_log_diff'].diff()
# netherlands_log_diff2_clean = df[df['country'] == 'Netherlands']['rgdpe_log_diff2'].dropna()
# adf_result_log2 = adfuller(netherlands_log_diff2_clean)
# print('ADF Statistic for Netherlands 2nd Order Log Differenced rdgpe: %f' % adf_result_log2[0])
# print('p-value: %f' % adf_result_log2[1])
# print('Critical Values:')
# for key, value in adf_result_log2[4].items():
#     print('\t%s: %.3f' % (key, value))

# # plot acf and pacf of 2nd order log differenced data for Netherlands
# fig, ax = plt.subplots(2,1, figsize=(10,8))
# plot_acf(netherlands_log_diff2_clean, ax=ax[0], lags=30)
# plot_pacf(netherlands_log_diff2_clean, ax=ax[1], lags=30)
# plt.suptitle('ACF and PACF of 2nd Order Log Differenced rdgpe for Netherlands', fontsize=16)
# plt.savefig('storms_plots/netherlands_rdgpe_2ndorder_log_differenced_acf_pacf.png')
# plt.close()

# # plot acf and pacf for original data for Netherlands
# fig, ax = plt.subplots(2,1, figsize=(10,8))
# plot_acf(netherlands_data['rgdpe'], ax=ax[0], lags=30)
# plot_pacf(netherlands_data['rgdpe'], ax=ax[1], lags=30)
# plt.suptitle('ACF and PACF of rdgpe for Netherlands (raw data)', fontsize=16)
# plt.savefig('storms_plots/netherlands_rdgpe_acf_pacf.png')



# # plot acf and pacf of random walk data for comparison
# random_walk = np.cumsum(np.random.normal(size=len(netherlands_data)))
# fig, ax = plt.subplots(2,1, figsize=(10,8))
# plot_acf(random_walk, ax=ax[0], lags=30)
# plot_pacf(random_walk, ax=ax[1], lags=30)
# plt.suptitle('ACF and PACF of Random Walk Data', fontsize=16)
# plt.savefig('storms_plots/random_walk_acf_pacf.png')
# plt.close()







# ssa = SingularSpectrumAnalysis(df[df['country']=='Netherlands']['rgdpe'])
# ssa.decompose()
# print(ssa)
# u, s, vt = ssa.decomposition_results
# print('Eigentriple dimension:', [i.shape for i in ssa.decomposition_results])

# fig, ax = ssa.plot(n_components=37, marker='.')
# plt.suptitle('Singular Spectrum Analysis of rdgpe', fontsize=16)

# plt.savefig('storms_plots/ssa_rdgpe_values.png')
# plt.close()

# fig, ax = ssa.plot(rank_by='freq', n_components=37, marker='.', ls='none')
# plt.suptitle('Singular Spectrum Analysis of rdgpe Ranked by Frequency', fontsize=16)
# plt.savefig('storms_plots/ssa_rdgpe_freq.png')
# plt.close()

# # reconstruct trend 
# groups = {'Trend': list(range(0,3)), 'Seasonal': list(range(3,10)), 'Noise': list(range(10,37))}
# ssa.reconstruct(groups)

# # plot trend
# plt.figure(figsize=(10,6))
# plt.plot(netherlands_data['year'], netherlands_data['rgdpe'], label='Original rdgpe', color='lightgray')
# plt.plot(netherlands_data['year'], ssa['Trend'], label='SSA Trend', color='blue')
# plt.title('SSA Trend Component of rdgpe for Netherlands')
# plt.xlabel('Year')
# plt.gca().yaxis.get_major_formatter().set_scientific(False)
# plt.ylabel('rdgpe')
# plt.legend()
# plt.savefig('storms_plots/ssa_rdgpe_trend.png')
# plt.close()

# # see if statsmodels agrees
# # convert Netherlands data to time series with year as DateTimeIndex
# df_for_statsmodels = netherlands_data.copy()
# df_for_statsmodels['year'] = pd.to_datetime(df_for_statsmodels['year'], format='%Y')
# df_for_statsmodels.set_index('year', inplace=True)
# #
# decomposed =  seasonal_decompose(df_for_statsmodels['rgdpe'], model='additive', period=1)
# # plot decomposed
# decomposed.plot()
# plt.suptitle('Seasonal Decomposition of rdgpe for Netherlands', fontsize=16)

# # see if differenced data does anything
# df_for_statsmodels_diff = netherlands_data.copy()
# df_for_statsmodels_diff['year'] = pd.to_datetime(df_for_statsmodels_diff['year'], format='%Y')
# df_for_statsmodels_diff.set_index('year', inplace=True)
# #diff the data
# df_for_statsmodels_diff['rgdpe_diff'] = df_for_statsmodels_diff['rgdpe'].diff()

# decomposed_diff = seasonal_decompose(df_for_statsmodels_diff['rgdpe_diff'].dropna(), model='additive', period=1) 
# decomposed_diff.plot()
# plt.suptitle('Seasonal Decomposition of Differenced rdgpe for Netherlands', fontsize=16)
# plt.savefig('storms_plots/Netherlands_rdgpe_differenced_decomposition.png')
# plt.close()

# # compare employment to rdgpe for United States
# df_nl = df[df['country'] == 'United States'][['year', 'rgdpe', 'emp']]
# plt.figure(figsize=(10,6))
# plt.plot(df_nl['year'], df_nl['emp'] * 1e6, label='Employment (millions)', color='orange')
# plt.title('Employment graph for USA') 
# plt.xlabel('Year')
# plt.ylabel('Employment (millions)')
# plt.gca().yaxis.get_major_formatter().set_scientific(False)
# plt.axvspan(2007, 2009, color='red', alpha=0.3, label='2008-2009 Financial Crisis')
# # mark oil crisis 1973
# plt.axvspan(1973, 1975, color='orange', alpha=0.3, label='1973-1975 Oil Crisis')
# # mark dotcom bubble 2000-2001
# plt.axvspan(2000, 2001, color='green', alpha=0.3, label='2000-2001 Dotcom Bubble')
# plt.legend()
# # mark covid-19 recession 2020-onwards
# plt.axvspan(2020, 2023, color='purple', alpha=0.3, label='2020 Covid-19 Recession')
# plt.legend()
# plt.savefig('storms_plots/USA_employment_graph.png')
# plt.close()

# # does not prove much, employment only very slightly drops during recessions, but not by much.


# # try to do linear regression on rdgpe data for Netherlands
# from sklearn.linear_model import LinearRegression

# X = netherlands_data['year'].values.reshape(-1, 1)
# y = netherlands_data['rgdpe'].values

# # do log transformation to y to see if that helps
# y_log = np.log(y)
# model_linear = LinearRegression()
# linear_model = model_linear.fit(X, y)
# model_logarithmic = LinearRegression()
# logarithmic_model = model_logarithmic.fit(X, y_log)
# y_pred_linear = linear_model.predict(X)
# y_log_pred = logarithmic_model.predict(X)
# # exponentiate y_log_pred to get back to original scale
# y_log_pred_exp = np.exp(y_log_pred)

# # try to see what happens when you subtract both fits from original data
# plt.figure(figsize=(10,6))
# plt.plot(netherlands_data['year'], netherlands_data['rgdpe'], label='Original rdgpe', color='lightgray')
# plt.plot(netherlands_data['year'], y_pred_linear, label='Linear Regression Fit', color='red')
# plt.plot(netherlands_data['year'], y_log_pred_exp, label='Logarithmic Regression Fit', color='blue')
# plt.title('Linear Regression on rdgpe for Netherlands')
# plt.xlabel('Year')
# plt.gca().yaxis.get_major_formatter().set_scientific(False)
# plt.ylabel('rdgpe')
# plt.legend()
# plt.savefig('storms_plots/Netherlands_rdgpe_linear_regression.png')
# plt.close()

# # plot residuals of both models
# residuals_linear = y - y_pred_linear
# residuals_log = y - y_log_pred_exp
# plt.figure(figsize=(10,6))
# plt.scatter(netherlands_data['year'], residuals_linear, label='Linear Regression Residuals', color='red')
# plt.scatter(netherlands_data['year'], residuals_log, label='Logarithmic Regression Residuals', color='blue')
# plt.title('Residuals of Linear and Logarithmic Regression on rdgpe for Netherlands')
# plt.xlabel('Year')
# plt.gca().yaxis.get_major_formatter().set_scientific(False)
# plt.ylabel('Residuals')
# plt.legend()
# plt.savefig('storms_plots/Netherlands_rdgpe_regression_residuals.png')