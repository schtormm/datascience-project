import pandas as pd 
import numpy as np
import statsmodels as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm_api
from statsmodels.tsa.arima.model import ARIMA
from ssalib import SingularSpectrumAnalysis
import os

df = pd.read_csv('cleaned_V11.csv')
# create folder "storms_plots" if it doesn't exist
if not os.path.exists('storms_plots'):
    os.makedirs('storms_plots')
else:
    print("Directory 'storms_plots' already exists.")


#try to remove the trend from country data using differencing
df['rgdpe_diff'] = df.groupby('country')['rgdpe'].diff()
# plot differenced data for Netherlands
netherlands_diff = df[df['country'] == 'Netherlands'][['year', 'rgdpe_diff']]
plt.figure(figsize=(10,6))
plt.plot(netherlands_diff['year'], netherlands_diff['rgdpe_diff'], label='Netherlands Differenced')
plt.title('Differenced rdgpe for Netherlands')
plt.xlabel('Year')
plt.ylabel('Differenced rdgpe')
# mark 2007-2009 financial crisis
plt.axvspan(2007, 2009, color='red', alpha=0.3, label='2008-2009 Financial Crisis')
# mark oil crisis 1973
plt.axvspan(1973, 1974, color='orange', alpha=0.3, label='1973 Oil Crisis')
# mark dotcom bubble 2000-2001
plt.axvspan(2000, 2001, color='green', alpha=0.3, label='2000-2001 Dotcom Bubble')
plt.legend()
# mark covid-19 recession 2020-onwards
plt.axvspan(2020, 2023, color='purple', alpha=0.3, label='2020 Covid-19 Recession')
plt.savefig('storms_plots/netherlands_rdgpe_differenced.png')


# second order differencing
df['rgdpe_diff2'] = df.groupby('country')['rgdpe_diff'].diff()
# plot second order differenced data for Netherlands
netherlands_diff2 = df[df['country'] == 'Netherlands'][['year', 'rgdpe_diff2']]
plt.figure(figsize=(10,6))
plt.plot(netherlands_diff2['year'], netherlands_diff2['rgdpe_diff2'], label='Netherlands 2nd Order Differenced')
plt.title('2nd Order Differenced rdgpe for Netherlands')
plt.xlabel('Year')
plt.ylabel('2nd Order Differenced rdgpe')
plt.xlabel('Year')
plt.ylabel('Differenced rdgpe')
# mark 2007-2009 financial crisis
plt.axvspan(2007, 2009, color='red', alpha=0.3, label='2008-2009 Financial Crisis')
# mark oil crisis 1973
plt.axvspan(1973, 1974, color='orange', alpha=0.3, label='1973 Oil Crisis')
# mark dotcom bubble 2000-2001
plt.axvspan(2000, 2001, color='green', alpha=0.3, label='2000-2001 Dotcom Bubble')
plt.legend()
# mark covid-19 recession 2020-onwards
plt.axvspan(2020, 2023, color='purple', alpha=0.3, label='2020 Covid-19 Recession')
plt.savefig('storms_plots/NL_rdgpe_doubledifferenced.png')
plt.close()


# try to fit ARIMA on differenced data for Netherlands
netherlands_data = df[df['country'] == 'Netherlands'][['year', 'rgdpe']]
netherlands_diff_clean = netherlands_diff.dropna()
model = ARIMA(netherlands_diff_clean['rgdpe_diff'], order=(2,2,2))
model_fit = model.fit()
print(model_fit.summary())
# plot residuals (in scatterplot form)
residuals = model_fit.resid
plt.figure(figsize=(10,6))
plt.scatter(netherlands_diff_clean['year'], residuals)
plt.title('Residuals of ARIMA model on Differenced rdgpe for Netherlands')
plt.xlabel('Year')
plt.ylabel('Residuals')
plt.savefig('storms_plots/netherlands_rdgpe_differenced_arima_residuals.png')


# qq plot of residuals
sm_api.qqplot(residuals, line='s')
plt.title('QQ Plot of Residuals of ARIMA model on Differenced rdgpe for Netherlands')
plt.savefig('storms_plots/netherlands_rdgpe_differenced_arima_qqplot.png')


# try so see if differencing works for USA as well
usa_diff = df[df['country'] == 'United States'][['year', 'rgdpe_diff']]
plt.figure(figsize=(10,6))
plt.plot(usa_diff['year'], usa_diff['rgdpe_diff'], label='USA Differenced', color='green')
plt.title('Differenced rdgpe for USA')
plt.xlabel('Year')
plt.ylabel('Differenced rdgpe')
# mark 2007-2009 financial crisis
plt.axvspan(2007, 2009, color='red', alpha=0.3, label='2008-2009 Financial Crisis')
# mark oil crisis 1973
plt.axvspan(1973, 1974, color='orange', alpha=0.3, label='1973 Oil Crisis')
# mark dotcom bubble 2000-2001
plt.axvspan(2000, 2001, color='green', alpha=0.3, label='2000-2001 Dotcom Bubble')
plt.legend()
# mark covid-19 recession 2020-onwards
plt.axvspan(2020, 2023, color='purple', alpha=0.3, label='2020 Covid-19 Recession')
plt.savefig('storms_plots/USA_rdgpe_differenced.png')

# try second order differencing for Netherlands


# try log differencing
df['rgdpe_log'] = np.log(df['rgdpe'])
df['rgdpe_log_diff'] = df.groupby('country')['rgdpe_log'].diff()
# plot log differenced data for Netherlands
netherlands_log_diff = df[df['country'] == 'Netherlands'][['year', 'rgdpe_log_diff']]
plt.figure(figsize=(10,6))
plt.plot(netherlands_log_diff['year'], netherlands_log_diff['rgdpe_log_diff'], label='Netherlands Log Differenced')
plt.title('Log Differenced rdgpe for Netherlands')
plt.xlabel('Year')
plt.ylabel('Log Differenced rdgpe')
plt.axvspan(2007, 2009, color='red', alpha=0.3, label='2008-2009 Financial Crisis')
# mark oil crisis 1973
plt.axvspan(1973, 1974, color='orange', alpha=0.3, label='1973 Oil Crisis')
# mark dotcom bubble 2000-2001
plt.axvspan(2000, 2001, color='green', alpha=0.3, label='2000-2001 Dotcom Bubble')
plt.legend()
# mark covid-19 recession 2020-onwards
plt.axvspan(2020, 2023, color='purple', alpha=0.3, label='2020 Covid-19 Recession')
plt.savefig('storms_plots/Netherlands_rdgpe_log_differenced.png')




ssa = SingularSpectrumAnalysis(df[df['country']=='Netherlands']['rgdpe'])
ssa.decompose()
print(ssa)
u, s, vt = ssa.decomposition_results
print('Eigentriple dimension:', [i.shape for i in ssa.decomposition_results])

fig, ax = ssa.plot(n_components=37, marker='.')
plt.suptitle('Singular Spectrum Analysis of rdgpe', fontsize=16)

plt.savefig('storms_plots/ssa_rdgpe_values.png')
plt.close()

fig, ax = ssa.plot(rank_by='freq', n_components=37, marker='.', ls='none')
plt.suptitle('Singular Spectrum Analysis of rdgpe Ranked by Frequency', fontsize=16)
plt.savefig('storms_plots/ssa_rdgpe_freq.png')
plt.close()

groups = {
    'Trend': [1, 2]
}
ssa.reconstruct(groups)

# plot trend
plt.figure(figsize=(10,6))
plt.plot(netherlands_data['year'], netherlands_data['rgdpe'], label='Original rdgpe', color='lightgray')
plt.plot(netherlands_data['year'], ssa['Trend'], label='SSA Trend', color='blue')
plt.title('SSA Trend Component of rdgpe for Netherlands')
plt.xlabel('Year')
plt.gca().yaxis.get_major_formatter().set_scientific(False)
plt.ylabel('rdgpe')
plt.legend()
plt.savefig('storms_plots/ssa_rdgpe_trend.png')
plt.close()

# see if statsmodels agrees
# convert Netherlands data to time series with year as DateTimeIndex
df_for_statsmodels = netherlands_data.copy()
df_for_statsmodels['year'] = pd.to_datetime(df_for_statsmodels['year'], format='%Y')
df_for_statsmodels.set_index('year', inplace=True)
#
decomposed =  seasonal_decompose(df_for_statsmodels['rgdpe'], model='additive', period=1)
# plot decomposed
decomposed.plot()
plt.suptitle('Seasonal Decomposition of rdgpe for Netherlands', fontsize=16)

# see if differenced data does anything
df_for_statsmodels_diff = netherlands_data.copy()
df_for_statsmodels_diff['year'] = pd.to_datetime(df_for_statsmodels_diff['year'], format='%Y')
df_for_statsmodels_diff.set_index('year', inplace=True)
#diff the data
df_for_statsmodels_diff['rgdpe_diff'] = df_for_statsmodels_diff['rgdpe'].diff()

decomposed_diff = seasonal_decompose(df_for_statsmodels_diff['rgdpe_diff'].dropna(), model='additive', period=1) 
decomposed_diff.plot()
plt.suptitle('Seasonal Decomposition of Differenced rdgpe for Netherlands', fontsize=16)
plt.savefig('storms_plots/Netherlands_rdgpe_differenced_decomposition.png')
plt.close()

# compare employment to rdgpe for United States
df_nl = df[df['country'] == 'United States'][['year', 'rgdpe', 'emp']]
plt.figure(figsize=(10,6))
plt.plot(df_nl['year'], df_nl['emp'] * 1e6, label='Employment (millions)', color='orange')
plt.title('Employment graph for USA') 
plt.xlabel('Year')
plt.ylabel('Employment (millions)')
plt.gca().yaxis.get_major_formatter().set_scientific(False)
plt.axvspan(2007, 2009, color='red', alpha=0.3, label='2008-2009 Financial Crisis')
# mark oil crisis 1973
plt.axvspan(1973, 1975, color='orange', alpha=0.3, label='1973-1975 Oil Crisis')
# mark dotcom bubble 2000-2001
plt.axvspan(2000, 2001, color='green', alpha=0.3, label='2000-2001 Dotcom Bubble')
plt.legend()
# mark covid-19 recession 2020-onwards
plt.axvspan(2020, 2023, color='purple', alpha=0.3, label='2020 Covid-19 Recession')
plt.legend()
plt.savefig('storms_plots/USA_employment_graph.png')
plt.close()

# does not prove much, employment only very slightly drops during recessions, but not by much.