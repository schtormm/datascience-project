import pandas as pd 
import numpy as np

df = pd.read_csv('cleaned_V11.csv')

#Earliest year is 1950
years = np.arange(1950, 2023)
country = "Fakelandia"
# get mean and std dev of rdgpe for all countries
mean_rdgpe = df['rgdpe'].mean()
std_rdgpe = df['rgdpe'].std()

print("Mean rdgpe:", mean_rdgpe)
print("Std dev rdgpe:", std_rdgpe)

# generate synthetic rdgpe data for Fakedatalia
np.random.seed(42)
# make sure it doesn't go below 0
# create the data seperately for each year to avoid negative values and wild swings
# make it generally increasing over time with some noise and around the mean

synthetic_rdgpe = []
current_value = mean_rdgpe * 0.5  # start below mean
for year in years:
    growth = np.random.normal(loc=mean_rdgpe * 0.03, scale=std_rdgpe * 0.01)  # average growth of 3% of mean rdgpe
    noise = np.random.normal(loc=0, scale=std_rdgpe * 0.05)  # some noise
    new_value = max(0, current_value + growth + noise)  # ensure non-negative
    synthetic_rdgpe.append(new_value)
    current_value = new_value
synthetic_data = pd.DataFrame({
    'year': years,
    'country': "Fakedatalia",
    'countrycode': 'FDA',
    'rgdpe': synthetic_rdgpe
})



# plot the synthetic data, scale it without "1e6" notation
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.plot(synthetic_data['year'], synthetic_data['rgdpe'])
plt.gca().yaxis.get_major_formatter().set_scientific(False)
plt.title('Synthetic rdgpe data for Fakedatalia')
plt.xlabel('Year')
plt.ylabel('rdgpe')
plt.savefig('fake_data_plots/synthetic_rdgpe.png')
plt.close

# compare to real country data, e.g., Netherlands
netherlands_data = df[df['country'] == 'Netherlands'][['year', 'rgdpe']]
plt.figure(figsize=(10,6))
plt.plot(netherlands_data['year'], netherlands_data['rgdpe'], label='Netherlands')
plt.plot(synthetic_data['year'], synthetic_data['rgdpe'], label='Fakedatalia', linestyle='--')
plt.gca().yaxis.get_major_formatter().set_scientific(False)
plt.title('Comparison of rdgpe: Netherlands vs Fakedatalia')
plt.xlabel('Year')
plt.ylabel('rdgpe')
plt.legend()
# output this plot to file in "fake_data_plots" folder
plt.savefig('fake_data_plots/rdgpe_comparison.png')
plt.close()

# make different synthetic data which is just mean of all countries
mean_data = df.groupby('year')['rgdpe'].mean().reset_index()
mean_data['country'] = 'Meanland'
mean_data['countrycode'] = 'MEA'
# plot mean data
plt.figure(figsize=(10,6))
plt.plot(mean_data['year'], mean_data['rgdpe'], color='orange')
plt.gca().yaxis.get_major_formatter().set_scientific(False)
plt.title('Mean rdgpe across all countries (Meanland)')
plt.xlabel('Year')
plt.ylabel('rdgpe')
plt.savefig('fake_data_plots/mean_rdgpe.png')
plt.close()

# compare to real country data, e.g., Netherlands
plt.figure(figsize=(10,6))
plt.plot(netherlands_data['year'], netherlands_data['rgdpe'], label='Netherlands')
plt.plot(mean_data['year'], mean_data['rgdpe'], label='Meanland', linestyle='--', color='orange')
plt.gca().yaxis.get_major_formatter().set_scientific(False)
plt.title('Comparison of rdgpe: Netherlands vs Meanland')
plt.xlabel('Year')
plt.ylabel('rdgpe')
plt.legend()
# output this plot to file in "fake_data_plots" folder
plt.savefig('fake_data_plots/mean_rdgpe_comparison.png')
plt.close()
