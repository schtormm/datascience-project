# get year and country columns
import pandas as pd


df  = pd.read_csv('cleaned_V11.csv')
years = df['year'].unique()
country_codes_dict = df[['country', 'countrycode']].drop_duplicates().set_index('country')['countrycode'].to_dict()
country_codes = {country: code for country, code in country_codes_dict.items()}
# add percentage change column for rgdpe per country
df['rgdpe_pct_change'] = df.groupby('country')['rgdpe'].pct_change() * 100
df["rgdpe_acceleration"] = df['rgdpe_pct_change'].pct_change() * 100

# ask for user input of year
input_year = int(input("Enter a year between 1950 and 2031: "))
# make sure year is valid
if input_year < 1950 or input_year > 2031:
    print("Year out of range. Please enter a year between 1950 and 2031.")
    exit()
# filter dataframe for that year
filtered_df = df[df['year'] == input_year][['country', 'rgdpe', 'rgdpe_pct_change', 'rgdpe_acceleration']]
# ask for country input
input_country = input("Enter a country name (e.g., Netherlands) or country code (e.g., NLD): ")
# search for country in dictionary (first by name (also partial), then by code)
# if multiple countries match the partial name, retry
country_name = None
# first try by name (case insensitive, partial match)
if len(input_country) < 3 or len(input_country) > 3:
    matching_countries = [country for country in country_codes.keys() if input_country.lower() in country.lower()]
    if len(matching_countries) == 1:
        country_name = matching_countries[0]
    elif len(matching_countries) > 1:
        print("Matching countries:", matching_countries)
        print("Multiple countries match that name. Please be more specific.")
        exit()
elif  len(input_country) == 3:
    # try by code (case insensitive)
    matching_countries_by_code = [country for country, code in country_codes.items() if input_country.lower() == code.lower()]
    print(matching_countries_by_code)
    if len(matching_countries_by_code) == 1:
        country_name = matching_countries_by_code[0]
if country_name is not None:
    # get rgdpe for that country and year
    country_data = filtered_df[filtered_df['country'] == country_name]
    if not country_data.empty:
        # get rgdpe percentage change for that year and country (select year, country, rgdpe_pct_change)
        rgdpe_value = country_data['rgdpe'].values[0]
        rgdpe_pct_change_value = country_data['rgdpe_pct_change'].values[0]
        rgdpe_acceleration_value = country_data['rgdpe_acceleration'].values[0]
        print(f"In {input_year}, the rgdpe for {country_name} was {rgdpe_value:.2f}, with a percentage change of {rgdpe_pct_change_value:.2f}%. and a acceleration of {rgdpe_acceleration_value:.2f}.")
    else:
        print(f"No data found for {country_name} in {input_year}.")
else:
    print("Country not found. Please check your input.")
    exit()