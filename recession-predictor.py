# get year and country columns
import pandas as pd


df  = pd.read_csv('cleaned_V11.csv')
years = df['year'].unique()
country_codes_dict = df[['country', 'countrycode']].drop_duplicates().set_index('country')['countrycode'].to_dict()
country_codes = {country: code for country, code in country_codes_dict.items()}
# add percentage change column for rgdpe per country
df['rgdpe_pct_change'] = df.groupby('country')['rgdpe'].pct_change() * 100
df["rgdpe_acceleration"] = df.groupby('country')['rgdpe_pct_change'].pct_change(fill_method=None) * 100
# calculate gdp per capita (rgdpe / population)
df['rdgpe_per_capita'] = df['rgdpe'] / df['pop'] * 1e6 / 1000000


# ask for user input of year
input_year = int(input("Enter a year between 1950 and 2031: "))
# make sure year is valid
if input_year < 1950 or input_year > 2031:
    print("Year out of range. Please enter a year between 1950 and 2031.")
    exit()
# filter dataframe for that year
filtered_df = df[df['year'] == input_year][['country', 'rgdpe', 'rgdpe_pct_change', 'rgdpe_acceleration', 'rdgpe_per_capita']]
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
    if len(matching_countries_by_code) == 1:
        country_name = matching_countries_by_code[0]
if country_name is not None:
    print(f"Found country: {country_name}")
    # get rgdpe for that country and year
    country_data = filtered_df[filtered_df['country'] == country_name]
    if not country_data.empty:
        # get rgdpe percentage change for that year and country (select year, country, rgdpe_pct_change)
        rgdpe_value = country_data['rgdpe'].values[0]
        rgdpe_pct_change_value = country_data['rgdpe_pct_change'].values[0]
        rgdpe_acceleration_value = country_data['rgdpe_acceleration'].values[0]
        rdgpe_per_capita_value = country_data["rdgpe_per_capita"].values[0]
        # if rgdpe_pct_change_value is < 0, check acceleration of previous years to see if there was a trend change
        print(f"Data for {country_name} in {input_year}:")
        print(f"  Expenditure-side real GDP (rgdpe): {rgdpe_value:.2f} million USD")
        print(f"  Per capita expenditure-side real GDP: {rdgpe_per_capita_value} USD")
        print(f"  rdgpe percentage change: {rgdpe_pct_change_value:.2f}%")
        print(f"  rdgpe acceleration: {rgdpe_acceleration_value:.2f}%")
        
        # check for recession indication: two years of negative acceleration (including current year), and negative pct change this year, or accelation < -100
        # get previous year data
        if rgdpe_pct_change_value < 0 or rgdpe_acceleration_value < 0:
            previous_year = input_year - 1
            previous_year_data = df[(df['year'] == previous_year) & (df['country'] == country_name)]
            if not previous_year_data.empty:
                previous_year_acceleration = previous_year_data['rgdpe_acceleration'].values[0]
                if (previous_year_acceleration < 0 and rgdpe_acceleration_value < 0) or (rgdpe_acceleration_value < -100):
                    print(f"  Previous year ({previous_year}) rdgpe acceleration: {previous_year_acceleration:.2f}")
                    print(f"  Recession indication: YES (either two years of negative GDP acceleration / negative GDP growth in the current year) or GDP acceleration < -100%)")

                else:
                    print(f"  Recession indication: No")
            else:
                print(f"  No data for previous year ({previous_year}) to determine recession indication.")
        else:
            print(f"  Recession indication: No")           

            
    else:
        print(f"No data found for {country_name} in {input_year}.")
else:
    print("Country not found. Please check your input.")
    exit()