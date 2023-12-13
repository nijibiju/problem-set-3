#!/usr/bin/env python
# coding: utf-8

# Question#1
# Step 1: Import the necessary libraries
# Step 2: Import the dataset 
# Step 3. Assign it to a variable called users

# In[1]:


import pandas as pd
users = pd.read_csv("https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user", sep = "|")
users


# Step 4. Discover what is the mean age per occupation

# In[2]:


users.groupby("occupation").age.mean()


# Step 5. Discover the Male ratio per occupation and sort it from the most to the least

# In[8]:


users['male'] = users.gender.apply(lambda x: True if x=='M' else False)
(users.groupby('occupation').male.sum() / users.groupby('occupation').gender.count()).sort_values(ascending = False).round(2)


# Step 6. For each occupation, calculate the minimum and maximum ages

# In[9]:


result_1 = users.groupby('occupation').agg({'age': ['min','max']})
result_1


# Step 7. For each combination of occupation and sex, calculate the mean age

# In[10]:


result_2 = users.groupby(['occupation','gender']).agg({'age': ['mean']})
result_2


# Step 8. For each occupation present the percentage of women and men

# In[11]:


df = (users.groupby('occupation')['gender'].value_counts(normalize=True).reset_index(name='perc'))
df


# Question#2
# Step 1. Import the necessary libraries Step 2. Import the dataset from this address Step 3. Assign it to a variable called users

# In[13]:


euro12 = pd.read_csv("https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/02_Filtering_%26_Sorting/Euro12/Euro_2012_stats_TEAM.csv")
euro12.head()


# Step 4. Select only the Goal column

# In[14]:


goals = euro12['Goals']
goals


# Step 5. How many team participated in the Euro2012?

# In[15]:


teams_count = euro12['Team'].count()
teams_count


# Step 6. What is the number of columns in the dataset?

# In[16]:


len(euro12.columns)


# Step 7. View only the columns Team, Yellow Cards and Red Cards and assign them to a dataframe called discipline

# In[17]:


discipline = euro12[['Team','Yellow Cards','Red Cards']]
discipline


# Step 8. Sort the teams by Red Cards, then to Yellow Cards

# In[18]:


discipline.sort_values(['Red Cards', 'Yellow Cards'], ascending = False)


# Step 9. Calculate the mean Yellow Cards given per Team

# In[19]:


mean_yellow_cards = discipline.groupby('Team').agg({'Yellow Cards': ['mean']})
mean_yellow_cards


# Step 10. Filter teams that scored more than 6 goals

# In[20]:


goals_per_team = euro12[['Team','Goals']]
greater_goals = goals_per_team[goals_per_team['Goals']  > 6]
greater_goals


# Step 11. Select the teams that start with G

# In[21]:


Teams_names = euro12['Team']
new_val = Teams_names[Teams_names.str.startswith("G")]
new_val


# Step 12. Select the first 7 columns

# In[22]:


euro12.iloc[:, list(range(7))]


# Step 13. Select all columns except the last 3

# In[23]:


euro12.iloc[: , :-3]


# Step 14. Present only the Shooting Accuracy from England, Italy and Russia

# In[24]:


shooting_accuracy = euro12[['Team','Shooting Accuracy']]
val = shooting_accuracy.loc[shooting_accuracy.Team.isin(['England','Italy','Russia'])]
val


# Question#3
# Step 1. Import the necessary libraries Step 2. Create 3 differents Series, each of length 100, as follows: • The first a random number from 1 to 4 • The second a random number from 1 to 3 • The third a random number from 10,000 to 30,000

# In[25]:


import pandas as pd
import numpy as np
s1 = pd.Series(np.random.randint(1, high=5, size=100, dtype='l'))
s2 = pd.Series(np.random.randint(1, high=4, size=100, dtype='l'))
s3 = pd.Series(np.random.randint(10000, high=30001, size=100, dtype='l'))

print(s1, s2, s3)


# Step 3. Create a DataFrame by joinning the Series by column

# In[26]:


housemkt = pd.concat([s1, s2, s3], axis=1)
housemkt.head()


# Step 4. Change the name of the columns to bedrs, bathrs, price_sqr_meter

# In[27]:


housemkt.rename(columns = {0: 'bedrs', 1: 'bathrs', 2: 'price_sqr_meter'}, inplace=True)
housemkt.head()


# Step 5. Create a one column DataFrame with the values of the 3 Series and assign it to 'bigcolumn'

# bigcolumn = pd.concat([s1, s2, s3], axis=0)
# bigcolumn = bigcolumn.to_frame()
# print(type(bigcolumn))
# 
# bigcolumn

# Step 6. Ops it seems it is going only until index 99. Is it true?

# In[29]:


len(bigcolumn)


# Step 7. Reindex the DataFrame so it goes from 0 to 299

# In[30]:


bigcolumn.reset_index(drop=True, inplace=True)
bigcolumn


# Question#4
# The data have been modified to contain some missing values, identified by NaN. Using pandas should make this exercise easier, in particular for the bonus question. You should be able to perform all of these operations without using a for loop or other looping construct.
# 
# Step 1. Import the necessary libraries Step 2. Import the dataset from the attached file wind.txt

# In[34]:


import pandas as pd

file_path = r'C:\Users\nijib\Downloads\wind.txt'  # Use a raw string (r) or escape the backslashes
wind_data = pd.read_csv(file_path, delimiter='\s+', skiprows=1)
print(wind_data.head())


# Step 3. Assign it to a variable called data and replace the first 3 columns by a proper datetime index.

# In[35]:


# Assign column names
column_names = ["Yr", "Mo", "Dy", "RPT", "VAL", "ROS", "KIL", "SHA", "BIR", "DUB", "CLA", "MUL", "CLO", "BEL", "MAL"]
# Assign it to a variable called 'data'
data = wind_data.copy()
# Add column names
data.columns = column_names
# Combine the first three columns into a single datetime index
date_columns = ["Yr", "Mo", "Dy"]
data["DateTime"] = pd.to_datetime(data[date_columns].astype(str).agg('-'.join, axis=1), format='%y-%m-%d')
data = data.set_index('DateTime')
data = data.drop(columns=date_columns)
print(data.head())


# Step 4. Year 2061? Do we really have data from this year? Create a function to fix it and apply it.

# In[36]:


data = wind_data.copy() 
# Add column names
data.columns = column_names
date_columns = ["Yr", "Mo", "Dy"]
data["DateTime"] = pd.to_datetime(data[date_columns].astype(str).agg('-'.join, axis=1), format='%y-%m-%d')

# Function to fix the year
def fix_year(year):
    if year >= 70:
        return year + 1900
    else:
        return year + 2000
data['Yr'] = data['Yr'].apply(fix_year)
data = data.set_index('DateTime')
data = data.drop(columns=date_columns)
print(data.head())


# Step 5. Set the right dates as the index. Pay attention at the data type, it should be datetime64.

# In[37]:


data = wind_data.copy() 
data.columns = column_names
date_columns = ["Yr", "Mo", "Dy"]
data["DateTime"] = pd.to_datetime(data[date_columns].astype(str).agg('-'.join, axis=1), format='%y-%m-%d')

# Function to fix the year
def fix_year(year):
    if year >= 70:
        return year + 1900
    else:
        return year + 2000
data['Yr'] = data['Yr'].apply(fix_year)
data = data.set_index('DateTime')
data = data.drop(columns=date_columns)
print(data.head())


# Step 6. Compute how many values are missing for each location over the entire record.They should be ignored in all calculations below

# In[38]:


data.index = pd.to_datetime(data.index)
missing_values = data.isnull().sum()
print("Number of missing values for each location:")
print(missing_values)


# Step 7. Compute how many non-missing values there are in total.

# In[39]:


data = wind_data.copy()
data.columns = column_names
date_columns = ["Yr", "Mo", "Dy"]
data["DateTime"] = pd.to_datetime(data[date_columns].astype(str).agg('-'.join, axis=1), format='%y-%m-%d')

# Function to fix the year
def fix_year(year):
    if year >= 70:
        return year + 1900
    else:
        return year + 2000

data['Yr'] = data['Yr'].apply(fix_year)

# Setting  the 'DateTime' column as index
data = data.set_index('DateTime')
data = data.drop(columns=date_columns)
data.index = pd.to_datetime(data.index)

# Counting the total number of non-missing values in the dataset
total_non_missing = data.count().sum()
print("Total number of non-missing values in the dataset:", total_non_missing)


# Step 8. Calculate the mean windspeeds of the windspeeds over all the locations and all the times

# In[40]:


data = wind_data.copy() 
data.columns = column_names

# Combine the first three columns into a single datetime index
date_columns = ["Yr", "Mo", "Dy"]
data["DateTime"] = pd.to_datetime(data[date_columns].astype(str).agg('-'.join, axis=1), format='%y-%m-%d')

# Function to fix the year
def fix_year(year):
    if year >= 70:
        return year + 1900
    else:
        return year + 2000

# Apply the fix_year function to the 'Yr' column
data['Yr'] = data['Yr'].apply(fix_year)
data = data.set_index('DateTime')
data = data.drop(columns=date_columns)
data.index = pd.to_datetime(data.index)
# Calculating the windspeeds over all locations and times
mean_windspeed = data.mean().mean()
print("Mean windspeed over all locations and times:", mean_windspeed)


# Step 9. Create a DataFrame called loc_stats and calculate the min, max and mean windspeeds and standard deviations of the windspeeds at each location over all the days

# In[44]:


data.describe(percentiles=[])


# Step 10. Create a DataFrame called day_stats and calculate the min, max and mean windspeed and standard deviations of the windspeeds across all the locations at each day.

# In[45]:


data = wind_data.copy()
data.columns = column_names
date_columns = ["Yr", "Mo", "Dy"]
data["DateTime"] = pd.to_datetime(data[date_columns].astype(str).agg('-'.join, axis=1), format='%y-%m-%d')
def fix_year(year):
    if year >= 70:
        return year + 1900
    else:
        return year + 2000
data['Yr'] = data['Yr'].apply(fix_year)
data = data.set_index('DateTime')
data = data.drop(columns=date_columns)
data.index = pd.to_datetime(data.index)

# Calculating min, max, mean, and standard deviations of windspeeds at each day
day_stats = pd.DataFrame({
    'Min': data.min(axis=1),
    'Max': data.max(axis=1),
    'Mean': data.mean(axis=1),
    'Std': data.std(axis=1)
})
print(day_stats)


# Step 11. Find the average windspeed in January for each location. Treat January 1961 and January 1962 both as January.

# In[46]:


data = wind_data.copy()
data.columns = column_names
date_columns = ["Yr", "Mo", "Dy"]
data["DateTime"] = pd.to_datetime(data[date_columns].astype(str).agg('-'.join, axis=1), format='%y-%m-%d')

# Function to fix the year
def fix_year(year):
    if year >= 70:
        return year + 1900
    else:
        return year + 2000
data['Yr'] = data['Yr'].apply(fix_year)
data = data.set_index('DateTime')
data = data.drop(columns=date_columns)
data.index = pd.to_datetime(data.index)
# Calculate average windspeed in January for each location
january_avg = data[data.index.month == 1].mean()
print(january_avg)


# Step 12. Downsample the record to a yearly frequency for each location.

# In[47]:


data = wind_data.copy()
data.columns = column_names
date_columns = ["Yr", "Mo", "Dy"]
data["DateTime"] = pd.to_datetime(data[date_columns].astype(str).agg('-'.join, axis=1), format='%y-%m-%d')
def fix_year(year):
    if year >= 70:
        return year + 1900
    else:
        return year + 2000
data['Yr'] = data['Yr'].apply(fix_year)
data = data.set_index('DateTime')
data = data.drop(columns=date_columns)
data.index = pd.to_datetime(data.index)
yearly_data = data.resample('Y').mean()
print(yearly_data.head())


# Step 13. Downsample the record to a monthly frequency for each location.

# In[48]:


data = wind_data.copy()
data.columns = column_names
date_columns = ["Yr", "Mo", "Dy"]
data["DateTime"] = pd.to_datetime(data[date_columns].astype(str).agg('-'.join, axis=1), format='%y-%m-%d')
def fix_year(year):
    if year >= 70:
        return year + 1900
    else:
        return year + 2000
data['Yr'] = data['Yr'].apply(fix_year)
data = data.set_index('DateTime')
data = data.drop(columns=date_columns)
data.index = pd.to_datetime(data.index)
monthly_data = data.resample('M').mean()
print(monthly_data.head())


# Step 14. Downsample the record to a weekly frequency for each location.

# In[49]:


data = wind_data.copy()
data.columns = column_names
date_columns = ["Yr", "Mo", "Dy"]
data["DateTime"] = pd.to_datetime(data[date_columns].astype(str).agg('-'.join, axis=1), format='%y-%m-%d')
def fix_year(year):
    if year >= 70:
        return year + 1900
    else:
        return year + 2000
data['Yr'] = data['Yr'].apply(fix_year)
data = data.set_index('DateTime')
data = data.drop(columns=date_columns)
data.index = pd.to_datetime(data.index)
weekly_data = data.resample('W').mean()
print(weekly_data.head())


# Step 15. Calculate the min, max and mean windspeeds and standard deviations of the windspeeds across all locations for each week (assume that the first week starts on January 2 1961) for the first 52 weeks.

# In[50]:


data = wind_data.copy()
data.columns = column_names
date_columns = ["Yr", "Mo", "Dy"]
data["DateTime"] = pd.to_datetime(data[date_columns].astype(str).agg('-'.join, axis=1), format='%y-%m-%d')
def fix_year(year):
    if year >= 70:
        return year + 1900
    else:
        return year + 2000
data['Yr'] = data['Yr'].apply(fix_year)
data = data.set_index('DateTime')
data = data.drop(columns=date_columns)
data.index = pd.to_datetime(data.index)
data = data.sort_index()
weekly_data = data.loc['1961-01-02':].resample('W-Mon').agg(['min', 'max', 'mean', 'std']).head(52)
print(weekly_data.head())


# Question#5
# Step 1. Import the necessary libraries Step 2. Import the dataset from this address. Step 3. Assign it to a variable called chipo

# In[51]:


chipo = pd.read_table("https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv")
chipo.head(10)


# Step 4. See the first 10 entries

# In[53]:


chipo.head(10)


# Step 5. What is the number of observations in the dataset?

# In[54]:


len(chipo.columns)


# Step 6. What is the number of columns in the dataset?

# In[56]:


chipo.shape[1]


# Step 7. Print the name of all the columns.

# In[58]:


chipo.columns.values


# Step 8. How is the dataset indexed?

# In[60]:


chipo.index


# Step 9. Which was the most-ordered item?

# In[61]:


chipo.groupby(['item_name']).quantity.sum().sort_values(ascending = False).index[0]


# In[62]:


chipo.groupby('item_name').sum().sort_values(['quantity'], ascending=False).head(1)


# Step 10. For the most-ordered item, how many items were ordered?

# In[63]:


chipo.groupby(['item_name']).quantity.sum().sort_values(ascending = False).values[0]


# In[64]:


chipo.groupby('item_name').sum().sort_values(['quantity'], ascending=False).head(1)


# Step 11. What was the most ordered item in the choice_description column?

# In[66]:


chipo.groupby(['choice_description']).quantity.sum().sort_values(ascending = False).index[0]


# In[67]:


chipo.groupby('choice_description').sum().sort_values(['quantity'], ascending=False).head(1)


# Step 12. How many items were orderd in total?

# In[68]:


chipo.quantity.sum()


# Step 13. • Turn the item price into a float • Check the item price type • Create a lambda function and change the type of item price • Check the item price type

# In[14]:


import pandas as pd

url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'

try:
    chipotle_data = pd.read_csv(url, delimiter='\t')
    chipotle_data['item_price'] = chipotle_data['item_price'].apply(lambda x: float(x[1:]) if isinstance(x, str) else x)
    print(f"Item price type before conversion: {chipotle_data['item_price'].dtype}")

    convert_to_float = lambda x: float(x[1:]) if isinstance(x, str) else x
    chipotle_data['item_price'] = chipotle_data['item_price'].apply(convert_to_float)

    print(f"Item price type after conversion: {chipotle_data['item_price'].dtype}")

except pd.errors.EmptyDataError:
    print("Error: The CSV file is empty.")
except pd.errors.ParserError:
    print("Error: There was an issue parsing the CSV file.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


# Step 14. How much was the revenue for the period in the dataset?

# In[15]:


chipotle_data = pd.read_csv(url, delimiter='\t')
chipotle_data['item_price'] = chipotle_data['item_price'].replace('[\$,]', '', regex=True).astype(float)
chipotle_data['revenue'] = chipotle_data['quantity'] * chipotle_data['item_price']
total_revenue = chipotle_data['revenue'].sum()
print(f"Total revenue for the period: ${total_revenue:.2f}")


# Step 15. How many orders were made in the period?

# In[16]:


chipotle_data = pd.read_csv(url, delimiter='\t')
total_orders = chipotle_data['order_id'].nunique()
print(f"Total number of orders in the period: {total_orders}")


# Step 16. What is the average revenue amount per order?

# In[17]:


chipotle_data = pd.read_csv(url, delimiter='\t')
chipotle_data['item_price'] = chipotle_data['item_price'].apply(lambda x: float(x[1:]))
chipotle_data['revenue'] = chipotle_data['quantity'] * chipotle_data['item_price']
average_revenue_per_order = chipotle_data.groupby('order_id')['revenue'].sum().mean()
print(f"Average revenue amount per order: ${average_revenue_per_order:.2f}")


# Step 17. How many different items are sold?

# In[18]:


chipotle_data = pd.read_csv(url, delimiter='\t')
num_different_items = chipotle_data['item_name'].nunique()
print(f"Number of different items sold: {num_different_items}")


# Question#6
# Create a line plot showing the number of marriages and divorces per capita in the U.S. between 1867 and 2014. Label both lines and show the legend.Don't forget to label your axes!

# In[44]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
us_mar_div = pd.read_csv("C:\\Users\\nijib\\Downloads\\us-marriages-divorces-1867-2014.csv")

# Sort the DataFrame by 'Year' in-place
us_mar_div.sort_values('Year', inplace=True)

# Extract data for plotting
year = us_mar_div['Year'].tolist()
mar = us_mar_div['Marriages_per_1000'].tolist()
div = us_mar_div['Divorces_per_1000'].tolist()

# Create a DataFrame for plotting
df = pd.DataFrame({
   'Marriage': mar,
   'Divorce': div
}, index=year)

# Plot the data
lines = df.plot.line(xlabel='Year', ylabel='Count')
plt.show()



# Question#7
# Create a vertical bar chart comparing the number of marriages and divorces per capita in the U.S. between 1900, 1950, and 2000. Don't forget to label your axes!

# In[45]:


df_selected = us_mar_div[['Year','Marriages_per_1000','Divorces_per_1000']]
val = df_selected.loc[df_selected.Year.isin([1900,1950,2000])]
val
Year = val['Year'].tolist()
Year_str = []
for x in Year:
    Year_str.append(str(x))

Marriages = val['Marriages_per_1000'].tolist()
Divorces = val['Divorces_per_1000'].tolist()

plt.bar(Year_str,Marriages,0.3,label="Marriages")
plt.bar(Year_str,Divorces,0.3,label="Divorces")
       
plt.xlabel("year")
plt.ylabel("count")
plt.legend()
plt.show()


# Question#8
# Create a horizontal bar chart that compares the deadliest actors in Hollywood. Sort the actors by their kill count and label each bar with the corresponding actor's name. Don't forget to label your axes!

# In[48]:


kill_counts = pd.read_csv("C:\\Users\\nijib\\Downloads\\actor_kill_counts.csv")
kill_counts_sorted = kill_counts.sort_values('Count',ascending = False)
kill_counts_sorted
plt.barh(kill_counts_sorted['Actor'],kill_counts_sorted['Count'])
plt.xlabel('count')
plt.ylabel('actor')
plt.show()


# Question#9
# Create a pie chart showing the fraction of all Roman Emperors that were assassinated.Make sure that the pie chart is an even circle, labels the categories, and shows the percentage breakdown of the categories.

# In[49]:


roman_regime = pd.read_csv("C:\\Users\\nijib\\Downloads\\roman-emperor-reigns.csv")
roman_regime.head()
cause_of_death = roman_regime['Cause_of_Death'].value_counts()
cause_of_death
cause_of_death.plot(kind='pie',autopct='%1.0f%%')


# Question#10
# Create a scatter plot showing the relationship between the total revenue earned by arcades and the number of Computer Science PhDs awarded in the U.S. between 2000 and 2009. Don't forget to label your axes!Color each dot according to its year
# 
# import pandas as 

# In[50]:


import pandas as pd
import matplotlib.pyplot as plt
arcade_revenue = pd.read_csv("C:\\Users\\nijib\\Downloads\\arcade-revenue-vs-cs-doctorates.csv")
fig, ax = plt.subplots()
scatter = ax.scatter(
    x='Computer Science Doctorates Awarded (US)',
    y='Total Arcade Revenue (billions)',
    s=60,
    c='Year',
    cmap='viridis',
    data=arcade_revenue
)
plt.xlabel('Computer Science Doctorates Awarded (US)')
plt.ylabel('Total Arcade Revenue (billions)')
plt.title('Relationship between Arcade Revenue and Computer Science PhDs (2000-2009)')
cbar = plt.colorbar(scatter)
cbar.set_label('Year')
plt.show()

