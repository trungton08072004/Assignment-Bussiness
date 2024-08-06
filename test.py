import pandas as pd
import matplotlib.pyplot as plt
__file__path = "Access.csv"
# Load the data from the CSV file
data = pd.read_csv(__file__path)

# Display the original data
print("Original Data:")
print(data)

# Display the cleaned data
print("\nCleaned Data:")
print(data)

cleaned_data = data.dropna()

print("\nData after removing rows with blank data:")
print(cleaned_data)

print("Columns in the DataFrame:")
print(cleaned_data.columns)

# Change Data Type
# For demonstration, let's change 'Quantity Ordered' to integer and 'Price Each' to float
cleaned_data['Quantity Ordered'] = cleaned_data['Quantity Ordered'].astype(int)
cleaned_data['Price Each'] = cleaned_data['Price Each'].astype(float)

print("\nData after changing data types:")
print(cleaned_data.dtypes)

# Replace Data
# For demonstration, let's replace a specific value in 'City' column
cleaned_data['City'] = cleaned_data['City'].replace('NYC', 'New York City')

print("\nData after replacing values in 'City' column:")
print(cleaned_data)

# Merge Data
# For demonstration, let's create another DataFrame and merge it with cleaned_data
additional_data = pd.DataFrame({
    'ProductID': [101, 102, 103, 104, 105],
    'ProductName': ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
})

merged_data = pd.merge(cleaned_data, additional_data, on='ProductID', how='left')

print("\nData after merging with additional data:")
print(merged_data)

# Create a New Column
# For demonstration, let's create a new column 'TotalPrice' as Quantity Ordered * Price Each
merged_data['TotalPrice'] = merged_data['Quantity Ordered'] * merged_data['Price Each']

print("\nData after creating a new column 'TotalPrice':")
print(merged_data)
# Split Columns
# For demonstration, let's split the 'Address' column into 'Street' and 'Number' assuming 'Address' is in format '123 Main Street'
address_split = merged_data['Address'].str.extract(r'(?P<Number>\d+)\s(?P<Street>.+)')
merged_data = merged_data.join(address_split)

print("Data after splitting 'Address' into 'Number' and 'Street':")
print(merged_data)
# 1. What was the best month for sales? How much was earned that month?
monthly_sales = cleaned_data.groupby('Month')['TotalPrice'].sum()
best_month = monthly_sales.idxmax()
best_month_sales = monthly_sales.max()

plt.figure(figsize=(10, 6))
monthly_sales.plot(kind='line', marker='o')
plt.title(f'Best Month for Sales: {best_month} with Sales of ${best_month_sales:.2f}')
plt.xlabel('Month')
plt.ylabel('Total Sales in USD')
plt.grid(True)
plt.show()

# 2. What city has the best sales?
city_sales = cleaned_data.groupby('City')['TotalPrice'].sum()
best_city = city_sales.idxmax()
best_city_sales = city_sales.max()

plt.figure(figsize=(10, 6))
city_sales.plot(kind='area', alpha=0.5)
plt.title(f'City with Best Sales: {best_city} with Sales of ${best_city_sales:.2f}')
plt.xlabel('City')
plt.ylabel('Total Sales in USD')
plt.show()

# 3. Top N StoreID with the highest/lowest sales
N = 5  # You can change this value for top N stores
store_sales = cleaned_data.groupby('StoreID')['TotalPrice'].sum()
top_stores = store_sales.nlargest(N)
bottom_stores = store_sales.nsmallest(N)

# Combine top and bottom stores into a single DataFrame for comparison
comparison_data = pd.concat([top_stores, bottom_stores], axis=1)
comparison_data.columns = ['Top Sales', 'Bottom Sales']
comparison_data = comparison_data.fillna(0)

# Plot the comparison chart
plt.figure(figsize=(14, 7))
comparison_data.plot(kind='bar', figsize=(14, 7))
plt.title(f'Top {N} and Bottom {N} StoreIDs by Sales')
plt.xlabel('StoreID')
plt.ylabel('Total Sales in USD')
plt.xticks(rotation=45)
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# 4. Top N City by Sales? How much was earned that month?
N = 5  # You can change this value for top N cities
city_sales = cleaned_data.groupby('City')['TotalPrice'].sum()
top_cities = city_sales.nlargest(N)

# Sort the values in descending order and calculate the cumulative percentage
top_cities_sorted = top_cities.sort_values(ascending=False)
cumulative_percentage = top_cities_sorted.cumsum() / top_cities_sorted.sum() * 100

# Plot the horizontal Pareto chart
fig, ax1 = plt.subplots(figsize=(14, 7))

# Horizontal bar plot (top N cities by sales)
ax1.barh(top_cities_sorted.index, top_cities_sorted.values, color='c0')
ax1.set_xlabel('Total Sales in USD', color='c0')
ax1.set_ylabel('City')
ax1.tick_params(axis='x', labelcolor='c0')

# Line plot (cumulative percentage)
ax2 = ax1.twiny()
ax2.plot(cumulative_percentage, top_cities_sorted.index, color='c1', marker='o', linestyle='-', linewidth=2)
ax2.set_xlabel('Cumulative Percentage', color='c1')
ax2.tick_params(axis='x', labelcolor='c1')
ax2.axvline(x=80, color='r', linestyle='--')

# Title and grid
plt.title(f'Top {N} Cities by Sales')
fig.tight_layout()
plt.grid(True)
plt.show()

# Best Month for Sales
monthly_sales = cleaned_data.groupby('Month')['TotalPrice'].sum()

plt.figure(figsize=(10, 6))
monthly_sales.plot(kind='bar')
plt.title('Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Total Sales ($)')
plt.show()

best_month = monthly_sales.idxmax()
best_month_sales = monthly_sales.max()
print(f'The best month for sales was {best_month} with total earnings of ${best_month_sales:.2f}')
# Best City for Sales
city_sales = city_data.groupby('City')['TotalPrice'].sum()
plt.figure(figsize=(10, 6))
city_sales.plot(kind='bar')
plt.title('Sales by City')
plt.xlabel('City')
plt.ylabel('Total Sales ($)')
plt.show()

best_city = city_sales.idxmax()
print(f"The city with the best sales is {best_city} with total earnings of {city_sales.max():.2f}")

import pandas as pd
import matplotlib.pyplot as plt

# Top 5 Stores with the Highest Sales
store_sales = pd.read_csv('store_data.csv')
top_stores = store_sales.groupby('StoreID')['TotalPrice'].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
plt.bar(top_stores.index[:5], top_stores.head(), color='green')
plt.title('Top 5 Stores with the Highest Sales')
plt.xlabel('StoreID')
plt.ylabel('Total Sales ($)')

# Top 5 Stores with the Lowest Sales
plt.figure(figsize=(10, 6))
plt.bar(top_stores.index[-5:], top_stores.tail(), color='red')
plt.title('Top 5 Stores with the Lowest Sales')
plt.xlabel('StoreID')
plt.ylabel('Total Sales ($)')

plt.show()

# For illustrative purposes, using a simple time series plot. In practice, use advanced forecasting techniques.
sales_trend = cleaned_data.groupby('Date')['TotalPrice'].sum()

plt.figure(figsize=(10, 6))
plt.title('Sales Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales ($)')
plt.show()

from sklearn.cluster import KMeans

# Using K-means clustering for customer segmentation
customer_data = cleaned_data.groupby('City')['TotalPrice'].sum().reset_index()
kmeans = KMeans(n_clusters=3, n_init=10)
customer_data['Cluster'] = kmeans.fit_predict(customer_data[['TotalPrice']])

plt.figure(figsize=(10, 6))
plt.scatter(customer_data['City'], customer_data['TotalPrice'], c=customer_data['Cluster'], cmap='viridis')
plt.title('Customer Segmentation by City and Total Sales')
plt.xlabel('City')
plt.ylabel('Total Sales ($)')
plt.show()

