import pandas as pd

# Load data from CSV with the correct delimiter
data = pd.read_csv('data_penjualan.csv', delimiter=';')

# Display first few rows of the data to check if it is read correctly
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Drop rows with missing values
data.dropna(inplace=True)

# Remove duplicate rows
data.drop_duplicates(inplace=True)

# Convert 'Tanggal' to datetime format
data['Tanggal'] = pd.to_datetime(data['Tanggal'])

# Create new columns for 'Year' and 'Month'
data['Year'] = data['Tanggal'].dt.year
data['Month'] = data['Tanggal'].dt.month

# Exploratory Data Analysis (EDA)
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of sales by gender
sns.countplot(data=data, x='Jenis Kelamin')
plt.show()

# Total sales by product type
sales_by_product = data.groupby('Jenis Barang')['Jumlah'].sum().reset_index()
sns.barplot(data=sales_by_product, x='Jenis Barang', y='Jumlah')
plt.show()

# Monthly sales trend
monthly_sales = data.groupby(['Year', 'Month'])['Jumlah'].sum().reset_index()
sns.lineplot(data=monthly_sales, x='Month', y='Jumlah', hue='Year')
plt.show()

# Prepare data for modeling
X = data[['Year', 'Month', 'Jenis Barang', 'Jenis Kelamin']]
X = pd.get_dummies(X, drop_first=True)
y = data['Jumlah']

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate metrics