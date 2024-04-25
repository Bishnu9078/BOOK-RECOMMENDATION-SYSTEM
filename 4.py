import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Read the data
df = pd.read_csv("book files.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Check basic information about the dataframe
print(df.info())

# Drop unnecessary columns
df.drop(['bookID', 'isbn', 'isbn13'], axis=1, inplace=True)

# Extract the year from the publication date
df['year'] = df['publication_date'].str.split('/').apply(lambda x: x[2] if len(x) > 2 else None)
df['year'] = df['year'].astype(float)

# Handle missing values
df.dropna(inplace=True)

# Define input features (X) and target variable (Y)
X = df['average_rating'].values.reshape(-1, 1)
Y = df['ratings_count'].values

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)

# Data analysis and visualization

# First graph: Top 10 Authors with Maximum Book Publications
plt.figure(figsize=(20, 6))
sns.countplot(x='authors', data=df, order=df['authors'].value_counts().iloc[:10].index)
plt.title("Top 10 Authors with Maximum Book Publications")
plt.xticks(fontsize=12, rotation=45)
plt.show()

# Second graph: Plotting the second graph
plt.figure(figsize=(10, 6))
data = {
    'Graph 1': [1, 2, 3, 4, 5],
    'Graph 2': [5, 4, 3, 2, 1],
    'Graph 3': [2, 4, 6, 8, 10],
}
colors = [plt.cm.tab10(i) for i in range(len(data))]
for label, values in data.items():
    plt.plot(values, color=colors[len(plt.gca().lines) - 1], label=label)

plt.legend()
plt.title('Second Graph')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.show()

# Evaluate the model
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)

plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred)
plt.title('Actual vs. Predicted Ratings Count')
plt.xlabel('Actual Ratings Count')
plt.ylabel('Predicted Ratings Count')
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)

# Visualize the actual vs. predicted ratings count
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred)
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], '--', color='red')  # Diagonal line
plt.title('Actual vs. Predicted Ratings Count')
plt.xlabel('Actual Ratings Count')
plt.ylabel('Predicted Ratings Count')
plt.show()
