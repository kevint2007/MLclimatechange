import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
%matplotlib inline

data= pd.read_csv('/content/GlobalTemperatures.csv')
data.head()
data.shape
#Visualizing the upward trend of global temperatures
data = data[['dt', 'LandAndOceanAverageTemperature']]
data.dropna(inplace=True)

# Convert 'dt' to datetime and extract the year
data['dt'] = pd.to_datetime(data['dt'])
data['year'] = data['dt'].dt.year

# Group by year and calculate mean temperature
data = data.groupby('year')['LandAndOceanAverageTemperature'].mean().reset_index()

# Plot the trend
plt.figure(figsize=(16, 6))
ax = sns.lineplot(
    x=data['year'],
    y=data['LandAndOceanAverageTemperature']
)
ax.set_title('Average Global Temperature Movement')
ax.set_ylabel('Average Global Temperature')
ax.set_xlabel('Year (1750 - 2015)')
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X = data[['year']]  # Feature (Year)
y = data['LandAndOceanAverageTemperature']  # Target (Temperature)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

plt.figure(figsize=(10,6))
plt.scatter(X_train,y_train,color ='green', label ="Training Set")
plt.plot(X_train, regressor.predict(X_train), color='blue', label='Linear regression line')
plt.title('Average Global Temperature (Training Set)')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()

#Visualizing the Test set results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='red', label='Test set')
plt.plot(X_train, regressor.predict(X_train), color='blue', label='Linear regression line')
plt.title('Average Global Temperature (Test Set)')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()

#prediction
year = 2125
predicted_temp = regressor.predict(np.array([[year]]))
print(f"The predicted temperature for {year} is {predicted_temp[0]:.2f} °C.")

