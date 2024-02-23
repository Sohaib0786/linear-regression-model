import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# writing our data
sizes = np.array([6, 8, 10, 12, 14]).reshape(-1, 1)  # Reshape to a 2D array
prices = np.array([10, 15, 20, 25, 30])

# Create a Linear Regression model
reg = LinearRegression()

# Fitting  the model into the data
reg.fit(sizes, prices)

# now we have to make prediction
predicted_prices = reg.predict(sizes)






plt.scatter(sizes, prices, label='Data')
plt.plot(sizes, predicted_prices, label='Linear Regression', color='green')
plt.xlabel('Size')
plt.ylabel('Price')
plt.legend()
plt.show()
