import numpy as np
import matplotlib.pyplot as plt

def linear_regression(X,y):
    x_mean = np.mean(X)
    y_mean = np.mean(y)
    numerator = np.sum((X - x_mean) * (y - y_mean))
    denominator = np.sum((X - x_mean) ** 2)
    m = numerator / denominator
    b = y_mean - (m * x_mean)
    return m, b

def predict(X, m, b):
    return m * X + b

def rmse(y_true, y_pred):
    return np.sqrt(np.mean(y_true - y_pred) ** 2)

X = np.array([7,8,10,12,15,18])
Y = np.array([9,10,12,13,16,20])

# training the model
m, b = linear_regression(X,Y)

# making predctions
predictions = predict(X, m, b)

# calculating RMSE
error = rmse(Y, predictions)

print(f'Slope m: {m}')
print(f'Intercept b: {b}')
print(f'Predictions: {predictions}')
print(f'RMSE: {error}')

plt.scatter(X,Y, color='blue', label='Data points')
plt.plot(X, predictions, color='red', label='Regression Line')
plt.xlabel('Independent variable X')
plt.ylabel('Dependent variable y')
plt.title('Linear Regression Model')
plt.legend()
plt.show()