import numpy as np
from numpy import ndarray
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from typing import Tuple


def get_theta() -> ndarray:
    try:
        with open("theta.pickle", "rb") as f:
            theta = pickle.load(f)
    except FileNotFoundError:
        theta = np.zeros((2, 1))
    return theta


def get_data() -> Tuple[ndarray, ndarray]:
    try:
        with open("data.csv", "r") as f:
            data = pd.read_csv(f)
        x = data['km'].to_numpy().reshape((-1, 1))
        y = data['price'].to_numpy().reshape((-1, 1))
        return x, y
    except FileNotFoundError as e:
        print(e)
        exit()


def add_intercept(x: ndarray) -> ndarray:
    return np.hstack((np.ones((x.shape[0], 1)), x))


def minmax(x: ndarray) -> ndarray:
    xmin, xmax = x.min(), x.max()
    return (x - xmin) / (xmax - xmin)


def mse(y: ndarray, y_hat: ndarray) -> float:
    cost = (y_hat - y).reshape(-1)
    return np.dot(cost, cost) / y.shape[0]


def draw(x, x_prime_T, y, m):
    y_hat = np.zeros_like(y)
    fig = plt.figure(figsize=(10, 6))
    ax2 = fig.add_subplot(121)
    ax3 = fig.add_subplot(122, projection="3d")
    # ax2
    ax2.scatter(x, y, label="true")
    line2, = ax2.plot(x, y_hat, label="predicted")
    ax2.set_xlabel("Mileage")
    ax2.set_ylabel("Price")
    ax2.legend()
    # ax3
    zero0, zero1 = 8008.43983265, -4656.59144472
    zero = [[zero0], [zero1]]
    d = np.abs(zero).max()
    theta0, theta1 = np.meshgrid(np.linspace(zero0 - d, zero0 + d, 100),
                                 np.linspace(zero1 - d, zero1 + d, 100))
    cost = np.subtract(
        np.dot(np.dstack([theta0, theta1]), x_prime_T), y.reshape(-1))
    cost = (cost * cost).sum(axis=2) / 2 / m
    ax3.contour3D(theta0, theta1, cost, 150, cmap='twilight', alpha=0.4)
    mse_ = mse(y, y_hat)
    line3, = ax3.plot([0], [0], [mse_ / 2], c='r', marker='o', ms=3)
    ax3.set_xlabel("$\\theta_0$")
    ax3.set_ylabel("$\\theta_1$")
    ax3.set_zlabel("Cost Function $J(\\theta_0, \\theta_1)$")
    fig.canvas.draw()
    plt.pause(0.1)
    return fig, ax2, ax3, line2, line3


def draw2(x, x_prime, y, theta, fig, ax2, line2, line3):
    y_hat = x_prime.dot(theta)
    mse_ = mse(y, y_hat)
    ax2.set_title(f"MSE: {mse_}")
    line2.set_data(x, y_hat)
    x_tmp, y_tmp, z_tmp = line3.get_data_3d()
    line3.set_data_3d(np.append(x_tmp, theta[0]), np.append(
        y_tmp, theta[1]), np.append(z_tmp, mse_ / 2))
    fig.canvas.draw()
    plt.pause(0.1)
