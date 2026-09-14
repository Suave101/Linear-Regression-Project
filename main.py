import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def parseInput():
    data = np.genfromtxt("data.txt", delimiter=",")
    d_min = data.min(axis=0)
    d_max = data.max(axis=0)
    data = (data - d_min) / (d_max - d_min)
    return data

def get_theta_history(X, y, theta, iterations, learning_rate):
    m = len(y)
    theta_history = [theta.copy()]

    for i in range(iterations):
        prediction = X @ theta # Matrix multiplication
        error = prediction - y
        gradient = (1 / m) * (X.T @ error)
        theta = theta - (learning_rate * gradient)
        theta_history.append(theta.copy())

    return theta_history


if __name__ == "__main__":
    raw_data = parseInput()
    x_vals = raw_data[:, 0]
    y_vals = raw_data[:, 1]
    X = np.c_[np.ones(len(y_vals)), x_vals]

    # Get the history of weights
    iters = 1200
    history = get_theta_history(X, y_vals, np.zeros(2), iters, 0.1)

    # Create the figure and layout
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)  # Make room for the slider

    # Initial plot
    scatter = ax.scatter(x_vals, y_vals, color='red', alpha=0.5)
    line, = ax.plot(x_vals, X @ history[0], color='blue', lw=2)
    ax.set_title("Slide to see Gradient Descent Progress")

    # Add the slider
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])  # [left, bottom, width, height]
    slider = Slider(ax_slider, 'Iteration', 0, iters, valinit=0, valfmt='%d')


    # Update function called when slider moves
    def update(val):
        iteration = int(slider.val)
        current_theta = history[iteration]
        line.set_ydata(X @ current_theta)
        fig.canvas.draw_idle()


    slider.on_changed(update)
    plt.show()
