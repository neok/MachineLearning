import numpy as np
import sys


def compute_hypothesis_for_a_line(theta0, theta1, dataset):
    e = 0
    length = len(dataset)
    for el in dataset:
        e += (1/length) * pow(el[1] - (theta0 * el[0] + theta1), 2)

    return e


def comp(dataset, current_thetha_zero, current_thetha_one, learning_rate):
    theta_zero = 0
    theta_one = 0
    length = len(dataset)
    for el in dataset:
        x = el[0]
        y = el[1]
        theta_zero += -(2/length) * (y - (current_thetha_one*x + current_thetha_zero))
        theta_one += -(2/length) * x * (y - (current_thetha_one * x + current_thetha_zero))
    new_theta_zero = current_thetha_zero - (learning_rate * theta_zero)
    new_theta_one = current_thetha_one - (learning_rate * theta_one)

    return [new_theta_zero, new_theta_one]


def run_gradient_descent(dataset, theta0, theta1, learning_rate, num_interations):
    temp_theta_zero = theta0
    temp_theta_one = theta1
    for i in range(num_interations):
        temp_theta_zero, temp_theta_one = comp(
            np.array(dataset), temp_theta_zero, temp_theta_one, learning_rate)

    return [temp_theta_zero, temp_theta_one]


def runScript():
    dataset = np.genfromtxt('dataset.csv', delimiter=',')
    learning_rate = 0.000009999
    starting_m = 0
    starting_b = 0
    num_interations = 1000
    print('Starting gradient descent x={0} y={1} e={2}'.format(starting_m, starting_b, compute_hypothesis_for_a_line(starting_m, starting_b, dataset)))
    [theta1, theta0] = run_gradient_descent(dataset, starting_b, starting_m, learning_rate, num_interations)
    print('Result {0} {1}'.format(theta0, theta1))
    print('Ending gradient descent x={0} y={1} e={2}'.format(theta0, theta1, compute_hypothesis_for_a_line(theta0, theta1, dataset)))

runScript()
