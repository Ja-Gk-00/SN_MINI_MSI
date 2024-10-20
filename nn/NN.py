import numpy as np
import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import typing

def MSE(x, y):
    return np.sum((x - y)**2) / len(x)

def lin_act(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def grad_sigmoid(x: float) -> float:
    return sigmoid(x) * (1 - sigmoid(x))

def GELU(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * np.power(x, 3))))

def ELU(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

def grad_ELU(x, alpha=1.0):
    return np.where(x >= 0, 1, alpha * np.exp(x))

def grad_GELU(x: float) -> float:
    tanh_part = np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))
    return 0.5 * (1 + tanh_part) + 0.5 * x * (1 - tanh_part**2) * (np.sqrt(2/np.pi) + 0.134145 * x**2)

class Layer:
    def __init__(self, neurons, input_shape, weights, bias, activation):
        self.neurons = neurons
        self.input_shape = input_shape
        assert weights.shape == (input_shape[1], neurons)
        self.weights = weights
        assert bias.shape == (1, neurons)
        self.bias = bias
        self.activation = activation
        self.last_a = None

    def make_factory(neurons, input_shape, activation, factory):
        return Layer(
            neurons = neurons,
            input_shape = input_shape,
            weights = factory((input_shape[1], neurons)),
            bias = factory((1, neurons)),
            activation = activation
        )
    
    def make_zero(neurons, input_shape, activation):
        return Layer.make_factory(neurons, input_shape, activation, np.zeros)
    
    def make_random(neurons, input_shape, activation):
        random_balanced = lambda shape: np.random.random(shape) - 0.5
        return Layer.make_factory(neurons, input_shape, activation, random_balanced)
    
    def apply(self, inputs):
        intensities = inputs @ self.weights
        self.last_a = intensities + self.bias
        return self.activation(intensities + self.bias)
    
    def __str__(self):
        return f"LAYER(\nW:\n {repr(self.weights)} \nb:\n{repr(self.bias)})\n"

    def __repr__(self):
        return str(self)


class NN:
    def __init__(self, *layers, input_shape, use_gpu=False):
        self.input_shape = input_shape
        self.layers = [*layers]
        self.errors = None
        self.last_inputs = None
        self.use_gpu = use_gpu
    
    def get_last_shape(self):
        if self.layers:
            return self.layers[-1].weights.shape
        else:
            return self.input_shape

    def add_new_zero_layer(self, neurons, activation=sigmoid):
        layer = Layer.make_zero(
            neurons,
            self.get_last_shape(),
            activation
        )
        self.layers.append(layer)
        return layer
    
    def add_new_random_layer(self, neurons, activation=sigmoid):
        layer = Layer.make_random(
            neurons,
            self.get_last_shape(),
            activation
        )
        self.layers.append(layer)
        return layer
    
    def apply(self, inputs):
        self.last_inputs = inputs
        x = inputs
        for layer in self.layers:
            x = layer.apply(x)
        return x

    def calculate_errors(self, yhat, y):
        errors = [None] * len(self.layers)
        errors[-1] = (yhat - y)
        for i in range(len(errors)-2, -1, -1):
            uhm = errors[i+1] @ np.transpose(self.layers[i+1].weights)
            errors[i] = grad_sigmoid(self.layers[i].last_a) * uhm
        return errors
    
    def calculate_grads(self, errors):
        grad = [None] * len(self.layers)
        grad_b = [None] * len(self.layers)
        
        for k in range(len(errors)):
            if k == 0:
                f_a = self.layers[0].activation(self.last_inputs)
            else:
                cur_layer = self.layers[k-1]
                f_a = cur_layer.activation(cur_layer.last_a)
            
            grad[k] = np.transpose(f_a) @ errors[k]
            grad_b[k] = errors[k] 
        return grad, grad_b

    def get_zero_grads(self):
        grad = [None] * len(self.layers)
        grad_b = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            grad[i] = np.zeros(layer.weights.shape)
            grad_b[i] = np.zeros(layer.bias.shape)
        return grad, grad_b

    def backpropagate(self, yhat, y):
        errors = self.calculate_errors(yhat, y)
        return self.calculate_grads(errors)
    
    def log_data(self, weights_path: str, errors_path: str, save_format: str = 'pickle'):
        weight_data = [(layer.weights, layer.bias) for layer in self.layers]
        if save_format == 'pickle':
            with open(weights_path, 'wb') as f:
                pickle.dump(weight_data, f)
        elif save_format == 'txt':
            with open(weights_path, 'w') as f:
                for w, b in weight_data:
                    f.write(f"Weights:\n{w}\nBiases:\n{b}\n")

    def log_errors(self, errors, path: str, save_format: str = 'pickle'):
        if save_format == 'pickle':
            with open(path, 'wb') as f:
                pickle.dump(errors, f)
        elif save_format == 'txt':
            with open(path, 'w') as f:
                for err in errors:
                    f.write(f"Errors:\n{err}\n")

    def gradient_descent(self, x, y, rate=1e-3, weights_path=None, errors_path=None, log_format='pickle'):
        sumg, sumgb = self.get_zero_grads()
        for i, x_i in enumerate(x):
            yhat = self.apply(x_i)
            g, gb = self.backpropagate(yhat, y[i])
            for i in range(len(self.layers)):
                sumg[i] -= rate * g[i]
                sumgb[i] -= rate * gb[i]

        for i in range(len(self.layers)):
            self.layers[i].weights += sumg[i] / x.shape[0]
            self.layers[i].bias += sumgb[i] / x.shape[0]

        if weights_path and errors_path:
            self.log_data(weights_path, errors_path, log_format)

    def batch_descent(self, x, y, rate=1e-3, batch_size=x.shape[0]//10, weights_path=None, errors_path=None, log_format='pickle'):
        indexes = np.random.randint(x.shape[0], size=(batch_size, 1))
        x_chosen = x[indexes]
        y_chosen = y[indexes]
        self.gradient_descent(x_chosen, y_chosen, rate=rate, weights_path=weights_path, errors_path=errors_path, log_format=log_format)

    def stochastic_descent(self, x, y, rate=1e-3, weights_path=None, errors_path=None, log_format='pickle'):
        index = np.random.randint(x.shape[0])
        x_i = x[index]
        y_i = y[index]
        yhat_i = self.apply(x_i)
        g, gb = self.backpropagate(yhat_i, y_i)
        for i in range(len(self.layers)):
            self.layers[i].weights -= rate * g[i]
            self.layers[i].bias -= rate * gb[i]

        if weights_path and errors_path:
            self.log_data(weights_path, errors_path, log_format)
