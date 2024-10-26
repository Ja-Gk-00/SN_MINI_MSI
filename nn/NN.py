from abc import ABC, abstractmethod

import numpy as np
import pickle

EPS = 1e-15

class LossFunction(ABC):

    @abstractmethod
    def fun(self, yhat, y):
        pass

    @abstractmethod
    def grad(self, yhat, y):
        pass

    def __call__(self, yhat, y):
        return self.fun(yhat, y)

class MSE(LossFunction):
    def fun(self, yhat, y):
        return np.sum((yhat - y)**2) / len(yhat)
    
    def grad(self, yhat, y):
        return 2 * (yhat-y)

class LogisticCrossEntropy(LossFunction):
    def fun(self, yhat , y):
        y_pred = np.clip(yhat, EPS, 1 - EPS)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    
    def grad(self, yhat, y):
        y_pred = np.clip(yhat, EPS, 1 - EPS)
        return -(y / y_pred) + (1 - y) / (1 - y_pred)
    


class ActivationFunction(ABC):

    @abstractmethod
    def fun(self, x):
        pass

    @abstractmethod
    def grad(self, x):
        pass

    def __call__(self, x):
        return self.fun(x)
    

class Linear(ActivationFunction):
    def fun(self, x):
        return x

    def grad(self, x):
        return np.diagflat(np.ones(x.shape))
    

class Sigmoid(ActivationFunction):
    def fun(self, x):
        return 1 / (1 + np.exp(-x))
    
    def grad(self, x):
        return np.diagflat(self.fun(x) * (1 - self.fun(x)))
    
class Softmax(ActivationFunction):
    def fun(self, x):
        shifted_x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted_x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def grad(self, x):
        s = self.fun(x).reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)
    
class GELU(ActivationFunction):
    def fun(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * np.power(x, 3))))
    
    def grad(self, x):
        tanh_part = np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))
        return np.diagflat(0.5 * (1 + tanh_part) + 0.5 * x * (1 - tanh_part**2) * (np.sqrt(2/np.pi) + 0.134145 * x**2))


class ELU(ActivationFunction):
    def __init__(self, alpha = 1.0):
        self.alpha = alpha

    def fun(self, x):
        return np.where(x >= 0, x, self.alpha * (np.exp(x) - 1))
    
    def grad(self, x):
        return np.diagflat(np.where(x >= 0, 1, self.alpha * np.exp(x)))
    

class Layer:
    def __init__(self, neurons, input_shape, weights, bias, bias_active, activation: ActivationFunction):
        self.neurons = neurons
        self.input_shape = input_shape
        assert weights.shape == (input_shape[1], neurons)
        self.weights = weights
        assert bias.shape == (1, neurons)
        self.bias = bias if bias_active else np.zeros_like(bias)
        self.bias_active = bias_active
        self.activation = activation
        self.last_a = None

    def make_factory(neurons, input_shape, activation: ActivationFunction, factory, bias_active = True):
        return Layer(
            neurons = neurons,
            input_shape = input_shape,
            weights = factory((input_shape[1], neurons)),
            bias = factory((1, neurons)),
            activation = activation,
            bias_active=bias_active,
        )
    
    def make_zero(neurons, input_shape, activation, bias_active = True):
        return Layer.make_factory(neurons, input_shape, activation, np.zeros, bias_active)
    
    def make_random(neurons, input_shape, activation, bias_active = True):
        random_balanced = lambda shape: np.random.random(shape) - 0.5
        return Layer.make_factory(neurons, input_shape, activation, random_balanced, bias_active)
    
    def apply(self, inputs):
        intensities = inputs @ self.weights
        if self.bias_active:
            self.last_a = intensities + self.bias
        else:
            self.last_a = intensities
        return self.activation(self.last_a)
    
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

    def add_new_zero_layer(self, neurons, activation: ActivationFunction | None = None, bias_active=True):
        activation = activation or Sigmoid()
        layer = Layer.make_zero(
            neurons,
            self.get_last_shape(),
            activation,
            bias_active,
        )
        self.layers.append(layer)
        return layer
    
    def add_new_random_layer(self, neurons, activation: ActivationFunction | None = None, bias_active=True):
        activation = activation or Sigmoid()
        layer = Layer.make_random(
            neurons,
            self.get_last_shape(),
            activation,
            bias_active,
        )
        self.layers.append(layer)
        return layer
    
    def apply(self, inputs):
        self.last_inputs = inputs
        x = inputs
        for layer in self.layers:
            x = layer.apply(x)
        return x

    def calculate_errors(self, yhat, y, loss):
        errors = [None] * len(self.layers)
        errors[-1] = loss.grad(yhat, y)
        for i in range(len(errors)-2, -1, -1):
            uhm = errors[i+1] @ np.transpose(self.layers[i+1].weights)
            grad_fun = self.layers[i].activation.grad
            errors[i] = uhm @ grad_fun(self.layers[i].last_a)
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
            
            grad[k] = np.vstack(np.transpose(f_a)) @ errors[k]
            grad_b[k] = errors[k] 
        return grad, grad_b

    def get_zero_grads(self):
        grad = [None] * len(self.layers)
        grad_b = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            grad[i] = np.zeros(layer.weights.shape)
            grad_b[i] = np.zeros(layer.bias.shape)
        return grad, grad_b

    def backpropagate(self, yhat, y, loss):
        errors = self.calculate_errors(yhat, y, loss)
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

    def gradient_descent(self, x, y, loss: LossFunction, rate=1e-3, weights_path=None, errors_path=None, log_format='pickle'):
        sumg, sumgb = self.get_zero_grads()
        for i, x_i in enumerate(x):
            yhat = self.apply(x_i.reshape(1, -1))
            g, gb = self.backpropagate(yhat, y[i], loss)
            for i in range(len(self.layers)):
                sumg[i] -= rate * g[i]
                sumgb[i] -= rate * gb[i]

        for i in range(len(self.layers)):
            self.layers[i].weights += sumg[i] / x.shape[0]
            self.layers[i].bias += sumgb[i] / x.shape[0]

        if weights_path and errors_path:
            self.log_data(weights_path, errors_path, log_format)

    def batch_descent(self, x, y, loss: LossFunction, rate=1e-3, batch_size=None, weights_path=None, errors_path=None, log_format='pickle'):
        batch_size = batch_size or x.shape[0]//10
        indexes = np.random.randint(x.shape[0], size=(batch_size, 1))
        x_chosen = x[indexes]
        y_chosen = y[indexes]
        self.gradient_descent(x_chosen, y_chosen, loss, rate=rate, weights_path=weights_path, errors_path=errors_path, log_format=log_format)

    def stochastic_descent(self, x, y, loss: LossFunction, rate=1e-3, weights_path=None, errors_path=None, log_format='pickle'):
        index = np.random.randint(x.shape[0])
        x_i = x[index]
        y_i = y[index]
        yhat_i = self.apply(x_i.reshape(1, -1))
        g, gb = self.backpropagate(yhat_i, y_i, loss)
        for i in range(len(self.layers)):
            self.layers[i].weights -= rate * g[i]
            self.layers[i].bias -= rate * gb[i]

        if weights_path and errors_path:
            self.log_data(weights_path, errors_path, log_format)
