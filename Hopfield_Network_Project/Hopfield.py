import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import ipywidgets as widgets
from IPython.display import display

class HopfieldNetwork:
    def __init__(self, n_neurons, shape=None):
        self.n_neurons = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons)) 
        self.state = np.random.choice([-1, 1], size=n_neurons) 
        self.shape = shape

        if np.prod(self.shape) != n_neurons:
            raise ValueError("Shape must match the number of neurons.")

    def train(self, patterns, learning_rule, eta=0.1):
        if learning_rule == 'hebbian':
            for pattern in patterns:
                self.weights += np.outer(pattern, pattern) 
        elif learning_rule == 'oja':
            for pattern in patterns:
                output = np.dot(self.weights, pattern)
                for i in range(self.n_neurons):
                    self.weights[i] += eta * output[i] * (pattern - self.weights[i] * output[i])
        else:
            raise ValueError("Learning rule must be either 'hebbian' or 'oja'")

        np.fill_diagonal(self.weights, 0)
        self.weights = self.weights / len(patterns)

    def update_asynchronous(self, neuron):

            # neuron = np.random.randint(0, self.n_neurons) 
            input_sum = np.dot(self.weights[neuron], self.state)
            self.state[neuron] = 1 if input_sum >= 0 else -1

    def update_synchronous(self):
        input_sums = np.dot(self.weights, self.state)
        self.state = np.where(input_sums >= 0, 1, -1)

    def retrieve(self, input_pattern, mode='asynchronous', tolerance=1e-2):
        self.state = np.array(input_pattern)
        states = [self.state.copy()]
        previous_energy = self.energy() 

        if mode == 'asynchronous':
            for neuron in range(self.n_neurons):
                self.update_asynchronous(neuron)
                states.append(self.state.copy())
        elif mode == 'synchronous':
            self.update_synchronous()
            states.append(self.state.copy())
        else:
            raise ValueError("Mode must be 'asynchronous' or 'synchronous'")


        current_energy = self.energy()

        if abs(current_energy - previous_energy) < tolerance:
            # break
            pass

        previous_energy = current_energy 

        return states
    
    def energy(self):

        return -0.5 * np.dot(self.state.T, np.dot(self.weights, self.state))

    def interactive_visualize(self, states):

        current_step = [0] 

        def update_plot(step):

            ax.imshow(states[step].reshape(self.shape), cmap='gray', interpolation='nearest')
            ax.set_title(f"Step {step + 1}/{len(states)}")
            plt.draw()

        def next_state(event):

            if current_step[0] < len(states) - 1:
                current_step[0] += 1
                update_plot(current_step[0])

        def prev_state(event):

            if current_step[0] > 0:
                current_step[0] -= 1
                update_plot(current_step[0])


        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        update_plot(0)


        ax_prev = plt.axes([0.1, 0.05, 0.1, 0.075])  
        ax_next = plt.axes([0.8, 0.05, 0.1, 0.075])
        btn_prev = Button(ax_prev, 'Previous')
        btn_next = Button(ax_next, 'Next')

        btn_prev.on_clicked(prev_state)
        btn_next.on_clicked(next_state)

        plt.show()