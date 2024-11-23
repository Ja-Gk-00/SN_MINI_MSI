import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class HopfieldNetwork:
    def __init__(self, n_neurons, shape=None):
        self.n_neurons = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons)) 
        self.state = np.random.choice([-1, 1], size=n_neurons) 
        self.bias = np.zeros(n_neurons)
        self.shape = shape

        if np.prod(self.shape) != n_neurons:
            raise ValueError("Shape must match the number of neurons.")

    def train(self, patterns, learning_rule, eta=0.01):
        if learning_rule == 'hebbian':
            for pattern in patterns:
                self.weights += np.outer(pattern, pattern) 
        elif learning_rule == 'oja':
            for pattern in patterns:
                for i in range(self.n_neurons):
                    y_i = np.dot(self.weights[i], pattern)
                    for j in range(self.n_neurons):
                        self.weights[i, j] += eta * y_i * (pattern[j] - y_i * self.weights[i, j])
        else:
            raise ValueError("Learning rule must be either 'hebbian' or 'oja'")

        np.fill_diagonal(self.weights, 0)
        self.weights = self.weights / len(patterns)
        self.bias = np.mean(patterns, axis=0)

    def update_asynchronous(self, neuron):
            # neuron = np.random.randint(0, self.n_neurons) 
            input_sum = np.dot(self.weights[neuron], self.state) + self.bias[neuron]
            self.state[neuron] = 1 if input_sum >= 0 else -1

    def update_synchronous(self):
        input_sums = np.dot(self.weights, self.state) + self.bias
        self.state = np.where(input_sums >= 0, 1, -1)

    def retrieve(self, input_pattern, mode='asynchronous', tolerance=1e-2):
        self.state = np.array(input_pattern)
        states = [self.state.copy()]
        previous_energy = self.energy() 

        if mode == 'asynchronous':
            neurons_list = np.arange(self.n_neurons)
            np.random.shuffle(neurons_list)
            for neuron in neurons_list:
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

        return -0.5 * np.dot(self.state.T, np.dot(self.weights, self.state)) - np.dot(self.bias, self.state)

    def interactive_visualize(self, states, interval=500, repeat=True):
        if not all(state.size == np.prod(self.shape) for state in states):
            raise ValueError("Not all states match the network shape.")

        reshaped_states = [state.reshape(self.shape) for state in states]

        fig, ax = plt.subplots()
        ax.axis('off')  

        img = ax.imshow(reshaped_states[0], cmap='gray', vmin=0, vmax=1)

        def update(frame):
            img.set_array(reshaped_states[frame])
            return [img]

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(reshaped_states),
            interval=interval,
            repeat=repeat
        )

        return anim