import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class HopfieldNetwork:
    def __init__(self, n_neurons, shape=None):
        self.n_neurons = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons)) 
        self.state = np.random.choice([-1, 1], size=n_neurons) 
        self.shape = shape

        if shape is not None and np.prod(self.shape) != n_neurons:
            raise ValueError("Shape must match the number of neurons.")
        
        print(f"Initialized HopfieldNetwork with {self.n_neurons} neurons and shape {self.shape}")

    def train(self, patterns, learning_rule='hebbian', eta=0.01):
        print(f"Training with {len(patterns)} patterns using {learning_rule} rule.")
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

        self.weights = (self.weights + self.weights.T) / 2
        np.fill_diagonal(self.weights, 0)

        print(f"Weights shape after training: {self.weights.shape}")

    def update_asynchronous(self, neuron):
        input_sum = np.dot(self.weights[neuron], self.state)
        self.state[neuron] = 1 if input_sum >= 0 else -1

    def update_synchronous(self):
        input_sums = np.dot(self.weights, self.state)
        self.state = np.where(input_sums >= 0, 1, -1)

    def retrieve(self, input_pattern, mode='asynchronous', max_iterations=100):
        self.state = np.array(input_pattern)
        states = [self.state.copy()]
        energy_history = [self.energy()]
        iteration = 0
        converged = False
        seen_states = {tuple(self.state): 0}

        print(f"Starting retrieval with mode '{mode}' and max_iterations={max_iterations}")
        print(f"Initial energy: {energy_history[-1]}")

        while not converged and iteration < max_iterations:
            previous_state = self.state.copy()
            if mode == 'asynchronous':
                neurons_list = np.arange(self.n_neurons)
                np.random.shuffle(neurons_list)
                for neuron in neurons_list:
                    self.update_asynchronous(neuron)
            elif mode == 'synchronous':
                self.update_synchronous()
            else:
                raise ValueError("Mode must be 'asynchronous' or 'synchronous'")

            states.append(self.state.copy())
            current_energy = self.energy()
            energy_history.append(current_energy)

            print(f"Iteration {iteration + 1}: Energy = {current_energy}")

            if np.array_equal(self.state, previous_state):
                converged = True
                print("Convergence achieved.")
            elif tuple(self.state) in seen_states:
                converged = True
                print(f"Detected cycle between iterations {seen_states[tuple(self.state)]} and {iteration + 1}.")
            else:
                seen_states[tuple(self.state)] = iteration + 1

            iteration += 1

        if iteration == max_iterations and not converged:
            print("Reached maximum iterations without full convergence.")

        return states, energy_history

    def energy(self, state=None):
        if state is None:
            state = self.state
        energy = -0.5 * np.dot(state.T, np.dot(self.weights, state))
        return energy

    def is_stable(self, pattern, mode='synchronous', max_iterations=100):
        states, _ = self.retrieve(pattern, mode=mode, max_iterations=max_iterations)
        return np.array_equal(states[-1], pattern)

    def interactive_visualize(self, states, interval=500, repeat=True):
        if self.shape is None:
            raise ValueError("Shape must be defined for visualization.")
        if not all(state.size == np.prod(self.shape) for state in states):
            raise ValueError("Not all states match the network shape.")

        reshaped_states = [state.reshape(self.shape) for state in states]

        fig, ax = plt.subplots()
        ax.axis('off')  

        img = ax.imshow(reshaped_states[0], cmap='gray', vmin=-1, vmax=1)

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

    def plot_states_as_bitmaps(self, states, titles=None):
        """
        Wyświetla każdy stan jako bitmapę 5x5.
        """
        num_states = len(states)
        cols = 4
        rows = (num_states + cols - 1) // cols 

        plt.figure(figsize=(cols * 2, rows * 2))
        for idx, state in enumerate(states):
            plt.subplot(rows, cols, idx + 1)
            plt.imshow(state.reshape(self.shape), cmap='gray', vmin=-1, vmax=1)
            if titles:
                plt.title(titles[idx])
            else:
                plt.title(f'State {idx}')
            plt.axis('off')
        plt.tight_layout()
        plt.show()