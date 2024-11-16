from collections.abc import Iterable
from manim import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nn.NN import *


NODE_RADIOUS = 0.1
NODE_BORDER = 1.0


ANIMATION_SPEED = 60.0
DELAY = 30.0 / ANIMATION_SPEED


class Layer(VGroup):
    def __init__(self, size: int) -> None:
        self.nodes = [Circle(color=WHITE, radius=NODE_RADIOUS) for _ in range(size)]
        for node in self.nodes:
            node.set_stroke(WHITE, width=NODE_BORDER)
        super().__init__(*self.nodes)
        self.arrange(DOWN, buff=SMALL_BUFF)

    def set_values(self, values) -> None:
        for v, node in zip(values, self.nodes):
            node.set_fill(WHITE, opacity=v)

    def set_zeros(self) -> None:
        self.set_values(np.zeros(len(self.nodes)))


class Nodes(VGroup):
    def __init__(self, layers_sizes: Iterable[int]) -> None:
        self.layers = [Layer(size) for size in layers_sizes]
        super().__init__(*self.layers)
        self.arrange(RIGHT, buff=LARGE_BUFF)

    def set_zeros(self) -> None:
        for layer in self.layers:
            layer.set_zeros()


class NodeOutEdges(VGroup):
    def __init__(self, left_node: Circle, right_layer: Layer) -> None:
        self.edges = [
            Line(
                start=left_node.get_center(),
                end=right_node.get_center(),
                color=WHITE,
            )
            for right_node in right_layer
        ]
        super().__init__(*self.edges)

    def set_weights(self, weights, backpropagate = False) -> None:
        for w, edge in zip(weights, self.edges):
            if backpropagate:
                edge.set_color(YELLOW)
                edge.set_opacity(abs(w))
            else:
                edge.set_color(RED if w > 0 else BLUE)
                edge.set_opacity(abs(w))


class LayerOutEdges(VGroup):
    def __init__(self, left_layer: Layer, right_layer: Layer) -> None:
        self.edges = [
            NodeOutEdges(left_node, right_layer) for left_node in left_layer.nodes
        ]
        super().__init__(*self.edges)

    def set_weights(self, weights, backpropagate = False) -> None:
        for w, edges in zip(weights, self.edges):
            edges.set_weights(w, backpropagate)


class Edges(VGroup):
    def __init__(self, nodes: Nodes) -> None:
        self.edges_groups = [
            LayerOutEdges(left_layer, right_layer)
            for left_layer, right_layer in zip(nodes.layers[:-1], nodes.layers[1:])
        ]
        super().__init__(*self.edges_groups)

    def set_weights(self, weights) -> None:
        for w, edges in zip(weights, self.edges_groups):
            edges.set_weights(w)

def get_weights(network: NN):
    return [l.weights for l in network.layers]

def normalize_min_max(matrix: np.array):
    v_min = np.min(matrix)
    v_max = np.max(matrix)
    return (matrix - v_min) / (v_max - v_min)


def normalize_weights_layers(weights):
    return [Sigmoid()(w) * 2 - 1 for w in weights]


class NNAnimation(Scene):
    def construct(self):
        np.random.seed(42)
        df = pd.read_csv("data/classification/data.three_gauss.train.100.csv")
        x = np.array(df.iloc[:,:-1]).reshape((-1, 2))
        y = np.array(df.iloc[:,-1])
        num_classes = 3

        one_hot_encoded = np.zeros((y.size, num_classes))
        one_hot_encoded[np.arange(y.size), y - 1] = 1
        one_hot_encoded = one_hot_encoded.reshape((-1, num_classes))

        nn1 = NN(input_shape=(0,2))
        nn1.add_new_random_layer(3, ELU())
        nn1.add_new_random_layer(3, ELU())
        nn1.add_new_random_layer(3, activation=Softmax())

        nodes = Nodes([2, 3, 3, 3])
        edges = Edges(nodes)
        self.play(Create(nodes))
        self.play(Create(edges))
        self.wait(DELAY)
        edges.set_weights(normalize_weights_layers(get_weights(nn1)))

        LR = 0.0003
        ANIMATION_START = 0
        ANIMATION_END = 15_000
        ANIMATION_STEP = 1000
        loss = LogisticCrossEntropy()
        for step in range(ANIMATION_END):
            index = np.random.randint(x.shape[0])
            x_i = x[index]
            y_i = one_hot_encoded[index]
            yhat_i = nn1.apply(x_i.reshape(1, -1))

            # forward pass animation
            if step >= ANIMATION_START and step < ANIMATION_END and (step % ANIMATION_STEP) == 0:
                W = get_weights(nn1)
                self.wait(DELAY)
                vec = x_i
                for l_idx, layer in enumerate(nodes.layers):
                    layer.set_values(normalize_min_max(vec))
                    if l_idx < len(nodes.layers) - 1:
                        vec = vec @ W[l_idx]
                    self.wait(DELAY)

            old_W = normalize_weights_layers(get_weights(nn1))
            g, gb = nn1.backpropagate(yhat_i, y_i, loss)
            for i in range(len(nn1.layers)):
                nn1.layers[i].weights -= LR * g[i]
                nn1.layers[i].bias -= LR * gb[i]
            
            # back propagation animation
            if step >= ANIMATION_START and step < ANIMATION_END and (step % ANIMATION_STEP) == 0:
                W = normalize_weights_layers(get_weights(nn1))
                # show weights diffs
                for w_idx in range(len(nodes.layers) - 2, -1, -1):
                    edges.edges_groups[w_idx].set_weights(
                        normalize_min_max(abs(W[w_idx] - old_W[w_idx])),
                        backpropagate=True,
                        )
                    self.wait(DELAY)
                # show weights
                for w_idx in range(len(nodes.layers) - 2, -1, -1):
                    edges.edges_groups[w_idx].set_weights(W[w_idx])
                nodes.set_zeros()
        self.wait(5)
