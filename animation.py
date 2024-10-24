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

    def set_weights(self, weights) -> None:
        for w, edge in zip(weights, self.edges):
            edge.set_opacity(w)


class LayerOutEdges(VGroup):
    def __init__(self, left_layer: Layer, right_layer: Layer) -> None:
        self.edges = [
            NodeOutEdges(left_node, right_layer) for left_node in left_layer.nodes
        ]
        super().__init__(*self.edges)

    def set_weights(self, weights) -> None:
        for w, edges in zip(weights, self.edges):
            edges.set_weights(w)


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

def normalize_values(matrix: np.array):
    v_min = np.min(matrix)
    v_max = np.max(matrix)
    return (matrix - v_min) / (v_max - v_min)

def normalize_weights_layers(weights):
    return [normalize_values(w) for w in weights]


class NNAnimation(Scene):
    def construct(self):
        df = pd.read_csv("data/classification/data.three_gauss.train.100.csv")
        x = np.array(df.iloc[:,:-1]).reshape((300, 2))
        y = np.array(df.iloc[:,-1])
        num_classes = 3

        one_hot_encoded = np.zeros((y.size, num_classes))
        one_hot_encoded[np.arange(y.size), y - 1] = 1
        one_hot_encoded = one_hot_encoded.reshape((300, 3))

        nn1 = NN(input_shape=(0,2))
        nn1.add_new_random_layer(4, GELU())
        nn1.add_new_random_layer(4, GELU())
        nn1.add_new_random_layer(3, activation=Linear())

        nodes = Nodes([2, 4, 4, 3])
        edges = Edges(nodes)
        self.play(Create(nodes))
        self.play(Create(edges))
        self.wait(DELAY)
        edges.set_weights(normalize_weights_layers(get_weights(nn1)))

        LR = 0.01
        ANIMATION_START = 0
        ANIMATION_END = 10_000
        ANIMATION_STEP = 1000
        for step in range(10_000):
            index = np.random.randint(x.shape[0])
            x_i = x[index]
            y_i = y[index]
            yhat_i = nn1.apply(x_i)

            # forward pass animation
            if step >= ANIMATION_START and step < ANIMATION_END and (step % ANIMATION_STEP) == 0:
                W = get_weights(nn1)
                self.wait(DELAY)
                vec = x_i
                for l_idx, layer in enumerate(nodes.layers):
                    layer.set_values(normalize_values(vec))
                    if l_idx < len(nodes.layers) - 1:
                        vec = vec @ W[l_idx]
                    self.wait(DELAY)

            g, gb = nn1.backpropagate(yhat_i, y_i)
            for i in range(len(nn1.layers)):
                nn1.layers[i].weights -= LR * g[i]
                nn1.layers[i].bias -= LR * gb[i]
            
            # back propagation animation
            if step >= ANIMATION_START and step < ANIMATION_END and (step % ANIMATION_STEP) == 0:
                W = normalize_weights_layers(get_weights(nn1))
                for w_idx in range(len(nodes.layers) - 2, -1, -1):
                    edges.edges_groups[w_idx].set_weights(W[w_idx])
                    self.wait(DELAY)
                nodes.set_zeros()
        self.wait(5)
