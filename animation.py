from collections.abc import Iterable
from manim import *
import numpy as np


LAYER_SIZES = [3, 3, 2]

W1 = np.array(
    [
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.3, 0.3, 0.3],
    ]
)

W2 = np.array(
    [
        [0.1, 0.1],
        [0.2, 0.2],
        [0.2, 0.2],
    ]
)
W = [W1, W2]
INPUT = [
        np.array([1.0, 0.5, 0.5]),
        np.array([0.0, 0.1, 0.3]),
        np.array([0.9, 0.2, 0.2]),
        np.array([0.4, 0.7, 0.6]),
        np.array([0.7, 0.5, 0.9]),
]

NODE_RADIOUS = 0.2
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


class NNAnimation(Scene):
    def construct(self):
        nodes = Nodes(LAYER_SIZES)
        edges = Edges(nodes)
        self.play(Create(nodes))
        self.play(Create(edges))
        self.wait(DELAY)
        edges.set_weights(W)
        for input_vec in INPUT:
            # forward pass
            self.wait(DELAY)
            vec = input_vec
            for l_idx, layer in enumerate(nodes.layers):
                layer.set_values(vec)
                if l_idx < len(nodes.layers) - 1:
                    vec = vec @ W[l_idx]
                self.wait(DELAY)
            # back propagation
            for w_idx in range(len(nodes.layers) - 2, -1, -1):
                W[w_idx] *= 1.3
                edges.edges_groups[w_idx].set_weights(W[w_idx])
                self.wait(DELAY)
            nodes.set_zeros()
        self.wait(5 * DELAY)
