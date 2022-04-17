import random

import numpy as np


class Ant:
    """
    Minimum self organized unit
    :return:
    """

    def __init__(self, initial_node: int, nb_nodes: int, alpha: float, beta: float):
        self.initial_node = initial_node
        self.visited_nodes = [initial_node]
        self.nodes_to_visit = list(range(nb_nodes))
        self.nodes_to_visit.remove(initial_node)
        self.cost: int = 0
        self.nb_nodes = nb_nodes
        self.alpha = alpha
        self.beta = beta

    def p(self, i, j, phero_matrix, distances) -> float:
        m = (phero_matrix**self.alpha) * (distances ** (-self.beta))
        return (
            1 / distances[i, j]
            if phero_matrix.any()
            else m[i, j] / sum(m[i, self.nodes_to_visit])
        )

    def simulate(self, phero_matrix: np.ndarray, distances: np.ndarray) -> bool:
        while len(self.nodes_to_visit) > 0:
            probas = [
                self.p(
                    self.visited_nodes[-1],
                    j=j,
                    phero_matrix=phero_matrix,
                    distances=distances,
                )
                for j in self.nodes_to_visit
            ]
            next_node = random.choices(self.nodes_to_visit, weights=probas, k=1)[0]
            self.nodes_to_visit.remove(next_node)
            self.visited_nodes.append(next_node)
        return True
