import random

import numpy as np


class Ant:
    """
    Minimum self organized unit
    :return:
    """

    def __init__(self, initial_node: int, nb_nodes: int):
        self.initial_node = initial_node
        self.visited_nodes = [initial_node]
        self.nodes_to_visit = list(range(nb_nodes))
        self.nodes_to_visit.remove(initial_node)
        self.cost: int = 0
        self.nb_nodes = nb_nodes

    def p(self, i, j, mat_probs) -> float:
        # m = (phero_matrix**self.alpha) * (distances ** (-self.beta))
        # return (
        #     1 / distances[i, j]
        #     if phero_matrix.any()
        #     else m[i, j] / sum(m[i, self.nodes_to_visit])
        # )
        return mat_probs[i, j]

    def simulate(self, mat_probs: np.ndarray) -> bool:
        while len(self.nodes_to_visit) > 0:
            probas = [
                self.p(self.visited_nodes[-1], j=j, mat_probs=mat_probs)
                for j in self.nodes_to_visit
            ]
            next_node = random.choices(self.nodes_to_visit, weights=probas, k=1)[0]
            self.nodes_to_visit.remove(next_node)
            self.visited_nodes.append(next_node)
        return True
