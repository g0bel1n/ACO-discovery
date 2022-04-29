import itertools
import threading
from random import randint
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm

from Ant import Ant
from utils import utils

with open("config.yml") as config_file:
    config = yaml.safe_load(config_file)


def thread_task(
    nb_nodes: int,
    initial_node: int,
    listOfSolutions: list,
    mat_probs: np.ndarray,
    distances: np.ndarray,
):
    ant = Ant(initial_node=initial_node, nb_nodes=nb_nodes)
    ant.simulate(mat_probs=mat_probs)
    listOfSolutions.append(
        [
            ant.visited_nodes,
            sum(
                distances[ant.visited_nodes[i], ant.visited_nodes[i + 1]]
                for i in range(nb_nodes - 1)
            ),
        ]
    )


class AntSystem:
    def __init__(
        self,
        BATCH_SIZE: int = config["BATCH_SIZE"],
        NB_NODES: int = config["NB_NODES"],
        MAX_GRID_SIZE: int = config["MAX_GRID_SIZE"],
        MAX_ITER: int = config["MAX_ITER"],
        alpha: float = config["alpha"],
        beta: float = config["beta"],
        evaporation_rate: float = config["evaporation_rate"],
    ):
        self.phero_max: list = []
        self.score: float
        self.alpha = alpha
        self.beta = beta
        self.best_path: list = []
        self.BATCH_SIZE = BATCH_SIZE
        self.MAX_ITER = MAX_ITER
        self.NB_NODES = NB_NODES
        self.MAX_GRID_SIZE = MAX_GRID_SIZE
        self.evaporation_rate = evaporation_rate
        self.phero_matrix = np.ones((NB_NODES, NB_NODES))
        self.nodes_position = np.array(
            [
                [randint(0, MAX_GRID_SIZE), randint(0, MAX_GRID_SIZE)]
                for _ in range(NB_NODES)
            ]
        )

        self.distances = np.zeros((NB_NODES, NB_NODES))

        for i in range(NB_NODES):
            self.distances[i, i] = 1.0
            for j in range(i + 1, NB_NODES):
                self.distances[i, j] = utils.distance(
                    self.nodes_position[i], self.nodes_position[j]
                )
                self.distances[j, i] = self.distances[i, j]
        self.listOfSolutions = []

    def plot_graph(self):

        fig, ax = plt.subplots()

        plt.scatter(self.nodes_position[:, 0], self.nodes_position[:, 1], color="red")

        for i in range(self.NB_NODES - 1):
            ind1, ind2 = self.best_path[i], self.best_path[i + 1]
            plt.plot(
                [self.nodes_position[ind1, 0], self.nodes_position[ind2, 0]],
                [self.nodes_position[ind1, 1], self.nodes_position[ind2, 1]],
                linewidth=5,
                color="green",
                alpha=0.3,
            )
            ax.annotate(
                str(i),
                xy=(self.nodes_position[ind1, 0], self.nodes_position[ind1, 1]),
                xytext=(
                    1.02 * self.nodes_position[ind1, 0],
                    self.nodes_position[ind1, 1],
                ),
            )

        plt.show()

    def iteration(self, mat_probs: np.ndarray):
        initial_node = 1
        threads = [Any] * self.BATCH_SIZE
        for i in range(self.BATCH_SIZE):
            threads[i] = threading.Thread(
                target=thread_task,
                args=(
                    self.NB_NODES,
                    initial_node,
                    self.listOfSolutions,
                    mat_probs,
                    self.distances,
                ),
            )
            threads[i].start()

        for thread in threads:
            thread.join()

    def simulate(self):
        for i in tqdm(range(self.MAX_ITER)):
            m = (self.phero_matrix**self.alpha) * (self.distances ** (-self.beta))
            mat_probs = np.divide(m, m.sum(axis=1)[:, None])
            self.iteration(mat_probs=mat_probs)
            self.phero_matrix *= 1 - self.evaporation_rate
            new_solutions = self.listOfSolutions[
                i * self.BATCH_SIZE : (i + 1) * self.BATCH_SIZE
            ]
            for (solution, cost), i in itertools.product(
                new_solutions, range(self.NB_NODES - 1)
            ):
                self.phero_matrix[solution[i], solution[i + 1]] += 1 / cost
                self.phero_matrix[solution[i + 1], solution[i]] += 1 / cost

        best_ind = np.argmin([score for _, score in self.listOfSolutions])
        self.best_path = self.listOfSolutions[best_ind][0]

        initial_node = 1
        self.phero_max.append(initial_node)
        while len(self.phero_max) < self.NB_NODES:
            next_node = np.argmax(self.phero_matrix[self.phero_max[-1], :])
            self.phero_matrix[self.phero_max[-1], :] = [0] * self.NB_NODES
            self.phero_matrix[:, self.phero_max[-1]] = [0] * self.NB_NODES
            self.phero_max.append(next_node)
