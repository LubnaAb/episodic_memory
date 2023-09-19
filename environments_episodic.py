import numpy as np
import networkx as nx
import pandas as pd

from environments import GraphEnv
from utils import row_norm

class EpisodicGraph(GraphEnv):
    def __init__(self, states, semantic_sim, semantic_mds, spatial_sim, spatial_mds, k=0, m=1, n=1, o=1, start=0):
        self.n_state = len(states)
        self.start = start
        self.states = states
        self.semantic_sim = semantic_sim
        # Apply Multi Dimension Scaling using self.distances
        self.semantic_mds = semantic_mds

        self.spatial_sim = spatial_sim
        self.spatial_mds = spatial_mds
        self.k = k
        self.m = m
        self.n = n
        self.o = o
        self._access_matrix()
        super(EpisodicGraph, self).__init__()
        self._state_information()
        self._node_info()
        self.__name__ = "EpisodicGraph"
        self.degree_mat = np.diag(np.sum(self.A, axis=1).reshape(-1))
        self.laplacian = self.degree_mat - self.A
        self.n_edge = np.sum(self.A)
        self.T = row_norm(self.A)
        self.T_torch = None
        self.fname_graph = "figures/episodic_graph.png"

    def _access_matrix(self):
        """
        Sets the adjacency/stochastic matrix for the community graph.
        OUTPUTS: A = adjacency matrix
                 T = stochastic matrix
        """
        self.A = create_access_matrix(self.states, self.semantic_sim, self.spatial_sim, self.k, self.m, self.n, self.o)

    def _node_info(self):
        """
        FUNCTION: Defines node plot positions and communities/bnecks.
        """
        xyc = np.zeros((self.n_state, 3))
        for i, row in self.states.iterrows():
            xyc[i, 0] = row["time"]
            xyc[i, 1] = self.semantic_mds.loc[row["word"]]
            # xyc[i, 2] = row["location"]

        self.xy = xyc[:, :2]
        self.info_state.loc[:, "x"] = xyc[:, 0]
        self.info_state.loc[:, "y"] = xyc[:, 1]
        self.info_state.loc[:, "color"] = xyc[:, 2]
        self.pos = {}
        for state in self.info_state.index:
            self.pos[state] = self.xy[state, :]

    @property
    def node_layout(self):
        """Return node_layout."""
        return self._node_layout

    def _set_graph(self):
        """Defines networkx graph including info_state information"""
        # extract node/edge attributes
        nodesdf = self.info_state.reset_index()
        edgesdf = self.info_transition
        G = nx.from_pandas_edgelist(
            df=edgesdf, source="source", target="target", edge_attr="weight"
        )
        nx.set_node_attributes(G, name="x", values=nodesdf.x.to_dict())
        nx.set_node_attributes(G, name="y", values=nodesdf.y.to_dict())
        self.G = G

    def compute_reward(self, novel_episode: pd.DataFrame):
        """Defines reward function."""

        self.R_state = np.zeros((self.n_state))
        for i in range(self.n_state):
            self.R_state[i] = self._compute_reward(self.states.iloc[i], novel_episode)


    def _compute_reward(self, state, novel_episode: pd.DataFrame):
        """Computes reward for a given state."""
        # loop over rows in novel episode
        reward = 0
        for i, row in novel_episode.iterrows():
            # compute distance between row word and state word
            rew = self.semantic_sim.loc[state["word"]][row["word"]]
            rew += self.spatial_sim.loc[state["location"]][row["location"]]
            # compute reward
            reward += rew

        return reward


def create_access_matrix(states, semantic_sim, spatial_sim, k, m, n, o):
    """
    Creates an access matrix from a generator matrix.
    OUTPUTS: A = adjacency matrix
             T = stochastic matrix
    """
    def compute_v(state_i, state_j):

        semantic_i = state_i["word"]
        semantic_j = state_j["word"]
        temporal_i = state_i["time"]
        temporal_j = state_j["time"]
        spatial_i = state_i["location"]
        spatial_j = state_j["location"]

        episode_i = state_i["episode"]
        episode_j = state_j["episode"]

        semantic_s = semantic_sim.loc[semantic_i][semantic_j]  # similarity
        temporal_s = (1 - abs(temporal_i - temporal_j))
        spatial_s = spatial_sim.loc[spatial_i][spatial_j]

        # Model 1
        # delta = 1 if episode_i == episode_j else 0
        # V = k * delta + dist ** m * (1 - np.abs(temporal_i - temporal_j)) ** n
       
        # Model 2
        delta = 0 if episode_i == episode_j else 1
        V = k ** delta * semantic_s ** m * temporal_s ** n * spatial_s ** o

        return V

    O = np.zeros((len(states), len(states)))

    for i, state_i in states.iterrows():
        for j, state_j in states.iterrows():
            if i != j:
                O[i, j] = compute_v(state_i, state_j)
    for i in range(len(states)):
        O[i, i] = -O[i, ].sum()

    A = np.zeros((len(states), len(states)))
    n = -np.diag(O)

    for i in range(len(states)):
        for j in range(len(states)):
            if i != j:
                A[i, j] = O[i, j] / n[i]
            else:
                A[i, j] = 0
    
    return A

    

