import numpy as np
import networkx as nx

from environments import GraphEnv
from utils import row_norm

class EpisodicGraph(GraphEnv):
    def __init__(self, states, distances, semantic_mds, k=0, m=1, n=1, start=0):
        self.n_state = len(states)
        self.start = start
        self.states = states
        self.distances = distances
        # Apply Multi Dimension Scaling using self.distances
        self.semantic_mds = semantic_mds
        self.k = k
        self.m = m
        self.n = n
        self._access_matrix()
        super(EpisodicGraph, self).__init__()
        self._state_information()
        self._node_info()
        self.__name__ = "EpisodicGraph"
        self.degree_mat = np.diag(np.sum(self.A, axis=1).reshape(-1))
        self.laplacian = self.degree_mat - self.A
        self.n_edge = np.sum(self.A)
        self.T = row_norm(self.A)
        self.fname_graph = "figures/episodic_graph.png"

    def _access_matrix(self):
        """
        Sets the adjacency/stochastic matrix for the community graph.
        OUTPUTS: A = adjacency matrix
                 T = stochastic matrix
        """
        self.A = create_access_matrix(self.states, self.distances, self.k, self.m, self.n)

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



def create_access_matrix(states, similarities, k, m, n):
    """
    Creates an access matrix from a generator matrix.
    OUTPUTS: A = adjacency matrix
             T = stochastic matrix
    """
    def compute_v(state_i, state_j):

        semantic_i = state_i["word"]
        semantic_j = state_j["word"]
        time_i = state_i["time"]
        time_j = state_j["time"]
        episode_i = state_i["location"]
        episode_j = state_j["location"]

        dist = similarities.loc[semantic_i][semantic_j]  # similarity

        # Model 1
        # delta = 1 if episode_i == episode_j else 0
        # V = k * delta + dist ** m * (1 - np.abs(time_i - time_j)) ** n
       
        # Model 2
        delta = 0 if episode_i == episode_j else 1
        V = k ** delta * dist ** m * (1 - np.abs(time_i - time_j)) ** n

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

    

