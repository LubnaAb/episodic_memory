import numpy as np
from environments_episodic import EpisodicGraph
from generators import Generator
from propagators import Propagator
from simulators import Simulator
from explorers import Explorer

words = {
    0: "beach",
    1: "sky",
    2: "duck",
    3: "horse" ,
    4: "mountain",
}

episodes = {
    0: "beach day",
    1: "mountain day",
}

states = [
    (0, 0, 0),
    (2, 1, 0),
    (1, 2, 0),
    (0, 3, 0),
    (3, 4, 1),
    (4, 5, 1),
    (1, 6, 1)
]

distances = 4 - np.array([
    [0, 2, 2, 4, 3],
    [2, 0, 1, 3, 2],
    [2, 1, 0, 1, 3],
    [4, 3, 1, 0, 1],
    [3, 2, 3, 1, 0]
])

k = 200
n = 0
env = EpisodicGraph(states, distances, k, n)

# Define generator
gen = Generator(env)

# Define propagator
prop = Propagator(gen)

# Define initial state
init_state = 1

# Define simulator
sim = Simulator(prop, init_state)
sim.sample_sequences(n_step=20, n_samp=4)

print(sim.state_seqs)
