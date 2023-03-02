import numpy as np
from environments_episodic import EpisodicGraph
from generators import Generator
from propagators import Propagator
from simulators_episodic import EpisodicSimulator

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

states = np.array([
    [0, 0, 0],  # 0 beach beach day
    [2, 1, 0],  # 1 duck beach day
    [1, 2, 0],  # 2 sky beach day
    [2, 3, 0],  # 3 duck beach day
    [3, 4, 1],  # 4 horse mountain day
    [4, 5, 1],  # 5 mountain mountain day
    [1, 6, 1],  # 6 sky mountain day
    [0, 7, 2],  # 0 beach beach day
    [2, 8, 2],  # 1 duck beach day
    [1, 9, 2],  # 2 sky beach day
    [2, 10, 2],  # 3 duck beach day
    [3, 11, 3],  # 4 horse mountain day
    [4, 12, 3],  # 5 mountain mountain day
    [1, 13, 3],  # 6 sky mountain day
], dtype=float)

states[:, 1] = states[:, 1] / max(states[:, 1])

distances = np.array([
    [0, 2, 2, 4, 3],
    [2, 0, 1, 3, 2],
    [2, 1, 0, 1, 3],
    [4, 3, 1, 0, 1],
    [3, 2, 3, 1, 0]
])

distances = 1 - distances / np.max(distances)
# distances = distances ** 0 

k = 5 
n = 0
env = EpisodicGraph(states, distances, k, n)

# Define generator
gen = Generator(env, jump_rate=3)

# Define propagator
prop = Propagator(gen)

# Define initial state
init_state = 0

# Define simulator
sim = EpisodicSimulator(prop, init_state)
sim.sample_sequences(n_step=200, n_samp=5)

seqs = sim.state_seqs
print(seqs)
# Mean by rows

# Create a pandas dataframe with the state sequence
import pandas as pd
df = pd.DataFrame()
for i in range(seqs.shape[0]):
    data = states[seqs[i, :]]

    df_i = pd.DataFrame(data, columns=["word", "time", "episode"])
    
    # Append a column with the sequence number
    df_i["seq"] = i

    df = df.append(df_i, ignore_index=True)


# Plot with seaborn objects interface
import seaborn.objects as so

p = (
    so.Plot(df, x="time", y="word", color="seq")
    .add(so.Path(alpha=0.1), so.Jitter(x=.5, y=.5))
    .scale(
        color = so.Nominal(),
    )
)

p.show()

data = pd.DataFrame(np.abs(seqs[:, 1:] - seqs[:, :-1]).mean(axis=1), columns=["diff"])


p = (
    so.Plot(data, x="diff")
    .add(so.Area(), so.KDE())
)
p.show()


