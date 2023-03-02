from simulators import Simulator
from visualization import add_jitter, figsize
import seaborn.objects as so
import pandas as pd
import numpy as np

class EpisodicSimulator(Simulator):
    def plot_trajectory(self, samp=0):
        coords = np.array(self._retrieve_state(samp=samp, step=None, coords=True))
        print(coords.dtype)

        coords = pd.DataFrame(coords, columns = ["x", "y"])
        print(coords)

        return (
            so.Plot(self.ENV.info_state, x="x", y="y", color="color")
            .add(so.Dot())
            .scale(
                color = so.Nominal(),
                x = so.Temporal(),
            )
            .add(so.Line(coords, color="black"))
        )
