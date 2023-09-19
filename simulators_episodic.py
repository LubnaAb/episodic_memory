from simulators import Simulator
from visualization import add_jitter, figsize
import seaborn.objects as so
import pandas as pd
import numpy as np
from autocorrelation import estimate_episodic_acf, estimate_episodic_acf_v2

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
    
    def estimate_cf(self, axis=None):
        samps_state = self.ENV.states[self.state_seqs]
        samps_state[:, :, 0] = self.ENV.semantic_mds[samps_state[:, :, 0].astype(int)]
        samps_state[:, :, 2] = self.ENV.spatial_mds[samps_state[:, :, 2].astype(int)]
        samps_state = samps_state[:, :, 0:3]
        self.acf_mean, self.acf_sem = estimate_episodic_acf_v2(samps_state, axis=axis)

