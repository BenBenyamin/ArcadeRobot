import pickle
import numpy as np

class LatencySampler:
    """
    A utility class for sampling realistic latency values with controlled noise.

    This class loads a sequence of real-world latency measurements from a pickle
    file (in seconds), converts them to frame units using a specified FPS, and
    provides an interface for drawing latency samples that preserve the empirical
    distribution while adding small Gaussian noise.

    Parameters
    ----------
    pickle_filename : str
        Path to a pickle file containing an iterable of latency values in seconds.
    fps : int
        Frame rate used to convert latency from seconds to frame units.
    noise_std : float, optional (default=0.01)
        Standard deviation of Gaussian noise added to each latency sample.
    """

    def __init__(self, pickle_filename: str, fps: int, noise_std: float = 0.01):
        with open(pickle_filename, "rb") as f:
            lat = pickle.load(f)
        
        self.lat = np.array(lat) * fps   # convert seconds → frames
        self.noise_std = noise_std
        self.min_lat = np.min(self.lat)

    def __getitem__(self, idx):
        """
        Returns the latency at a given index with added noise and clamping.
        """
        # Gaussian integer noise
        noise = np.random.normal(0, self.noise_std)

        # Clamp so latency never goes below observed minimum
        while self.lat[idx] + noise < self.min_lat:
            noise = np.random.normal(0, self.noise_std)
        
        noisy_lat = self.lat[idx] + noise
        # Round to integer frames
        
        return int(round(noisy_lat))


    def sample(self):
        """Return a random noisy latency sample."""
        idx = np.random.randint(0, len(self.lat))
        return self[idx]


