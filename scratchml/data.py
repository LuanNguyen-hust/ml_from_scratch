from sklearn.datasets import make_moons
import numpy as np

def load_moons(n_samples = 200, noise = 0.1, seed = 42, add_bias = True):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    y = y.reshape(-1, 1).astype(float)
    
    if add_bias:
        X = np.hstack([X, np.ones((X.shape[0], 1))])

    return X, y
