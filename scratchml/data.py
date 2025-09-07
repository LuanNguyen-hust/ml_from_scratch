from sklearn.datasets import make_moons
import numpy as np
import os
from urllib.request import urlretrieve

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
DEFAULT_CACHE = os.path.expanduser("~/.cache/ml_from_scratch/mnist.npz")

# to modularize loading dataset, allow different optimization to have same preprocess
def load_moons(n_samples = 200, noise = 0.1, seed = 42, add_bias = True):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    y = y.reshape(-1, 1).astype(float)
    
    if add_bias:
        X = np.hstack([X, np.ones((X.shape[0], 1))])

    return X, y

# make sure the download path is correct
def _ensure_dir(path: str)->None:
    d = os.path.dirname(path)

    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# to avoid redownloading mnist.npz if already exist
def _ensure_mnist_npz(cache_path: str = DEFAULT_CACHE, url: str = MNIST_URL) -> str:
    if os.path.exists(cache_path):
        return cache_path
    _ensure_dir(cache_path)
    print(f"[mnist]downloading to {cache_path} ...")
    urlretrieve(url, cache_path)
    return cache_path

def _one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    labels = labels.astype(np.int64)
    oh = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    oh[np.arange(labels.shape[0]), labels] = 1.0
    return oh

