from sklearn.datasets import make_moons
import numpy as np
import os
from urllib.request import urlretrieve

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
DEFAULT_CACHE = os.path.expanduser("~/.cache/ml_from_scratch/mnist.npz")

# ---data loader---
# to modularize loading dataset, allow different optimization to have same preprocess
def load_moons(n_samples = 200, noise = 0.1, seed = 42, add_bias = True):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    y = y.reshape(-1, 1).astype(float)
    
    if add_bias:
        X = np.hstack([X, np.ones((X.shape[0], 1))])

    return X, y


def load_mnist(split: str = "train",
               flatten: bool = True,
               normalize: bool = True,
               one_hot: bool = False,
               cache_path: str = DEFAULT_CACHE):
    """
    load MNIST from a cached `mnist.npz`, downloading once if needed.

    returns:
        X: np.ndarray float32, shape (N, 784) if flattened else (N, 28, 28).
        y: np.ndarray; int64 (N,) if not one-hot; float32 (N, 10) if one-hot.
    """
    path = _ensure_mnist_npz(cache_path)

    # load arrays
    with np.load(path) as f:
        x_train, y_train = f["x_train"], f["y_train"]  # (60000, 28, 28), (60000,)
        x_test,  y_test  = f["x_test"],  f["y_test"]   # (10000, 28, 28), (10000,)

    # pick split
    if split == "train":
        X, y = x_train, y_train
    elif split == "test":
        X, y = x_test, y_test
    else:
        raise ValueError("split must be 'train' or 'test'")

    # flatten if requested
    if flatten:
        X = X.reshape(X.shape[0], -1)  # (N, 784)

    # convert to float32 then (optionally) normalize
    X = X.astype(np.float32, copy=False)
    if normalize:
        X /= 255.0  # safe because float32 now

    # labels: int64 or one-hot float32
    y = y.astype(np.int64, copy=False)
    if one_hot:
        y = _one_hot(y, num_classes=10)  # float32 (N, 10)

    return X, y

# ---loader helpers---

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


# ---helper functions---

# create one hot matrix to support CE computation
def _one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    labels = labels.astype(np.int64)
    oh = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    oh[np.arange(labels.shape[0]), labels] = 1.0
    return oh

# creating binary subset
def filter_digits(X: np.ndarray, y: np.ndarray, digits=(0, 1)):
    """
    Keep only samples from `digits` and map labels to {0,1} by the order in `digits`.
      - digits[0] -> 0
      - digits[1] -> 1
    Works if y is (N,) ints or (N,10) one-hot.

    Returns:
        X_sub: filtered inputs
        y_bin: int64 labels in {0,1} with shape (M,)
    """
    # Convert one-hot labels to class indices if needed
    if y.ndim == 2:
        y_idx = np.argmax(y, axis=1)
    else:
        y_idx = y

    # Mask rows that are either of the two requested digits
    mask = np.isin(y_idx, digits)
    X_sub = X[mask]
    y_sub = y_idx[mask]

    # Deterministic mapping based on the order in `digits`
    y_bin = (y_sub == digits[1]).astype(np.int64)
    return X_sub, y_bin


def standardize(X):
    X01 = X if X.max() <= 1.0 + 1e-6 else X / 255.0
    mu  = X01.mean(axis=0, keepdims=True)
    sd  = X01.std(axis=0, keepdims=True) + 1e-6
    return (X01 - mu) / sd, mu, sd

# splite the data set into train and test set
def split_train_val(X, y, val_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    idx = np.arange(N); rng.shuffle(idx)
    nval = int(N * val_ratio)
    vidx, tidx = idx[:nval], idx[nval:]
    return X[tidx], y[tidx], X[vidx], y[vidx]
