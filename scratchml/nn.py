import numpy as np
from .losses import (
    bce_with_logits, bce_with_logits_grad,
    cross_entropy_with_logits, cross_entropy_with_logits_grad,
    softmax,
)

# =========================
#   Classifiers (Day 2)
# =========================

class LogisticRegression:
    def __init__(self, in_dim, lr=0.2, weight_decay=1e-4, seed=42):
        self.rng = np.random.default_rng(seed)
        self.W = self.rng.normal(0, 0.01, size=(in_dim, 1)).astype(np.float64)
        self.b = np.zeros((1,), dtype=np.float64)
        self.lr = lr
        self.wd = weight_decay

    def logits(self, X):            # (N,1)
        return X @ self.W + self.b

    def predict_proba(self, X):     # (N,1), stable sigmoid
        z = self.logits(X)
        out = np.empty_like(z, dtype=np.float64)
        pos = z >= 0; neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ex = np.exp(z[neg]); out[neg] = ex / (1.0 + ex)
        return out

    def predict(self, X):           # (N,)
        return (self.predict_proba(X) >= 0.5).astype(np.int64).ravel()

    def fit(self, X, y, epochs=15, batch_size=512, X_val=None, y_val=None, verbose=True):
        N, D = X.shape
        idx = np.arange(N)
        for ep in range(1, epochs+1):
            np.random.shuffle(idx)
            Xs, ys = X[idx], y[idx]
            for i in range(0, N, batch_size):
                xb = Xs[i:i+batch_size]; yb = ys[i:i+batch_size]
                z  = self.logits(xb)
                gz = bce_with_logits_grad(yb, z)       # (B,1)
                gW = xb.T @ gz / xb.shape[0]           # (D,1)
                gb = gz.mean(axis=0)                   # (1,)
                if self.wd > 0: gW += self.wd * self.W
                self.W -= self.lr * gW
                self.b -= self.lr * gb
            if verbose:
                loss = bce_with_logits(y, self.logits(X))
                acc  = (self.predict(X) == y).mean()
                msg = f"[LogReg] ep {ep:02d} | loss {loss:.4f} acc {acc*100:.2f}%"
                if X_val is not None:
                    vloss = bce_with_logits(y_val, self.logits(X_val))
                    vacc  = (self.predict(X_val) == y_val).mean()
                    msg += f" | val {vloss:.4f} {vacc*100:.2f}%"
                print(msg)

class SoftmaxRegression:
    def __init__(self, in_dim, num_classes, lr=0.3, weight_decay=5e-4, seed=42):
        self.rng = np.random.default_rng(seed)
        self.W = self.rng.normal(0, 0.01, size=(in_dim, num_classes)).astype(np.float64)
        self.b = np.zeros((num_classes,), dtype=np.float64)
        self.lr = lr
        self.wd = weight_decay

    def logits(self, X):            # (N,C)
        return X @ self.W + self.b

    def predict_proba(self, X):     # (N,C)
        return softmax(self.logits(X))

    def predict(self, X):           # (N,)
        return np.argmax(self.predict_proba(X), axis=1)

    def fit(self, X, y, epochs=25, batch_size=512, X_val=None, y_val=None, verbose=True):
        N, D = X.shape
        C = self.W.shape[1]
        y1h_full = np.eye(C, dtype=np.float64)
        idx = np.arange(N)
        for ep in range(1, epochs+1):
            np.random.shuffle(idx)
            Xs, ys = X[idx], y[idx]
            for i in range(0, N, batch_size):
                xb = Xs[i:i+batch_size]; yb = ys[i:i+batch_size]
                y1h = y1h_full[yb]                    # (B,C)
                z   = self.logits(xb)                 # (B,C)
                gz  = cross_entropy_with_logits_grad(y1h, z)
                gW  = xb.T @ gz / xb.shape[0]         # (D,C)
                gb  = gz.mean(axis=0)                 # (C,)
                if self.wd > 0: gW += self.wd * self.W
                self.W -= self.lr * gW
                self.b -= self.lr * gb
            if verbose:
                loss = cross_entropy_with_logits(y1h_full[y], self.logits(X))
                acc  = (self.predict(X) == y).mean()
                msg = f"[Softmax] ep {ep:02d} | loss {loss:.4f} acc {acc*100:.2f}%"
                if X_val is not None:
                    vloss = cross_entropy_with_logits(y1h_full[y_val], self.logits(X_val))
                    vacc  = (self.predict(X_val) == y_val).mean()
                    msg += f" | val {vloss:.4f} {vacc*100:.2f}%"
                print(msg)