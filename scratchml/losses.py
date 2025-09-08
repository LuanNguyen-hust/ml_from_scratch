import numpy as np

# safeguard to avoid inputs like NaNs/Infs, used to make range (0,1) -> (_EPS, 1 - _EPS)
_EPS = 1e-8

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# subtract m = max(x) from exponential to avoid overflow -> exponentials clip to range (0,1)
def logsumexp(x, axis=-1, keepdims = False):
    m = np.max(x, axis=axis, keepdims=True)
    s = np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)) + m

    if not keepdims:
        s = np.squeeze(s, axis=axis)

    return s

# stable softmax: exp(z - max) / sum(exp(z - max)), using the same max-shift technique as logsumexp
def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)

    return e/ (e.sum(axis=1, keepdims=True) + _EPS)

# stable log-softmax: z - logsumexp(), avoid np.log(softmax(z)) since computing softmax first risk underflow/overflow
def log_softmax(logits):
    return logits - logsumexp(logits, axis=1, keepdims=True)

def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out

# --- binary cross-entropy (with logits) ---
def bce_with_logits(y_true, logits, reduction="mean"):
    y = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)
    z = np.asarray(logits,  dtype=np.float64).reshape(-1, 1)
    loss = np.maximum(0.0, z) - z * y + np.log1p(np.exp(-np.abs(z)))
    if reduction == "mean": return float(loss.mean())
    if reduction == "sum":  return float(loss.sum())
    return loss.reshape(-1)

def bce_with_logits_grad(y_true, logits):
    y = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)
    z = np.asarray(logits,  dtype=np.float64).reshape(-1, 1)
    return (_sigmoid_stable(z) - y).reshape(logits.shape)

# --- multiclass cross-entropy (with logits) ---
def cross_entropy_with_logits(y_onehot, logits, reduction="mean"):
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y_onehot, dtype=np.float64)
    log_probs = log_softmax(z)                 # your stable version
    loss = -(y * log_probs).sum(axis=1)        # (N,)
    if reduction == "mean": return float(loss.mean())
    if reduction == "sum":  return float(loss.sum())
    return loss

def cross_entropy_with_logits_grad(y_onehot, logits):
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y_onehot, dtype=np.float64)
    return (softmax(z) - y).astype(np.float64)
