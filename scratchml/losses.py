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
