def sgd_step(params, grads, lr=0.01):
    for p,g in zip(params, grads):
        p[...] = p - lr * g