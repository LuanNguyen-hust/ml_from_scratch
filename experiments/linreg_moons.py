import numpy as np
from scratchml.data import load_moons
from scratchml.losses import mse_loss
from scratchml.metrics import rmse
from scratchml.optim import sgd_step

def closed_form(X, y):
    return np.linalg.pinv(X.T @ X) @ (X.T @ y)

def main():
    X, y = load_moons(n_samples=400, noise=0.2)
    n, d = X.shape
    rng = np.random.default_rng(42)
    W = rng.normal(0, 0.1, size=(d, 1))
    lr = 0.1
    epochs = 1000

    for ep in range(epochs):
        y_pred = X @ W
        loss = mse_loss(y, y_pred)

        grad = (2.0 / n) * (X.T @ (y_pred - y))
        sgd_step([W], [grad], lr)
        
        if ep % 100 == 0:
            print(f"Epoch {ep:4d} | MSE: {loss:.6f}")

    gd_rmse = rmse(y, X @ W)
    W_cf = closed_form(X, y)
    cf_rmse = rmse(y, X @ W_cf)

    print(f"Final RMSE (GD): {gd_rmse:6f}")
    print(f"Closed-form RMSE: {cf_rmse:6f}")
    print(f"Within 5%? {'YES' if gd_rmse <= 1.05 * cf_rmse else 'NO'}")

if __name__ == "__main__":
    main()