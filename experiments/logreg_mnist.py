# experiments/logreg_mnist.py
import numpy as np
from scratchml.nn import LogisticRegression, SoftmaxRegression
from scratchml.losses import bce_with_logits, cross_entropy_with_logits
from scratchml.data import load_mnist, filter_digits, standardize, split_train_val

def acc(y_true, y_pred): return (y_true == y_pred).mean()

def run_binary():
    print("== Binary Logistic: digits (5 vs 6) ==")
    # Get training set
    Xtr, ytr = load_mnist(split="train", flatten=True, normalize=False)

    # Get test set
    Xte, yte = load_mnist(split="test", flatten=True, normalize=False)

    Xtr_b, ytr_b = filter_digits(Xtr, ytr, digits=(5, 6))
    Xte_b, yte_b = filter_digits(Xte, yte, digits=(5, 6))

    Xtr_b, mu, sd = standardize(Xtr_b)
    Xte_b = ((Xte_b/255.0) - mu) / sd

    Xtr_, ytr_, Xval, yval = split_train_val(Xtr_b, ytr_b, 0.1, seed=42)
    model = LogisticRegression(in_dim=Xtr_.shape[1], lr=0.2, weight_decay=1e-4, seed=42)
    model.fit(Xtr_, ytr_, epochs=15, batch_size=512, X_val=Xval, y_val=yval, verbose=True)

    te_loss = bce_with_logits(yte_b, model.logits(Xte_b))
    te_acc  = acc(yte_b, model.predict(Xte_b))
    print(f"[Binary] test loss {te_loss:.4f} acc {te_acc*100:.2f}%")

def run_softmax():
    print("\n== Softmax Regression: MNIST (10 classes) ==")
    # Get training set
    Xtr, ytr = load_mnist(split="train", flatten=True, normalize=False)

    # Get test set
    Xte, yte = load_mnist(split="test", flatten=True, normalize=False)

    Xtr, mu, sd = standardize(Xtr)
    Xte = ((Xte/255.0) - mu) / sd

    Xtr_, ytr_, Xval, yval = split_train_val(Xtr, ytr, 0.1, seed=42)
    model = SoftmaxRegression(in_dim=Xtr_.shape[1], num_classes=10, lr=0.3, weight_decay=5e-4, seed=42)
    model.fit(Xtr_, ytr_, epochs=25, batch_size=512, X_val=Xval, y_val=yval, verbose=True)

    y1h = np.eye(10)[yte]
    te_loss = cross_entropy_with_logits(y1h, model.logits(Xte))
    te_acc  = acc(yte, model.predict(Xte))
    print(f"[Softmax] test loss {te_loss:.4f} acc {te_acc*100:.2f}%")

if __name__ == "__main__":
    run_binary()
    run_softmax()





