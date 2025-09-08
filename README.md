## Progress

**Day 1: Linear Regression with Gradient Descent**
- Implemented core modules: `losses.py`, `metrics.py`, `optim.py`, `data.py`.
- Ran `experiments/linreg_moons.py`.
- Verified that gradient descent RMSE is within 5% of closed-form solution.

**Day 2: Logistic / Softmax Regression**
- Implemented stable Binary Cross-Entropy (BCE) and Cross-Entropy (CE) with logits.
- Added `LogisticRegression` and `SoftmaxRegression` in `scratchml/nn.py`.
- Updated `scratchml/data.py` with `filter_digits`, `standardize`, `split_train_val`.
- Ran `experiments/logreg_mnist.py` for binary (5 vs 6) and full MNIST.
- Achieved ~97.9% test accuracy on binary task and ~92.3% on full MNIST (within target).