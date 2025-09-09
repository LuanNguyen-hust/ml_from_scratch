## Day 1 – NumPy Refresh + Linear Regression (GD)

**Tasks Assigned:**
- Review NumPy syntax (matrix ops, broadcasting).
- Implement linear regression with gradient descent.
- Write functions in `tensor_ops.py`, `losses.py`, `optim.py`, `metrics.py`, `data.py`.
- Run `experiments/linreg_moons.py` and commit results.
- Target: RMSE within 5% of closed-form.

**Tasks Completed:**
- Reviewed NumPy syntax (matrix ops, broadcasting).
- Implemented:
  - `tensor_ops.py`: `matmul`, `add`, `mean`
  - `losses.py`: `mse`
  - `metrics.py`: `rmse`
  - `optim.py`: `sgd_step`
  - `data.py`: `load_moons`
- Ran `experiments/linreg_moons.py` and committed results.
- Verified RMSE within 5% of closed-form.

**Notes:**
- NumPy syntax feels comfortable again.
- Core building blocks for future models are in place.
- May need extra guardrails and explicit data type declarations in ops for future tasks.

## Day 2 – Logistic / Softmax Regression

**Tasks Assigned:**
- Implement **binary cross-entropy (BCE)** and **cross-entropy (CE)** losses in `losses.py`.
- Train **logistic regression** (binary) and **softmax regression** (multiclass) on MNIST.
- Run `experiments/logreg_mnist.py`.
- Target: Achieve **≥88–92% accuracy** on MNIST.

**Tasks Completed (Day 2):**
- Implemented stable Binary Cross-Entropy (BCE) and Cross-Entropy (CE) with logits, plus gradients
- Implemented LogisticRegression and SoftmaxRegression in `scratchml/nn.py`
- Updated `scratchml/data.py` with `filter_digits`, `standardize`, and `split_train_val`
- Built experiment script `experiments/logreg_mnist.py` (binary vs full MNIST)
- Ran experiments and validated against acceptance targets **≥88–92% accuracy**

**Results:**
- Binary Logistic Regression (digits 5 vs 6):
  - Train acc ≈ 98.7%, Val acc ≈ 97.2%, Test acc ≈ 97.9%
  - Test loss inflated (≈3.6) due to a few overconfident misclassifications (expected behavior of BCE)
- Softmax Regression (10-class MNIST):
  - Train acc ≈ 93.6%, Val acc ≈ 91.7%, Test acc ≈ 92.3%

**Notes:**
- BCE is very sensitive to confident wrong predictions → high loss despite high acc.
- Softmax regression is stable and well within target.
- Preprocessing consistency (standardization with train stats) is critical for correct test results.
- Ready to proceed to Day 3: implement MLP layers (Linear, ReLU, Sigmoid, Tanh, Dropout) and manual backprop.

## Day 3 — MLP (Manual Backprop)

**Tasks Assigned:**
- Implement **Linear**, **ReLU**, **Sigmoid**, **Tanh**, **Dropout** layers.
- Build **Sequential container** with manual forward/backward support.
- Train **MLP on MNIST**: architecture [784 → 256 → 128 → 10].
- Run experiments and log metrics.