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
