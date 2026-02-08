# Copilot Instructions — Machine Learning from Scratch

## Project overview

Educational repository implementing ML/DL algorithms **using only Python and NumPy** (no PyTorch/TensorFlow for model logic). It contains:

- **`Deep Learning/src/`** — A custom DL framework with autograd, layers, optimizers, losses, and training loops.
- **`Deep Learning/{Computer Vision,Generative AI,Regression,Time Series Forecasting}/`** — Notebooks using the framework.
- **`Machine Learning/`** — Classic ML algorithms (Logistic Regression, KNN, SVM, PCA, etc.) each in standalone folders.

## DL Framework architecture

The core class hierarchy is `Module` → `SingleOutputModule` / `MultiOutputModule` → concrete layers and `Architecture` → `Sequential` → `AutoRegressive`. `Tensor` wraps `np.ndarray` with tape-based autograd: forward ops push intermediate data to a global tape, `backward()` builds a topological order via `_prev` links and propagates gradients.

- **Lazy initialization**: layers implement `_lazy_init(x)` to create weight tensors on first forward call. Weight `Tensor` objects with `requires_grad=True, is_parameter=True` are auto-registered via `Module.__setattr__`.
- **SingleOutputModule vs MultiOutputModule**: differ only in type hints (`Tensor` vs `Tuple[Tensor, ...]`), no runtime difference.
- **Training**: `Sequential.fit(TrainingArguments)` handles the full loop — batching, forward, loss, backward, optimizer step, gradient accumulation, validation, callbacks.

## Code conventions

- **Section comments**: every class uses `### Magic methods ###`, `### Properties ###`, `### Public methods ###`, `### Protected methods ###` to separate sections.
- **Docstrings**: Google-style on all public/protected methods with `Parameters:`, `Returns:`, `Raises:` blocks.
- **Naming**: classes `PascalCase`, methods/variables `snake_case`, protected methods prefixed with `_` (e.g., `_forward`, `_lazy_init`).
- **Type hints**: extensive use of `typing` (`Optional`, `Union`, `TYPE_CHECKING`). `Tensor` uses `__slots__` for performance.
- **Imports**: relative imports within the `src/` package (e.g., `from ..core import Tensor`); notebooks add parent to path with `sys.path.append(str(Path().resolve().parent))`.

## Adding new components

**New layer** — Subclass `SingleOutputModule`, implement `_forward(self, x: Tensor) -> Tensor` and `_lazy_init(self, x: Tensor)`. Register weights as `Tensor(..., requires_grad=True, is_parameter=True)`. See `Dense` in `Deep Learning/src/layers/dense.py` as the canonical example.

**New activation** — Subclass `Activation`, implement `__call__(self, x: Tensor) -> Tensor` using Tensor ops (e.g., `x.relu()`). See `Deep Learning/src/activations/relu.py`.

**New loss function** — Subclass `LossFn`, implement `__call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor`. See `Deep Learning/src/loss_functions/mean_square_error.py`.

**New kernel op** — Add forward + backward functions in `Deep Learning/src/core/functional/kernel/`, wire them through `tensor_unary_op` / `tensor_binary_op` in `Tensor`.

**New ML algorithm** — Create folder `Machine Learning/.../AlgorithmName/` with `src/algorithm.py` (class with `fit(x, y)` + `predict(x)` sklearn-like API using only NumPy), `train.ipynb`, and `README.md`.

## Testing

Tests use `unittest` and validate custom implementations against **PyTorch as reference**: copy weights to equivalent PyTorch layers, compare forward outputs and backward gradients with `np.allclose(atol=1e-5)`. Test base class: `Deep Learning/src/tests/base.py`.

Run all tests:
```bash
python -m unittest discover -s "Deep Learning/src/tests"
```

Run a single test file:
```bash
python "Deep Learning/src/tests/layers/test_dense.py"
```

## Notebook workflow

Notebooks follow a Keras-like pattern:
1. Build model via `Sequential(modules=[...])` or custom `Architecture`
2. Wrap data in `LabeledData(input={'x': tensor}, target=tensor)`
3. Configure `TrainingArguments(train_data, optimizer, loss_fn, ...)`
4. Call `model.fit(args)` → returns `history` dict
5. Inference under `no_grad()` context with `model.eval()`
6. Save/load with `model.save(path)` / `Sequential.load(path)`

## Key files reference

| Purpose | Path |
|---------|------|
| Tensor + autograd | `Deep Learning/src/core/tensor.py` |
| Module base class | `Deep Learning/src/core/module.py` |
| Training loop | `Deep Learning/src/architectures/sequential/sequential.py` |
| Autoregressive generation | `Deep Learning/src/architectures/auto_regressive/auto_regressive.py` |
| Training data structures | `Deep Learning/src/models/` |
| Autograd tape | `Deep Learning/src/core/functional/tape.py` |
| Kernel ops (forward+backward) | `Deep Learning/src/core/functional/kernel/` |
| Numerical stability constant | `Deep Learning/src/core/utils/constants.py` (`EPSILON = 1e-7`) |
