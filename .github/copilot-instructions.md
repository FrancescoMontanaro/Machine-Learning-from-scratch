# Copilot Instructions — Machine Learning from Scratch

## Project overview

Educational repository implementing ML/DL algorithms **using only Python and NumPy** (no PyTorch/TensorFlow for model logic). It contains:

- **`Deep Learning/src/`** — A custom DL framework with autograd, layers, optimizers, losses, metrics, and training loops.
- **`Deep Learning/{Computer Vision,Generative AI,Regression,Time Series Forecasting}/`** — Notebooks using the framework.
- **`Machine Learning/`** — Classic ML algorithms organized by category: `Supervised Learning/{Classification,Regression,General Algorithms}/`, `Unsupervised Learning/{Clustering,Dimensionality Reduction}/`, `Reinforcement Learning/`. Each algorithm has its own folder with `src/`, `train.ipynb`, and `README.md`.

## DL Framework architecture

The core class hierarchy is `Module` → concrete layers and `Architecture` → `Sequential` → `AutoRegressive`. `Tensor` wraps `np.ndarray` with tape-based autograd: forward ops push intermediate data to a global dict-based tape (keyed by unique int), `backward()` builds a topological order via `_prev` links and propagates gradients.

- **Unified output**: all modules return `ModuleOutput` (defined in `Deep Learning/src/core/module_output.py`), which wraps a primary tensor (`output`) plus optional named auxiliary tensors (`**aux`, e.g., `mu`, `logvar` for VAE). Auxiliary tensors are accessible as attributes and are forwarded to the loss function via `**aux`.
- **Core containers**: `ModuleList` (`Deep Learning/src/core/modules_list.py`) manages ordered lists of sub-modules with proper parameter registration. `TensorsList` (`Deep Learning/src/core/tensors_list.py`) manages lists of `Tensor` objects with automatic parameter tracking.
- **Lazy initialization**: layers implement `_lazy_init(x)` to create weight tensors on first forward call. Weight `Tensor` objects with `requires_grad=True, is_parameter=True` are auto-registered via `Module.__setattr__`.
- **Training**: `Sequential.fit(TrainingArguments)` handles the full loop — batching, forward, loss, backward, optimizer step, gradient accumulation, validation, callbacks.

### Architectures

| Architecture | Path | Description |
|---|---|---|
| `Sequential` | `Deep Learning/src/architectures/sequential/` | Standard feed-forward training loop |
| `AutoRegressive` | `Deep Learning/src/architectures/auto_regressive/autoregressive.py` | Extends Sequential with autoregressive generation |
| `AlexNet` | `Deep Learning/src/architectures/alexnet/` | AlexNet image classification architecture |
| `VAE` | `Deep Learning/src/architectures/auto_encoder/` | Variational Autoencoder (with custom loss in `loss_fn.py`) |
| `Transformer` | `Deep Learning/src/architectures/transformer/` | Transformer with self/cross/latent attention, MLP, MoE, encoder/decoder blocks, tokenizer, and config variants (vanilla, DeepSeek) |

## Code conventions

- **Section comments**: every class uses `### Magic methods ###`, `### Properties ###`, `### Public methods ###`, `### Protected methods ###` to separate sections.
- **Docstrings**: Google-style on all public/protected methods with `Parameters:`, `Returns:`, `Raises:` blocks.
- **Naming**: classes `PascalCase`, methods/variables `snake_case`, protected methods prefixed with `_` (e.g., `_forward`, `_lazy_init`).
- **Type hints**: extensive use of `typing` (`Optional`, `Union`, `TYPE_CHECKING`). `Tensor` uses `__slots__` for performance.
- **Imports**: relative imports within the `src/` package (e.g., `from ..core import Tensor`); notebooks add parent to path with `sys.path.append(str(Path().resolve().parent))`.

## Adding new components

**New layer** — Subclass `Module`, implement `_forward(self, x: Tensor) -> Tensor` and `_lazy_init(self, x: Tensor)`. Register weights as `Tensor(..., requires_grad=True, is_parameter=True)`. See `Dense` in `Deep Learning/src/layers/dense.py` as the canonical example.

**New activation** — Subclass `Activation`, implement `__call__(self, x: Tensor) -> Tensor` using Tensor ops (e.g., `x.relu()`). See `Deep Learning/src/activations/relu.py`.

**New loss function** — Subclass `LossFn`, implement `__call__(self, y_true: Tensor, y_pred: Tensor, **aux: Tensor) -> Tensor`. The `**aux` parameter receives auxiliary tensors from `ModuleOutput` (e.g., `mu`, `logvar` for VAE). See `Deep Learning/src/loss_functions/mean_square_error.py`.

**New kernel op** — Add forward + backward functions in `Deep Learning/src/core/functional/kernel/`, wire them through the helpers in `Deep Learning/src/core/functional/base.py` (`tensor_unary_op`, `tensor_binary_op`, `tensor_nary_op`, `tensor_unary_op_multiple_outputs`, `tensor_unary_op_binary_output`).

**New ML algorithm** — Create folder `Machine Learning/{Supervised Learning|Unsupervised Learning|Reinforcement Learning}/.../AlgorithmName/` with `src/algorithm.py` (class with `fit(x, y)` + `predict(x)` sklearn-like API using only NumPy), `train.ipynb`, and `README.md`.

## Testing

Tests use `unittest` and validate custom implementations against **PyTorch as reference**: copy weights to equivalent PyTorch layers, compare forward outputs and backward gradients with `np.allclose(atol=1e-5)`. Test base class: `Deep Learning/src/tests/base.py`.

Test subdirectories: `activations/`, `layers/`, `kernels/`, `loss_functions/`, `optimizers/`.

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
5. Inference under `context_manager.no_grad()` context with `model.eval()`
6. Save/load with `model.save(path)` / `Sequential.load(path)`

## Key files reference

| Purpose | Path |
|---------|------|
| Tensor + autograd | `Deep Learning/src/core/tensor.py` |
| Module base class | `Deep Learning/src/core/module.py` |
| Module output wrapper | `Deep Learning/src/core/module_output.py` |
| Module list container | `Deep Learning/src/core/modules_list.py` |
| Tensors list container | `Deep Learning/src/core/tensors_list.py` |
| Training loop | `Deep Learning/src/architectures/sequential/sequential.py` |
| Autoregressive generation | `Deep Learning/src/architectures/auto_regressive/autoregressive.py` |
| Transformer architecture | `Deep Learning/src/architectures/transformer/` |
| VAE architecture | `Deep Learning/src/architectures/auto_encoder/` |
| Training data structures | `Deep Learning/src/models/training_arguments.py` |
| Autograd tape (dict-based) | `Deep Learning/src/core/functional/tape.py` |
| Op wiring helpers | `Deep Learning/src/core/functional/base.py` |
| Kernel ops (forward+backward) | `Deep Learning/src/core/functional/kernel/` |
| Optimizers (Adam, SGD) | `Deep Learning/src/optimizers/` |
| Metrics (accuracy, F1, etc.) | `Deep Learning/src/metrics/` |
| Callbacks (EarlyStopping) | `Deep Learning/src/callbacks/` |
| Numerical stability constant | `Deep Learning/src/core/utils/constants.py` (`EPSILON = 1e-7`) |
| No-grad context manager | `Deep Learning/src/core/utils/context_manager.py` |