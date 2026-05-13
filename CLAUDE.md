# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Install in editable mode with all dev/test dependencies:

```bash
pip install -e ".[dev,full]"
pre-commit install
```

Optional accelerator packages (needed for some features):

```bash
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

## Commands

```bash
# Run all tests
pytest

# Run a single test file
pytest test/utils/test_convert.py

# Run with coverage
pytest --cov

# Run slow/extended tests
FULL_TEST=1 pytest --cov

# Lint (PEP8)
flake8 .

# Format code
yapf --in-place <file>

# Sort imports
isort <file>

# Run all pre-commit hooks manually
pre-commit run --all-files

# Build docs
cd docs && make html
```

## Architecture

The package lives entirely under `torch_geometric/`. Key submodules:

- **`data/`** — Core data structures. `Data` (homogeneous graph), `HeteroData` (heterogeneous), `Batch` (mini-batches), `Dataset`/`InMemoryDataset` base classes. `FeatureStore`/`GraphStore` provide the remote backend protocol.
- **`nn/`** — Neural network building blocks:
  - `nn/conv/` — All GNN convolution layers. Every layer subclasses `MessagePassing` (defined in `nn/conv/message_passing.py`), which implements the gather→message→aggregate→update pattern via code inspection and Jinja2 templates.
  - `nn/aggr/` — Standalone aggregation modules (sum, mean, LSTM, attention, etc.) usable independently of conv layers.
  - `nn/models/` — End-to-end model architectures (GCN, GAT, GraphSAGE, DimeNet, etc.).
  - `nn/pool/` — Graph pooling operators.
  - `nn/norm/` — Normalization layers (BatchNorm, LayerNorm, etc.).
- **`datasets/`** — 100+ dataset loaders. Each inherits `InMemoryDataset` or `Dataset`.
- **`transforms/`** — Pre-processing transforms applied to `Data` objects. Each subclasses `BaseTransform`.
- **`loader/`** — DataLoader variants for mini-batch training (`NeighborLoader`, `ClusterLoader`, etc.).
- **`sampler/`** — Graph sampling backends used by loaders.
- **`utils/`** — Graph utility functions (scatter, subgraph, coalesce, etc.). Functions are prefixed with `_` in the module filename but exported without prefix in `__init__.py`.
- **`explain/`** — GNN explainability (GNNExplainer, PGExplainer, etc.).
- **`transforms/`** — Data transforms applied during preprocessing.
- **`typing.py`** — Shared type aliases (`Adj`, `OptTensor`, `NodeType`, `EdgeType`, `SparseTensor`) and runtime capability flags (`WITH_PYG_LIB`, `WITH_PT28`, etc.).

## Key Patterns

**MessagePassing**: Implement `message()`, `aggregate()`, and `update()` (or just `message()` for simple cases). The base class uses `Inspector` to introspect method signatures and automatically routes `propagate()` arguments to the right methods. Jinja2 templates in `collect.jinja` / `edge_updater.jinja` generate the collect step at import time.

**Optional dependencies**: Capability flags in `torch_geometric/typing.py` (e.g., `WITH_PYG_LIB`, `WITH_PT28`, `torch_geometric.typing.SparseTensor`) gate features at runtime. Tests that require optional packages use `pytest.importorskip` or `@pytest.mark.skipif`.

**Heterogeneous graphs**: `HeteroData` stores node/edge features keyed by `NodeType` (str) and `EdgeType` (3-tuple). Conv layers can be converted to heterogeneous via `to_hetero()` or the `ToHeteroTransformer`.

**`Adj` type**: Most conv `forward()` signatures accept `edge_index: Adj`, which is `Union[Tensor, SparseTensor]`. The `SparseTensor` path (from `torch-sparse`) is an alternate sparse representation; always handle both or document which is required.

**Test layout**: Mirrors `torch_geometric/` exactly under `test/`. `test/conftest.py` provides shared fixtures and `load_dataset()` helper.

## Linting/Formatting Rules

- Line length: 80 characters (flake8 + ruff)
- Style: PEP8 via `yapf` with `based_on_style = pep8`
- Import order: `isort` with `multi_line_output = 3`
- Docstrings: Google style via `ruff` pydocstyle
- No commits directly to `master` (enforced by pre-commit hook)
