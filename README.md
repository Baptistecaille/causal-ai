# Causal AI Examples

This repository contains examples of Causal AI implementation using Python.

## Examples

### 1. Structure Learning with CausalNex
`causalnex_example.py` demonstrates how to learn a Bayesian Network structure from data using the NOTEARS algorithm.
- Generates synthetic data with known causal links ($A \to B \to C$).
- Recovers the adjacency matrix (causality matrix).

### 2. Causal Inference with DoWhy
`dowhy_example.py` demonstrates the 4-step causal inference workflow:
1.  **Model**: Define causal assumptions.
2.  **Identify**: Identify the causal effect to estimate.
3.  **Estimate**: Calculate the effect (e.g., using Linear Regression).
4.  **Refute**: Test the robustness of the estimate.

## Setup

### Prerequisites
- **Python**: 3.10 (Strict requirement, pinned in `.python-version`)
- **Package Manager**: [uv](https://github.com/astral-sh/uv) (Recommended)

### Installation

1.  Clone the repository and enter the directory.
2.  Install dependencies using `uv`:

```bash
uv sync
```

Alternatively, using pip (ensure you are using Python 3.10):
```bash
pip install -r requirements.txt # if generated
# Or install directly:
pip install causalnex dowhy pandas numpy networkx scikit-learn "scipy>=1.10"
```

## Usage

### Run CausalNex Example
```bash
uv run python causalnex_example.py
```

### Run DoWhy Example
```bash
uv run python dowhy_example.py
```
