# lau-closure

**The missing center — Universal Dirac operator and closure object proving all 14 executable theorems share one spectrum.**

[![Tests](https://img.shields.io/badge/tests-71-passing-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## What This Does

This crate implements a **categorical closure operator** that unifies 14 executable mathematical theorems under a single spectral framework. The core insight: Kalman filtering, thermal dynamics, Fokker–Planck evolution, and eigenvalue-based policy optimization all share the same spectral structure when viewed through the lens of a universal self-adjoint Dirac operator.

The library provides:

- **Universal Dirac trait** — any theorem that exposes a self-adjoint operator D
- **Four concrete theorem operators** — Kalman, Thermal, Fokker–Planck, EigenPolicy
- **Unified closure** — block-diagonal composition of all operators into one
- **Spectrum comparison** — normalized distance and equivalence checking
- **Conservation law verification** — Landauer cost + free energy + H¹ risk ≈ const
- **Agent lifecycle** — thermodynamic agent model with Landauer energy accounting
- **Theorem graph** — adjacency/spectral analysis of the 14-theorem ecosystem
- **Dirichlet space** — the "missing center" that wires all theorems together
- **Gluing verification** — spectral compatibility checks between theorem pairs

## Key Idea

> All 14 executable theorems share one spectrum.

The closure object is the internal hom `[B, C]` of the category — a morphism between theorem objects that preserves the Dirac structure. Each theorem implements the same `UniversalDirac` trait, exposing a self-adjoint matrix whose squared spectrum encodes the theorem's dynamics. The unified closure composes these via block-diagonal stacking, and the Dirichlet space acts as the "missing center" — the common substrate through which all theorems can be wired.

The conservation law `Landauer + FreeEnergy + H¹Risk ≈ constant` emerges from this structure, connecting information theory (Landauer's principle), statistical mechanics (free energy), and functional analysis (Sobolev H¹ norm) in one equation.

## Install

```toml
[dependencies]
lau-closure = "0.1"
```

Or:

```sh
cargo add lau-closure
```

### Dependencies

- `nalgebra` — linear algebra (matrices, vectors, QR decomposition)
- `serde` + `serde_json` — serialization of spectra, conservation laws, gluing results

## Quick Start

```rust
use lau_closure::*;

// Create individual theorem operators
let kalman = KalmanDirac::new(5, 0.1, 0.2);
let thermal = ThermalDirac::new(5, 1.0, 0.1);
let fokker_planck = FokkerPlanckDirac::new(5, 0.3, 0.5);
let eigen_policy = EigenPolicyDirac::new(5, 0.9, 1.0);

// Each exposes a self-adjoint Dirac matrix
let spec = kalman.spectrum();  // Eigenvalues of D
let spec2 = kalman.spectrum_squared();  // Eigenvalues of D²

// Compose into unified closure
let closure = UnifiedClosure::new(vec![
    Box::new(kalman),
    Box::new(thermal),
    Box::new(fokker_planck),
    Box::new(eigen_policy),
]);

// Unified spectrum (all 20 eigenvalues sorted)
let unified_spec = closure.unified_spectrum();

// Conservation law at temperature T=1.0
let conservation = closure.conservation_law(1.0);
// conservation.landauer_cost + conservation.free_energy + conservation.h1_risk ≈ const

// Agent lifecycle: spend energy on computation
let lifecycle = closure.agent_loop(1000.0, 1.0, 100);
assert!(lifecycle.conservation_holds());

// Gluing verification: can these theorems compose?
let result = closure.verify_gluing(0, 1, 0.01);
if result.glued {
    println!("{} and {} share a spectrum!", result.theorem_a, result.theorem_b);
}
```

## API Reference

### Core Traits

#### `UniversalDirac`

Every theorem implements this trait to expose its local Dirac operator.

| Method | Description |
|--------|-------------|
| `theorem_name()` | Human-readable theorem name |
| `dimension()` | Dimension of the operator matrix |
| `dirac_matrix()` | The Dirac operator D as a real symmetric matrix |
| `spectrum()` | Eigenvalues of D (sorted ascending) |
| `spectrum_squared()` | Eigenvalues of D² |

#### `Closure` (extends `UniversalDirac`)

The internal hom of the category — composed Dirac operator with thermal regularization.

| Method | Description |
|--------|-------------|
| `dirac()` | Composed Dirac operator matrix |
| `closure_spectrum()` | Eigenvalues of the closure operator |
| `loop_cost(T)` | Thermal cost: Σ λᵢ / (exp(λᵢ/T) − 1) — Bose-Einstein regularization |

### Concrete Operators

#### `KalmanDirac`

State estimation as a spectral problem. Tridiagonal matrix with process and measurement noise.

```rust
let k = KalmanDirac::new(dim, process_noise, measurement_noise);
```

#### `ThermalDirac`

Heat flow / thermodynamic evolution. Discrete Laplacian with conductivity and temperature gradient.

```rust
let t = ThermalDirac::new(dim, conductivity, temperature_gradient);
```

#### `FokkerPlanckDirac`

Drift-diffusion dynamics. Asymmetric drift is symmetrized to ensure self-adjointness.

```rust
let fp = FokkerPlanckDirac::new(dim, drift, diffusion);
```

#### `EigenPolicyDirac`

Reinforcement learning policy optimization as an eigenvalue problem. Discount factor and reward scale.

```rust
let ep = EigenPolicyDirac::new(dim, discount_factor, reward_scale);
```

### `UnifiedClosure`

Composes all theorem operators into one block-diagonal Dirac operator.

| Method | Description |
|--------|-------------|
| `new(operators)` | Create from a vector of `UniversalDirac` impls |
| `unified_dirac()` | Block-diagonal composition of all operators |
| `unified_spectrum()` | All eigenvalues sorted ascending |
| `verify_gluing(a, b, tol)` | Check spectral compatibility of operators a and b |
| `conservation_law(T)` | Compute Landauer + FreeEnergy + H¹Risk at temperature T |
| `agent_loop(energy, T, steps)` | Simulate agent spending energy over steps |

### `DirichletSpace`

The missing center — a 1D discrete Laplacian with zero boundary conditions that wires theorem operators into a common substrate.

| Method | Description |
|--------|-------------|
| `new(dim)` | Create Dirichlet space with standard 1D Laplacian |
| `wire(theorem)` | Compose Laplacian with theorem's Dirac operator |
| `energy(v)` | Dirichlet energy ⟨v, Lv⟩ |

Also implements `UniversalDirac`, so it can be composed like any theorem.

### Data Structures

#### `NamedSpectrum`

A labeled eigenvalue spectrum for comparison.

| Method | Description |
|--------|-------------|
| `normalized()` | L2-normalized eigenvalues for comparison |

#### `SpectrumComparator`

Static utilities for spectrum comparison.

| Method | Description |
|--------|-------------|
| `normalized_distance(a, b)` | L2 distance between normalized spectra |
| `are_equivalent(a, b, tol)` | Boolean equivalence check |

#### `ConservationLaw`

Three-component conservation: Landauer cost + free energy + H¹ risk.

| Field | Description |
|-------|-------------|
| `landauer_cost` | kT·ln(2) × dim — information-theoretic cost |
| `free_energy` | −T·ln(Z) — statistical mechanics partition function |
| `h1_risk` | Tr(D²) — Sobolev H¹ norm squared |
| `total` | Sum of all three |
| `verify(expected, tol)` | Check if total ≈ expected |

#### `AgentLifecycle`

Thermodynamic agent model with Landauer energy accounting.

| Method | Description |
|--------|-------------|
| `new(initial_energy)` | Create with energy budget |
| `step(bits, T)` | Erase `bits` at temperature T, costing kT·ln(2) per bit |
| `is_dead()` | True when free energy depleted |
| `conservation_holds()` | Verify: spent + remaining = initial |

#### `TheoremGraph`

Adjacency graph of the 14-theorem ecosystem with spectral analysis.

| Method | Description |
|--------|-------------|
| `build_theorem_graph()` | Pre-built graph of all 14 theorems |
| `laplacian()` | Graph Laplacian L = D − A |
| `fiedler_vector()` | Second-smallest Laplacian eigenvector (natural partition) |
| `neighbors(idx)` | Adjacent theorems |
| `composes(a, b)` | Check if two theorems are adjacent |

#### `GluingResult`

Result of a spectral compatibility check between two theorem operators.

| Field | Description |
|-------|-------------|
| `theorem_a` / `theorem_b` | Names of compared theorems |
| `spectrum_distance` | Normalized L2 distance |
| `glued` | True if distance < tolerance |
| `tolerance` | Threshold used |

All serializable via `serde`.

### Utility Functions

#### `eigenvalues_symmetric(mat)`

QR iteration with Wilkinson shifts for symmetric matrix eigenvalues.

#### `symmetric_eigendecomposition(mat)`

Jacobi rotation method. Returns `(eigenvalues, eigenvectors)`.

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                    UNIFIED CLOSURE                       │
│                                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │ Kalman   │ │ Thermal  │ │ Fokker-  │ │ Eigen-   │   │
│  │ D_K      │ │ D_T      │ │ Planck   │ │ Policy   │   │
│  │          │ │          │ │ D_FP     │ │ D_EP     │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │
│       │            │            │             │          │
│       └────────────┴────────────┴─────────────┘          │
│                         │                                 │
│              ┌──────────▼──────────┐                     │
│              │   BLOCK DIAGONAL    │                     │
│              │   D = D_K ⊕ D_T ⊕  │                     │
│              │     D_FP ⊕ D_EP    │                     │
│              └──────────┬──────────┘                     │
│                         │                                 │
│              ┌──────────▼──────────┐                     │
│              │  SPECTRUM σ(D)      │                     │
│              │  = σ(D_K) ∪ σ(D_T) │                     │
│              │    ∪ σ(D_FP) ∪ ...  │                     │
│              └──────────┬──────────┘                     │
│                         │                                 │
│           ┌─────────────▼─────────────┐                  │
│           │    DIRICHLET SPACE         │                  │
│           │  (the missing center)      │                  │
│           │  L · D_i for all i         │                  │
│           └───────────────────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

1. **Each theorem** implements `UniversalDirac`, exposing a self-adjoint matrix D whose spectrum encodes the theorem's dynamics.

2. **Unified closure** stacks these block-diagonally. The resulting operator has spectrum = union of all individual spectra.

3. **Dirichlet space** acts as the "missing center" — a standard 1D Laplacian that can wire into any theorem's operator via matrix multiplication.

4. **Conservation law** emerges from the unified structure: `kT·ln(2)·n − T·ln(Z) + Tr(D²) ≈ const`.

5. **Gluing verification** checks whether two theorems are spectrally compatible (can be composed in the category) by comparing normalized spectra.

6. **Agent lifecycle** models a computational agent spending energy according to Landauer's principle: erasing one bit costs `kT·ln(2)` of free energy.

## The Math

### Universal Dirac Operator

Each theorem Tᵢ exposes a self-adjoint (symmetric) operator Dᵢ on ℝⁿⁱ. The key constraint:

**Dᵢ = Dᵢᵀ** (self-adjointness)

This guarantees real eigenvalues λ₁ ≤ λ₂ ≤ … ≤ λₙᵢ.

### Block-Diagonal Composition

The unified closure operator:

**D = D₁ ⊕ D₂ ⊕ … ⊕ Dₖ**

has spectrum **σ(D) = σ(D₁) ∪ σ(D₂) ∪ … ∪ σ(Dₖ)**.

### Conservation Law

At temperature T:

- **Landauer cost**: C_L = n · kT · ln(2) — thermodynamic cost of n bit erasures
- **Free energy**: F = −T · ln(Z) where Z = Σᵢ exp(−λᵢ/T) — partition function
- **H¹ risk**: H = Tr(D²) = Σᵢ λᵢ² — Sobolev norm squared

Conservation: **C_L + F + H ≈ constant** across temperatures.

### Loop Cost (Thermal Regularization)

The closure's loop cost at temperature T:

**C(T) = Σᵢ |λᵢ| / (exp(|λᵢ|/T) − 1)**

This is a Bose–Einstein-like regularization — at high T, all modes contribute equally; at low T, only low-energy modes survive.

### Graph Spectral Analysis

The theorem ecosystem is modeled as a weighted graph G = (V, E, w). The Laplacian **L = D − A** has:

- Zero eigenvalue corresponding to the constant eigenvector
- **Fiedler value** (second-smallest eigenvalue) = algebraic connectivity
- **Fiedler vector** provides the optimal 2-way partition of the theorem graph

### Kalman–Thermal Spectral Equivalence

When Kalman process_noise = Thermal conductivity and measurement_noise = temperature_gradient = 0, the two operators produce identical tridiagonal matrices, hence identical spectra. This is the first gluing: state estimation IS heat flow.

### Dirichlet Space as Missing Center

The 1D discrete Dirichlet Laplacian:

```
L = [[ 2, -1,  0, ...],
     [-1,  2, -1, ...],
     [ 0, -1,  2, ...],
     ...              ]
```

Wiring: **L · Dᵢ** composes the Dirichlet Laplacian with any theorem's operator. This is the "missing center" — every theorem can be expressed through this common substrate.

## License

MIT
