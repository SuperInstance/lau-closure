# lau-closure

> The missing center — Universal Dirac operator and closure object proving all 14 executable theorems share one spectrum

## What This Does

The missing center — Universal Dirac operator and closure object proving all 14 executable theorems share one spectrum. Part of the PLATO/LAU ecosystem — a mathematically rigorous framework for building educational agents that learn, teach, and evolve.

## The Key Idea

This crate implements the core abstractions needed for its domain, with a focus on correctness, composability, and conservation guarantees. Every public type is serializable (serde), every algorithm is tested, and every invariant is verified.

## Install

```bash
cargo add lau-closure
```

## Quick Start

See the API Reference below for complete usage. Key entry points:

```rust
use lau_closure::*;
// See types and methods below for complete usage
```

## API Reference

```rust
pub fn eigenvalues_symmetric(mat: &DMatrix<f64>) -> DVector<f64> 
pub fn symmetric_eigendecomposition(mat: &DMatrix<f64>) -> (Vec<f64>, DMatrix<f64>) 
pub trait UniversalDirac: Send + Sync 
pub trait Closure: UniversalDirac 
pub struct NamedSpectrum 
    pub fn normalized(&self) -> Vec<f64> 
pub struct ConservationLaw 
    pub fn verify(&self, expected_constant: f64, tolerance: f64) -> bool 
pub struct AgentLifecycle 
    pub fn new(initial_energy: f64) -> Self 
    pub fn step(&mut self, bits_erased: f64, temperature: f64) -> f64 
    pub fn is_dead(&self) -> bool 
    pub fn conservation_holds(&self) -> bool 
pub struct TheoremGraph 
    pub fn new(names: Vec<String>, adj: DMatrix<f64>) -> Self 
    pub fn laplacian(&self) -> DMatrix<f64> 
    pub fn fiedler_vector(&self) -> (f64, DVector<f64>) 
    pub fn neighbors(&self, theorem_idx: usize) -> Vec<usize> 
    pub fn composes(&self, a: usize, b: usize) -> bool 
pub struct GluingResult 
pub struct SpectrumComparator;
    pub fn normalized_distance(a: &NamedSpectrum, b: &NamedSpectrum) -> f64 
    pub fn are_equivalent(a: &NamedSpectrum, b: &NamedSpectrum, tolerance: f64) -> bool 
pub struct KalmanDirac 
    pub fn new(dim: usize, process_noise: f64, measurement_noise: f64) -> Self 
pub struct ThermalDirac 
    pub fn new(dim: usize, conductivity: f64, temperature_gradient: f64) -> Self 
pub struct FokkerPlanckDirac 
    pub fn new(dim: usize, drift: f64, diffusion: f64) -> Self 
pub struct EigenPolicyDirac 
    pub fn new(dim: usize, discount_factor: f64, reward_scale: f64) -> Self 
pub struct UnifiedClosure 
    pub fn new(operators: Vec<Box<dyn UniversalDirac>>) -> Self 
    pub fn unified_dirac(&self) -> DMatrix<f64> 
    pub fn verify_gluing(&self, idx_a: usize, idx_b: usize, tolerance: f64) -> GluingResult 
    pub fn conservation_law(&self, temperature: f64) -> ConservationLaw 
    pub fn unified_spectrum(&self) -> DVector<f64> 
    pub fn agent_loop(&self, initial_energy: f64, temperature: f64, steps: usize) -> AgentLifecycle 
pub struct DirichletSpace 
    pub fn new(dim: usize) -> Self 
    pub fn wire(&self, theorem: &dyn UniversalDirac) -> DMatrix<f64> 
    pub fn energy(&self, v: &DVector<f64>) -> f64 
pub fn build_theorem_graph() -> TheoremGraph 
```

## How It Works

Read the source in `src/` for full implementation details. All algorithms are documented with inline comments explaining the mathematical foundations.

## The Math

This crate implements formal mathematical constructs. See the source documentation for theorem statements and proofs of correctness.

## Testing

**71 tests** covering construction, serialization, correctness properties, edge cases, and composability with other lau-* crates.

## License

MIT
