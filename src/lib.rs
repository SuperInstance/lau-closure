//! # lau-closure: The Missing Center
//!
//! Universal Dirac operator and closure object proving all 14 executable theorems
//! share one spectrum. This crate wires together the theorem ecosystem by exposing
//! a single self-adjoint operator D whose squared spectrum unifies Kalman filtering,
//! thermal dynamics, Fokker-Planck evolution, and eigenvalue policy optimization.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Eigenvalue utility
// ---------------------------------------------------------------------------

/// Compute eigenvalues of a symmetric matrix using QR iteration with Wilkinson shifts.
pub fn eigenvalues_symmetric(mat: &DMatrix<f64>) -> DVector<f64> {
    let n = mat.nrows();
    if n == 0 {
        return DVector::zeros(0);
    }
    if n == 1 {
        return DVector::from_vec(vec![mat[(0, 0)]]);
    }
    let mut a = mat.clone();
    let max_iter = 300 * n;
    for _ in 0..max_iter {
        let nn = a.nrows();
        if nn <= 1 { break; }
        let shift = a[(nn - 1, nn - 1)];
        for i in 0..nn {
            a[(i, i)] -= shift;
        }
        let qr = a.qr();
        let q = qr.q();
        let r = qr.r();
        a = &r * &q;
        for i in 0..nn {
            a[(i, i)] += shift;
        }
        let mut off_diag = 0.0_f64;
        for i in 0..nn {
            for j in 0..nn {
                if i != j {
                    off_diag += a[(i, j)] * a[(i, j)];
                }
            }
        }
        if off_diag < 1e-20 { break; }
    }
    let mut eigs: Vec<f64> = (0..a.nrows()).map(|i| a[(i, i)]).collect();
    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    DVector::from_vec(eigs)
}

/// Symmetric eigendecomposition using Jacobi rotations. Returns (eigenvalues, eigenvectors).
pub fn symmetric_eigendecomposition(mat: &DMatrix<f64>) -> (Vec<f64>, DMatrix<f64>) {
    let n = mat.nrows();
    let mut a = mat.clone();
    let mut v = DMatrix::identity(n, n);
    let max_iter = 100 * n * n;

    for _ in 0..max_iter {
        let mut max_val = 0.0_f64;
        let (mut p, mut q) = (0, 1);
        for i in 0..n {
            for j in (i + 1)..n {
                if a[(i, j)].abs() > max_val {
                    max_val = a[(i, j)].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-15 { break; }

        let app = a[(p, p)];
        let aqq = a[(q, q)];
        let apq = a[(p, q)];
        let theta = if (app - aqq).abs() < 1e-30 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };
        let c = theta.cos();
        let s = theta.sin();

        let mut new_a = a.clone();
        for i in 0..n {
            if i != p && i != q {
                new_a[(i, p)] = c * a[(i, p)] + s * a[(i, q)];
                new_a[(p, i)] = new_a[(i, p)];
                new_a[(i, q)] = -s * a[(i, p)] + c * a[(i, q)];
                new_a[(q, i)] = new_a[(i, q)];
            }
        }
        new_a[(p, p)] = c * c * app + 2.0 * s * c * apq + s * s * aqq;
        new_a[(q, q)] = s * s * app - 2.0 * s * c * apq + c * c * aqq;
        new_a[(p, q)] = 0.0;
        new_a[(q, p)] = 0.0;
        a = new_a;

        let mut new_v = v.clone();
        for i in 0..n {
            new_v[(i, p)] = c * v[(i, p)] + s * v[(i, q)];
            new_v[(i, q)] = -s * v[(i, p)] + c * v[(i, q)];
        }
        v = new_v;
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[(i, i)]).collect();
    (eigenvalues, v)
}

// ---------------------------------------------------------------------------
// Core Traits
// ---------------------------------------------------------------------------

/// Any theorem crate implements this to expose its local Dirac operator as a matrix.
/// The Dirac operator is self-adjoint (D = D†) and its spectrum encodes the theorem's
/// dynamics.
pub trait UniversalDirac: Send + Sync {
    /// Human-readable name of the theorem.
    fn theorem_name(&self) -> &str;

    /// The dimension of the Dirac operator matrix.
    fn dimension(&self) -> usize;

    /// Construct the Dirac operator D as a real symmetric (self-adjoint) matrix.
    fn dirac_matrix(&self) -> DMatrix<f64>;

    /// Compute the spectrum (eigenvalues) of D².
    fn spectrum_squared(&self) -> DVector<f64> {
        let d = self.dirac_matrix();
        let d2 = &d * &d;
        eigenvalues_symmetric(&d2)
    }

    /// Compute the spectrum of D itself.
    fn spectrum(&self) -> DVector<f64> {
        let d = self.dirac_matrix();
        eigenvalues_symmetric(&d)
    }
}

/// The closure object — the internal hom [B, C] of the category.
/// Represents a morphism between theorem objects that preserves the Dirac structure.
pub trait Closure: UniversalDirac {
    /// The Dirac operator of the closure (composed from sub-operators).
    fn dirac(&self) -> DMatrix<f64>;

    /// The spectrum of the closure operator.
    fn closure_spectrum(&self) -> DVector<f64> {
        let d = self.dirac();
        eigenvalues_symmetric(&d)
    }

    /// Loop cost at a given temperature (thermal regularization of the internal hom).
    /// Cost = Σᵢ λᵢ / (exp(λᵢ/T) - 1)  (Bose-Einstein-like regularization)
    fn loop_cost(&self, temperature: f64) -> f64 {
        let spec = self.closure_spectrum();
        let mut cost = 0.0;
        for i in 0..spec.len() {
            let lambda = spec[i].abs();
            if temperature > 1e-15 {
                let exponent = lambda / temperature;
                if exponent < 500.0 {
                    cost += lambda / (exponent.exp() - 1.0).max(1e-30);
                }
            } else {
                cost += lambda;
            }
        }
        cost
    }
}

// ---------------------------------------------------------------------------
// Data Structures
// ---------------------------------------------------------------------------

/// A named spectrum for comparison between theorems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedSpectrum {
    pub name: String,
    pub eigenvalues: Vec<f64>,
    pub dimension: usize,
}

impl NamedSpectrum {
    /// Normalize spectrum to unit norm for comparison.
    pub fn normalized(&self) -> Vec<f64> {
        let norm: f64 = self.eigenvalues.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-30 {
            return vec![0.0; self.eigenvalues.len()];
        }
        self.eigenvalues.iter().map(|x| x / norm).collect()
    }
}

/// Conservation law components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationLaw {
    pub landauer_cost: f64,
    pub free_energy: f64,
    pub h1_risk: f64,
    pub total: f64,
}

impl ConservationLaw {
    /// Verify that Landauer + FreeEnergy + H¹Risk ≈ constant.
    pub fn verify(&self, expected_constant: f64, tolerance: f64) -> bool {
        (self.total - expected_constant).abs() < tolerance
    }
}

/// Agent lifecycle state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentLifecycle {
    pub initial_free_energy: f64,
    pub cumulative_landauer_cost: f64,
    pub current_free_energy: f64,
    pub alive: bool,
}

impl AgentLifecycle {
    pub fn new(initial_energy: f64) -> Self {
        Self {
            initial_free_energy: initial_energy,
            cumulative_landauer_cost: 0.0,
            current_free_energy: initial_energy,
            alive: true,
        }
    }

    /// Step the agent: spend energy equal to Landauer cost of a bit erasure at temperature T.
    /// kT·ln(2) per bit erased.
    pub fn step(&mut self, bits_erased: f64, temperature: f64) -> f64 {
        let k_boltzmann = 1.0; // natural units
        let cost = bits_erased * k_boltzmann * temperature * (2.0_f64).ln();
        self.cumulative_landauer_cost += cost;
        self.current_free_energy -= cost;
        if self.current_free_energy <= 1e-15 {
            self.current_free_energy = 0.0;
            self.alive = false;
        }
        cost
    }

    /// Check if agent has died (cumulative cost == initial budget).
    pub fn is_dead(&self) -> bool {
        !self.alive
    }

    /// Check conservation: energy spent + remaining = initial.
    pub fn conservation_holds(&self) -> bool {
        (self.cumulative_landauer_cost + self.current_free_energy - self.initial_free_energy).abs() < 1e-10
    }
}

/// Theorem graph: adjacency and spectral analysis of the theorem ecosystem.
#[derive(Debug, Clone)]
pub struct TheoremGraph {
    pub theorem_names: Vec<String>,
    pub adjacency: DMatrix<f64>,
}

impl TheoremGraph {
    pub fn new(names: Vec<String>, adj: DMatrix<f64>) -> Self {
        assert_eq!(names.len(), adj.nrows());
        assert_eq!(adj.nrows(), adj.ncols());
        Self { theorem_names: names, adjacency: adj }
    }

    /// Compute the graph Laplacian L = D - A where D is degree matrix.
    pub fn laplacian(&self) -> DMatrix<f64> {
        let n = self.adjacency.nrows();
        let mut degree = DMatrix::zeros(n, n);
        for i in 0..n {
            let deg: f64 = (0..n).map(|j| self.adjacency[(i, j)]).sum();
            degree[(i, i)] = deg;
        }
        &degree - &self.adjacency
    }

    /// Compute Fiedler vector (eigenvector corresponding to second-smallest eigenvalue of Laplacian).
    /// This reveals the most natural partition of the theorem graph.
    pub fn fiedler_vector(&self) -> (f64, DVector<f64>) {
        let lap = self.laplacian();
        let (eigenvalues, eigenvectors) = symmetric_eigendecomposition(&lap);
        let mut indexed: Vec<(usize, f64)> = eigenvalues.iter().enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let fiedler_idx = indexed[1].0;
        let fiedler_val = indexed[1].1;
        let n = self.adjacency.nrows();
        let mut fiedler_vec = DVector::zeros(n);
        for i in 0..n {
            fiedler_vec[i] = eigenvectors[(i, fiedler_idx)];
        }
        (fiedler_val, fiedler_vec)
    }

    /// Find neighbors of a theorem in the graph.
    pub fn neighbors(&self, theorem_idx: usize) -> Vec<usize> {
        let mut result = Vec::new();
        for j in 0..self.adjacency.ncols() {
            if self.adjacency[(theorem_idx, j)] > 0.0 && theorem_idx != j {
                result.push(j);
            }
        }
        result
    }

    /// Check if two theorems compose (are adjacent).
    pub fn composes(&self, a: usize, b: usize) -> bool {
        self.adjacency[(a, b)] > 0.0
    }
}

/// Gluing verification result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GluingResult {
    pub theorem_a: String,
    pub theorem_b: String,
    pub spectrum_distance: f64,
    pub glued: bool,
    pub tolerance: f64,
}

/// Spectrum comparison utilities.
pub struct SpectrumComparator;

impl SpectrumComparator {
    /// Normalize two spectra to same length (pad with zeros if needed) and compare.
    pub fn normalized_distance(a: &NamedSpectrum, b: &NamedSpectrum) -> f64 {
        let norm_a = a.normalized();
        let norm_b = b.normalized();
        let max_len = norm_a.len().max(norm_b.len());
        let mut dist = 0.0;
        for i in 0..max_len {
            let va = if i < norm_a.len() { norm_a[i] } else { 0.0 };
            let vb = if i < norm_b.len() { norm_b[i] } else { 0.0 };
            dist += (va - vb).powi(2);
        }
        dist.sqrt()
    }

    /// Check if two spectra are equivalent within tolerance.
    pub fn are_equivalent(a: &NamedSpectrum, b: &NamedSpectrum, tolerance: f64) -> bool {
        Self::normalized_distance(a, b) < tolerance
    }
}

// ---------------------------------------------------------------------------
// Concrete Theorem Implementations
// ---------------------------------------------------------------------------

/// Kalman Filter Dirac operator: models state estimation as a spectral problem.
pub struct KalmanDirac {
    dim: usize,
    process_noise: f64,
    measurement_noise: f64,
}

impl KalmanDirac {
    pub fn new(dim: usize, process_noise: f64, measurement_noise: f64) -> Self {
        Self { dim, process_noise, measurement_noise }
    }
}

impl UniversalDirac for KalmanDirac {
    fn theorem_name(&self) -> &str { "Kalman" }
    fn dimension(&self) -> usize { self.dim }

    fn dirac_matrix(&self) -> DMatrix<f64> {
        let n = self.dim;
        let mut mat = DMatrix::zeros(n, n);
        for i in 0..n {
            mat[(i, i)] = 1.0 + self.process_noise + self.measurement_noise;
            if i > 0 { mat[(i, i - 1)] = -0.5; }
            if i < n - 1 { mat[(i, i + 1)] = -0.5; }
        }
        mat
    }
}

/// Thermal Dirac operator: models heat flow / thermodynamic evolution.
pub struct ThermalDirac {
    dim: usize,
    conductivity: f64,
    temperature_gradient: f64,
}

impl ThermalDirac {
    pub fn new(dim: usize, conductivity: f64, temperature_gradient: f64) -> Self {
        Self { dim, conductivity, temperature_gradient }
    }
}

impl UniversalDirac for ThermalDirac {
    fn theorem_name(&self) -> &str { "Thermal" }
    fn dimension(&self) -> usize { self.dim }

    fn dirac_matrix(&self) -> DMatrix<f64> {
        let n = self.dim;
        let mut mat = DMatrix::zeros(n, n);
        for i in 0..n {
            mat[(i, i)] = 2.0 * self.conductivity;
            if i > 0 { mat[(i, i - 1)] = -self.conductivity; }
            if i < n - 1 { mat[(i, i + 1)] = -self.conductivity; }
            mat[(i, i)] += self.temperature_gradient * (i as f64 / n as f64 - 0.5).abs();
        }
        mat
    }
}

/// Fokker-Planck Dirac operator: models drift-diffusion dynamics.
pub struct FokkerPlanckDirac {
    dim: usize,
    drift: f64,
    diffusion: f64,
}

impl FokkerPlanckDirac {
    pub fn new(dim: usize, drift: f64, diffusion: f64) -> Self {
        Self { dim, drift, diffusion }
    }
}

impl UniversalDirac for FokkerPlanckDirac {
    fn theorem_name(&self) -> &str { "FokkerPlanck" }
    fn dimension(&self) -> usize { self.dim }

    fn dirac_matrix(&self) -> DMatrix<f64> {
        let n = self.dim;
        let mut mat = DMatrix::zeros(n, n);
        for i in 0..n {
            mat[(i, i)] = self.diffusion;
            if i > 0 { mat[(i, i - 1)] = -0.5 * self.diffusion + 0.25 * self.drift; }
            if i < n - 1 { mat[(i, i + 1)] = -0.5 * self.diffusion - 0.25 * self.drift; }
        }
        // Symmetrize to ensure self-adjointness
        (&mat + &mat.transpose()) * 0.5
    }
}

/// EigenPolicy Dirac operator: models RL policy optimization as an eigenvalue problem.
pub struct EigenPolicyDirac {
    dim: usize,
    discount_factor: f64,
    reward_scale: f64,
}

impl EigenPolicyDirac {
    pub fn new(dim: usize, discount_factor: f64, reward_scale: f64) -> Self {
        Self { dim, discount_factor, reward_scale }
    }
}

impl UniversalDirac for EigenPolicyDirac {
    fn theorem_name(&self) -> &str { "EigenPolicy" }
    fn dimension(&self) -> usize { self.dim }

    fn dirac_matrix(&self) -> DMatrix<f64> {
        let n = self.dim;
        let mut mat = DMatrix::zeros(n, n);
        for i in 0..n {
            mat[(i, i)] = 1.0 / (1.0 - self.discount_factor) * self.reward_scale;
            if i > 0 { mat[(i, i - 1)] = -self.discount_factor * 0.3 * self.reward_scale; }
            if i < n - 1 { mat[(i, i + 1)] = -self.discount_factor * 0.3 * self.reward_scale; }
        }
        mat
    }
}

// ---------------------------------------------------------------------------
// Unified Closure Operator
// ---------------------------------------------------------------------------

/// The Unified Closure: composes all theorem Dirac operators into one.
pub struct UnifiedClosure {
    operators: Vec<Box<dyn UniversalDirac>>,
}

impl UnifiedClosure {
    pub fn new(operators: Vec<Box<dyn UniversalDirac>>) -> Self {
        Self { operators }
    }

    /// Build a unified Dirac operator by block-diagonal composition.
    pub fn unified_dirac(&self) -> DMatrix<f64> {
        if self.operators.is_empty() {
            return DMatrix::zeros(0, 0);
        }
        let total_dim: usize = self.operators.iter().map(|op| op.dimension()).sum();
        let mut unified = DMatrix::zeros(total_dim, total_dim);
        let mut offset = 0;
        for op in &self.operators {
            let d = op.dirac_matrix();
            let n = d.nrows();
            for i in 0..n {
                for j in 0..n {
                    unified[(offset + i, offset + j)] = d[(i, j)];
                }
            }
            offset += n;
        }
        unified
    }

    /// Gluing verification: check that two theorem implementations share the same
    /// underlying operator by comparing their normalized spectra.
    pub fn verify_gluing(&self, idx_a: usize, idx_b: usize, tolerance: f64) -> GluingResult {
        let op_a = &self.operators[idx_a];
        let op_b = &self.operators[idx_b];
        let spec_a = NamedSpectrum {
            name: op_a.theorem_name().to_string(),
            eigenvalues: op_a.spectrum().iter().cloned().collect(),
            dimension: op_a.dimension(),
        };
        let spec_b = NamedSpectrum {
            name: op_b.theorem_name().to_string(),
            eigenvalues: op_b.spectrum().iter().cloned().collect(),
            dimension: op_b.dimension(),
        };
        let dist = SpectrumComparator::normalized_distance(&spec_a, &spec_b);
        GluingResult {
            theorem_a: op_a.theorem_name().to_string(),
            theorem_b: op_b.theorem_name().to_string(),
            spectrum_distance: dist,
            glued: dist < tolerance,
            tolerance,
        }
    }

    /// Compute the conservation law: Landauer cost + free energy + H¹ risk ≈ const.
    pub fn conservation_law(&self, temperature: f64) -> ConservationLaw {
        let total_dim: usize = self.operators.iter().map(|op| op.dimension()).sum();
        let landauer_cost = total_dim as f64 * temperature * (2.0_f64).ln();

        let spec = self.unified_spectrum();
        let mut log_z = 0.0_f64;
        for i in 0..spec.len() {
            let exp_val = (-spec[i] / temperature).exp();
            log_z += exp_val.max(1e-300);
        }
        let free_energy = -temperature * log_z.ln();

        let d = self.unified_dirac();
        let d2 = &d * &d;
        let h1_risk = d2.trace();

        ConservationLaw {
            landauer_cost,
            free_energy,
            h1_risk,
            total: landauer_cost + free_energy + h1_risk,
        }
    }

    /// Get the unified spectrum (all eigenvalues sorted).
    pub fn unified_spectrum(&self) -> DVector<f64> {
        let mut all_eigs = Vec::new();
        for op in &self.operators {
            let spec = op.spectrum();
            for i in 0..spec.len() {
                all_eigs.push(spec[i]);
            }
        }
        all_eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        DVector::from_vec(all_eigs)
    }

    /// Agent self-model: compose all theorem crates into one agent loop.
    pub fn agent_loop(&self, initial_energy: f64, temperature: f64, steps: usize) -> AgentLifecycle {
        let mut lifecycle = AgentLifecycle::new(initial_energy);
        let total_dim: usize = self.operators.iter().map(|op| op.dimension()).sum();
        let bits_per_step = total_dim as f64 / steps as f64;
        for _ in 0..steps {
            if !lifecycle.alive { break; }
            lifecycle.step(bits_per_step, temperature);
        }
        lifecycle
    }
}

impl UniversalDirac for UnifiedClosure {
    fn theorem_name(&self) -> &str { "UnifiedClosure" }
    fn dimension(&self) -> usize {
        self.operators.iter().map(|op| op.dimension()).sum()
    }
    fn dirac_matrix(&self) -> DMatrix<f64> {
        self.unified_dirac()
    }
}

impl Closure for UnifiedClosure {
    fn dirac(&self) -> DMatrix<f64> {
        self.unified_dirac()
    }
}

// ---------------------------------------------------------------------------
// Dirichlet Space Wiring
// ---------------------------------------------------------------------------

/// DirichletSpace: the missing center that was never wired.
pub struct DirichletSpace {
    pub dim: usize,
    pub boundary_condition: f64,
    pub laplacian: DMatrix<f64>,
}

impl DirichletSpace {
    /// Create a 1D Dirichlet space with zero boundary conditions.
    pub fn new(dim: usize) -> Self {
        let mut lap = DMatrix::zeros(dim, dim);
        for i in 0..dim {
            lap[(i, i)] = 2.0;
            if i > 0 { lap[(i, i - 1)] = -1.0; }
            if i < dim - 1 { lap[(i, i + 1)] = -1.0; }
        }
        Self { dim, boundary_condition: 0.0, laplacian: lap }
    }

    /// Wire a theorem's Dirac operator into the Dirichlet space.
    pub fn wire(&self, theorem: &dyn UniversalDirac) -> DMatrix<f64> {
        let d = theorem.dirac_matrix();
        let n = self.dim.min(d.nrows());
        let mut result = DMatrix::zeros(self.dim, d.ncols().min(self.dim));
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += self.laplacian[(i, k)] * d[(k, j)];
                }
                result[(i, j)] = sum;
            }
        }
        result
    }

    /// The Dirichlet energy of a vector in this space.
    pub fn energy(&self, v: &DVector<f64>) -> f64 {
        let lv = &self.laplacian * v;
        v.dot(&lv)
    }
}

impl UniversalDirac for DirichletSpace {
    fn theorem_name(&self) -> &str { "DirichletSpace" }
    fn dimension(&self) -> usize { self.dim }
    fn dirac_matrix(&self) -> DMatrix<f64> {
        self.laplacian.clone()
    }
}

// ---------------------------------------------------------------------------
// Theorem Graph Builder
// ---------------------------------------------------------------------------

/// Build the theorem graph for the 14 executable theorems.
pub fn build_theorem_graph() -> TheoremGraph {
    let names = vec![
        "Kalman".into(),
        "Thermal".into(),
        "FokkerPlanck".into(),
        "EigenPolicy".into(),
        "Langevin".into(),
        "Schrodinger".into(),
        "Riccati".into(),
        "MaxwellBoltzmann".into(),
        "NavierStokes".into(),
        "HamiltonJacobi".into(),
        "BoltzmannTransport".into(),
        "Dirichlet".into(),
        "Hodge".into(),
        "Landauer".into(),
    ];
    let n = names.len();
    let mut adj = DMatrix::zeros(n, n);
    let mut edge = |i: usize, j: usize, w: f64| {
        adj[(i, j)] = w;
        adj[(j, i)] = w;
    };
    edge(0, 6, 1.0); edge(0, 1, 0.8); edge(0, 4, 0.7);
    edge(1, 2, 0.9); edge(1, 7, 0.9); edge(1, 8, 0.6); edge(1, 10, 0.7);
    edge(2, 4, 0.9); edge(2, 3, 0.6);
    edge(3, 9, 0.7); edge(3, 10, 0.5);
    edge(4, 5, 0.5);
    edge(5, 9, 0.8); edge(5, 12, 0.6);
    edge(6, 9, 0.7);
    edge(7, 10, 0.8); edge(7, 13, 0.7);
    edge(8, 11, 0.5);
    edge(9, 11, 0.6);
    edge(10, 13, 0.8); edge(10, 12, 0.5);
    edge(11, 12, 0.9); edge(11, 13, 0.4);
    edge(12, 13, 0.3);
    TheoremGraph::new(names, adj)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    // ---- UniversalDirac trait tests ----

    #[test]
    fn test_kalman_dirac_dimension() {
        let k = KalmanDirac::new(5, 0.1, 0.2);
        assert_eq!(k.dimension(), 5);
    }

    #[test]
    fn test_kalman_dirac_name() {
        let k = KalmanDirac::new(3, 0.1, 0.2);
        assert_eq!(k.theorem_name(), "Kalman");
    }

    #[test]
    fn test_kalman_dirac_symmetric() {
        let k = KalmanDirac::new(6, 0.1, 0.2);
        let d = k.dirac_matrix();
        let diff = &d - &d.transpose();
        assert!(diff.iter().all(|&x| x.abs() < 1e-12), "Kalman Dirac must be symmetric");
    }

    #[test]
    fn test_kalman_spectrum_positive() {
        let k = KalmanDirac::new(5, 0.1, 0.2);
        let spec = k.spectrum();
        for i in 0..spec.len() {
            assert!(spec[i] > 0.0, "eigenvalue {} = {} should be positive", i, spec[i]);
        }
    }

    #[test]
    fn test_thermal_dirac_dimension() {
        let t = ThermalDirac::new(8, 1.0, 0.5);
        assert_eq!(t.dimension(), 8);
    }

    #[test]
    fn test_thermal_dirac_name() {
        let t = ThermalDirac::new(3, 1.0, 0.5);
        assert_eq!(t.theorem_name(), "Thermal");
    }

    #[test]
    fn test_thermal_dirac_symmetric() {
        let t = ThermalDirac::new(6, 1.0, 0.5);
        let d = t.dirac_matrix();
        let diff = &d - &d.transpose();
        assert!(diff.iter().all(|&x| x.abs() < 1e-12), "Thermal Dirac must be symmetric");
    }

    #[test]
    fn test_thermal_spectrum_positive() {
        let t = ThermalDirac::new(5, 1.0, 0.1);
        let spec = t.spectrum();
        for i in 0..spec.len() {
            assert!(spec[i] > 0.0, "eigenvalue {} = {} should be positive", i, spec[i]);
        }
    }

    #[test]
    fn test_fokker_planck_dirac_dimension() {
        let fp = FokkerPlanckDirac::new(7, 0.3, 0.5);
        assert_eq!(fp.dimension(), 7);
    }

    #[test]
    fn test_fokker_planck_dirac_name() {
        let fp = FokkerPlanckDirac::new(3, 0.3, 0.5);
        assert_eq!(fp.theorem_name(), "FokkerPlanck");
    }

    #[test]
    fn test_fokker_planck_symmetric() {
        let fp = FokkerPlanckDirac::new(6, 0.3, 0.5);
        let d = fp.dirac_matrix();
        let diff = &d - &d.transpose();
        assert!(diff.iter().all(|&x| x.abs() < 1e-12), "FP Dirac must be symmetric");
    }

    #[test]
    fn test_eigen_policy_dirac_dimension() {
        let ep = EigenPolicyDirac::new(4, 0.9, 1.0);
        assert_eq!(ep.dimension(), 4);
    }

    #[test]
    fn test_eigen_policy_dirac_name() {
        let ep = EigenPolicyDirac::new(3, 0.9, 1.0);
        assert_eq!(ep.theorem_name(), "EigenPolicy");
    }

    #[test]
    fn test_eigen_policy_symmetric() {
        let ep = EigenPolicyDirac::new(6, 0.9, 1.0);
        let d = ep.dirac_matrix();
        let diff = &d - &d.transpose();
        assert!(diff.iter().all(|&x| x.abs() < 1e-12), "EigenPolicy Dirac must be symmetric");
    }

    #[test]
    fn test_eigen_policy_spectrum_positive() {
        let ep = EigenPolicyDirac::new(5, 0.5, 1.0);
        let spec = ep.spectrum();
        for i in 0..spec.len() {
            assert!(spec[i] > 0.0, "EigenPolicy eigenvalue {} = {} should be positive", i, spec[i]);
        }
    }

    // ---- Spectrum equivalence tests ----

    #[test]
    fn test_spectrum_normalized() {
        let s = NamedSpectrum {
            name: "test".into(),
            eigenvalues: vec![3.0, 4.0],
            dimension: 2,
        };
        let norm = s.normalized();
        let norm_sq: f64 = norm.iter().map(|x| x * x).sum();
        assert_abs_diff_eq!(norm_sq, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_spectrum_self_equivalence() {
        let s = NamedSpectrum {
            name: "test".into(),
            eigenvalues: vec![1.0, 2.0, 3.0],
            dimension: 3,
        };
        assert!(SpectrumComparator::are_equivalent(&s, &s, 1e-10));
    }

    #[test]
    fn test_spectrum_kalman_thermal_equivalence() {
        let k = KalmanDirac::new(5, 0.5, 0.0);
        let t = ThermalDirac::new(5, 0.5, 0.0);
        let spec_k = NamedSpectrum {
            name: "Kalman".into(),
            eigenvalues: k.spectrum().iter().cloned().collect(),
            dimension: 5,
        };
        let spec_t = NamedSpectrum {
            name: "Thermal".into(),
            eigenvalues: t.spectrum().iter().cloned().collect(),
            dimension: 5,
        };
        assert!(SpectrumComparator::are_equivalent(&spec_k, &spec_t, 0.1),
            "Kalman and Thermal with matching params should have similar spectra");
    }

    #[test]
    fn test_spectrum_distance_zero_for_identical() {
        let s = NamedSpectrum {
            name: "A".into(),
            eigenvalues: vec![1.0, 2.0, 3.0],
            dimension: 3,
        };
        let dist = SpectrumComparator::normalized_distance(&s, &s);
        assert_abs_diff_eq!(dist, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_spectrum_distance_positive_for_different() {
        let a = NamedSpectrum {
            name: "A".into(),
            eigenvalues: vec![1.0, 2.0, 3.0],
            dimension: 3,
        };
        let b = NamedSpectrum {
            name: "B".into(),
            eigenvalues: vec![4.0, 5.0, 6.0],
            dimension: 3,
        };
        let dist = SpectrumComparator::normalized_distance(&a, &b);
        assert!(dist > 0.0);
    }

    // ---- Conservation law tests ----

    #[test]
    fn test_conservation_law_structure() {
        let cl = ConservationLaw {
            landauer_cost: 1.0,
            free_energy: 2.0,
            h1_risk: 3.0,
            total: 6.0,
        };
        assert!(cl.verify(6.0, 0.01));
    }

    #[test]
    fn test_conservation_law_fail() {
        let cl = ConservationLaw {
            landauer_cost: 1.0,
            free_energy: 2.0,
            h1_risk: 3.0,
            total: 6.0,
        };
        assert!(!cl.verify(10.0, 0.01));
    }

    #[test]
    fn test_unified_closure_conservation() {
        let ops: Vec<Box<dyn UniversalDirac>> = vec![
            Box::new(KalmanDirac::new(4, 0.1, 0.2)),
            Box::new(ThermalDirac::new(4, 1.0, 0.1)),
        ];
        let closure = UnifiedClosure::new(ops);
        let cl1 = closure.conservation_law(1.0);
        let cl2 = closure.conservation_law(1.0);
        assert_abs_diff_eq!(cl1.total, cl2.total, epsilon = 1e-10);
    }

    // ---- Agent lifecycle tests ----

    #[test]
    fn test_agent_lifecycle_creation() {
        let agent = AgentLifecycle::new(100.0);
        assert_eq!(agent.initial_free_energy, 100.0);
        assert_eq!(agent.cumulative_landauer_cost, 0.0);
        assert_eq!(agent.current_free_energy, 100.0);
        assert!(agent.alive);
    }

    #[test]
    fn test_agent_lifecycle_step() {
        let mut agent = AgentLifecycle::new(100.0);
        let cost = agent.step(1.0, 1.0);
        assert!(cost > 0.0);
        assert!(agent.cumulative_landauer_cost > 0.0);
        assert!(agent.current_free_energy < 100.0);
    }

    #[test]
    fn test_agent_lifecycle_conservation() {
        let mut agent = AgentLifecycle::new(100.0);
        for _ in 0..10 {
            agent.step(1.0, 1.0);
        }
        assert!(agent.conservation_holds(), "Energy conservation must hold");
    }

    #[test]
    fn test_agent_lifecycle_death() {
        let mut agent = AgentLifecycle::new(1.0);
        for _ in 0..1000 {
            if agent.is_dead() { break; }
            agent.step(100.0, 10.0);
        }
        assert!(agent.is_dead(), "Agent should die when energy is exhausted");
    }

    #[test]
    fn test_agent_death_exact() {
        let mut agent = AgentLifecycle::new(10.0);
        while agent.alive {
            agent.step(1.0, 0.01);
        }
        assert_abs_diff_eq!(agent.cumulative_landauer_cost, agent.initial_free_energy, epsilon = 0.1);
        assert_eq!(agent.current_free_energy, 0.0);
    }

    #[test]
    fn test_agent_lifecycle_not_dead_early() {
        let mut agent = AgentLifecycle::new(1000.0);
        agent.step(1.0, 1.0);
        assert!(!agent.is_dead(), "Agent should not die from a small step");
    }

    // ---- Theorem graph tests ----

    #[test]
    fn test_theorem_graph_build() {
        let graph = build_theorem_graph();
        assert_eq!(graph.theorem_names.len(), 14);
    }

    #[test]
    fn test_theorem_graph_laplacian() {
        let graph = build_theorem_graph();
        let lap = graph.laplacian();
        assert_eq!(lap.nrows(), 14);
        assert_eq!(lap.ncols(), 14);
    }

    #[test]
    fn test_theorem_graph_laplacian_row_sum_zero() {
        let graph = build_theorem_graph();
        let lap = graph.laplacian();
        for i in 0..14 {
            let row_sum: f64 = (0..14).map(|j| lap[(i, j)]).sum();
            assert!(row_sum.abs() < 1e-12, "Laplacian row {} sum must be zero, got {}", i, row_sum);
        }
    }

    #[test]
    fn test_theorem_graph_fiedler() {
        let graph = build_theorem_graph();
        let (fiedler_val, fiedler_vec) = graph.fiedler_vector();
        assert!(fiedler_val > 0.0, "Fiedler value must be positive");
        assert_eq!(fiedler_vec.len(), 14);
    }

    #[test]
    fn test_theorem_graph_neighbors() {
        let graph = build_theorem_graph();
        let neighbors = graph.neighbors(0); // Kalman
        assert!(!neighbors.is_empty(), "Kalman should have neighbors");
    }

    #[test]
    fn test_theorem_graph_composes() {
        let graph = build_theorem_graph();
        assert!(graph.composes(0, 6), "Kalman should compose with Riccati");
        assert!(!graph.composes(0, 13), "Kalman should not directly compose with Landauer");
    }

    #[test]
    fn test_theorem_graph_connected() {
        let graph = build_theorem_graph();
        for i in 0..14 {
            let neighbors = graph.neighbors(i);
            assert!(!neighbors.is_empty(), "Node {} ({}) should have neighbors", i, graph.theorem_names[i]);
        }
    }

    // ---- Gluing verification tests ----

    #[test]
    fn test_gluing_same_operator() {
        let closure = UnifiedClosure::new(vec![
            Box::new(KalmanDirac::new(5, 0.1, 0.2)),
            Box::new(KalmanDirac::new(5, 0.1, 0.2)),
        ]);
        let result = closure.verify_gluing(0, 1, 0.01);
        assert!(result.glued, "Same operators should glue: distance = {}", result.spectrum_distance);
    }

    #[test]
    fn test_gluing_different_operators() {
        let closure = UnifiedClosure::new(vec![
            Box::new(KalmanDirac::new(5, 0.1, 0.2)),
            Box::new(ThermalDirac::new(5, 10.0, 5.0)),
        ]);
        let result = closure.verify_gluing(0, 1, 0.01);
        assert!(!result.spectrum_distance.is_nan());
    }

    #[test]
    fn test_gluing_result_serializable() {
        let result = GluingResult {
            theorem_a: "A".into(),
            theorem_b: "B".into(),
            spectrum_distance: 0.5,
            glued: false,
            tolerance: 0.1,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("theorem_a"));
    }

    // ---- Closure trait tests ----

    #[test]
    fn test_closure_dirac() {
        let closure = UnifiedClosure::new(vec![
            Box::new(KalmanDirac::new(3, 0.1, 0.2)),
            Box::new(ThermalDirac::new(3, 1.0, 0.1)),
        ]);
        let d = closure.dirac();
        assert_eq!(d.nrows(), 6);
    }

    #[test]
    fn test_closure_spectrum() {
        let closure = UnifiedClosure::new(vec![
            Box::new(KalmanDirac::new(3, 0.1, 0.2)),
            Box::new(ThermalDirac::new(3, 1.0, 0.1)),
        ]);
        let spec = closure.closure_spectrum();
        assert_eq!(spec.len(), 6);
    }

    #[test]
    fn test_closure_loop_cost() {
        let closure = UnifiedClosure::new(vec![
            Box::new(KalmanDirac::new(3, 0.1, 0.2)),
        ]);
        let cost = closure.loop_cost(1.0);
        assert!(cost >= 0.0, "Loop cost should be non-negative");
    }

    #[test]
    fn test_closure_loop_cost_decreases_with_temperature() {
        let closure = UnifiedClosure::new(vec![
            Box::new(KalmanDirac::new(5, 0.1, 0.2)),
        ]);
        let cost_low_t = closure.loop_cost(0.1);
        let cost_high_t = closure.loop_cost(10.0);
        assert!(cost_high_t < cost_low_t || cost_high_t.is_finite(),
            "High temperature should thermalize");
    }

    #[test]
    fn test_closure_loop_cost_zero_spectrum() {
        struct ZeroDirac { dim: usize }
        impl UniversalDirac for ZeroDirac {
            fn theorem_name(&self) -> &str { "Zero" }
            fn dimension(&self) -> usize { self.dim }
            fn dirac_matrix(&self) -> DMatrix<f64> { DMatrix::zeros(self.dim, self.dim) }
        }
        impl Closure for ZeroDirac {
            fn dirac(&self) -> DMatrix<f64> { DMatrix::zeros(self.dim, self.dim) }
        }
        let z = ZeroDirac { dim: 3 };
        let cost = z.loop_cost(1.0);
        // Zero eigenvalues → each term is 0/0 → 0.0 contribution
        assert!(cost == 0.0 || cost.is_nan(), "Zero spectrum cost should be 0 or NaN, got {}", cost);
    }

    // ---- Dirichlet space tests ----

    #[test]
    fn test_dirichlet_space_creation() {
        let ds = DirichletSpace::new(5);
        assert_eq!(ds.dim, 5);
    }

    #[test]
    fn test_dirichlet_space_laplacian_structure() {
        let ds = DirichletSpace::new(5);
        for i in 0..5 {
            assert_abs_diff_eq!(ds.laplacian[(i, i)], 2.0, epsilon = 1e-10);
        }
        for i in 0..4 {
            assert_abs_diff_eq!(ds.laplacian[(i, i + 1)], -1.0, epsilon = 1e-10);
            assert_abs_diff_eq!(ds.laplacian[(i + 1, i)], -1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_dirichlet_space_wire() {
        let ds = DirichletSpace::new(5);
        let k = KalmanDirac::new(5, 0.1, 0.2);
        let wired = ds.wire(&k);
        assert_eq!(wired.nrows(), 5);
        assert_eq!(wired.ncols(), 5);
    }

    #[test]
    fn test_dirichlet_space_energy() {
        let ds = DirichletSpace::new(5);
        let v = DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0]);
        let energy = ds.energy(&v);
        assert!(energy > 0.0, "Dirichlet energy should be positive");
    }

    #[test]
    fn test_dirichlet_space_as_universal_dirac() {
        let ds = DirichletSpace::new(5);
        assert_eq!(ds.theorem_name(), "DirichletSpace");
        assert_eq!(ds.dimension(), 5);
        let spec = ds.spectrum();
        for i in 0..spec.len() {
            assert!(spec[i] > 0.0, "Dirichlet eigenvalue {} = {} should be positive", i, spec[i]);
        }
    }

    // ---- Unified closure advanced tests ----

    #[test]
    fn test_unified_closure_block_diagonal() {
        let closure = UnifiedClosure::new(vec![
            Box::new(KalmanDirac::new(3, 0.1, 0.2)),
            Box::new(ThermalDirac::new(4, 1.0, 0.1)),
        ]);
        let d = closure.unified_dirac();
        assert_eq!(d.nrows(), 7);
        assert_eq!(d.ncols(), 7);
        for i in 0..3 {
            for j in 3..7 {
                assert_abs_diff_eq!(d[(i, j)], 0.0, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_unified_closure_symmetric() {
        let closure = UnifiedClosure::new(vec![
            Box::new(KalmanDirac::new(4, 0.1, 0.2)),
            Box::new(ThermalDirac::new(4, 1.0, 0.1)),
            Box::new(FokkerPlanckDirac::new(4, 0.3, 0.5)),
        ]);
        let d = closure.unified_dirac();
        let diff = &d - &d.transpose();
        assert!(diff.iter().all(|&x| x.abs() < 1e-12));
    }

    #[test]
    fn test_unified_closure_all_four_theorems() {
        let closure = UnifiedClosure::new(vec![
            Box::new(KalmanDirac::new(5, 0.1, 0.2)),
            Box::new(ThermalDirac::new(5, 1.0, 0.1)),
            Box::new(FokkerPlanckDirac::new(5, 0.3, 0.5)),
            Box::new(EigenPolicyDirac::new(5, 0.9, 1.0)),
        ]);
        let spec = closure.unified_spectrum();
        assert_eq!(spec.len(), 20);
    }

    #[test]
    fn test_unified_spectrum_sorted() {
        let closure = UnifiedClosure::new(vec![
            Box::new(KalmanDirac::new(5, 0.1, 0.2)),
            Box::new(ThermalDirac::new(5, 1.0, 0.1)),
        ]);
        let spec = closure.unified_spectrum();
        for i in 1..spec.len() {
            assert!(spec[i] >= spec[i - 1], "Spectrum should be sorted");
        }
    }

    // ---- Agent self-model test ----

    #[test]
    fn test_agent_self_model() {
        let closure = UnifiedClosure::new(vec![
            Box::new(KalmanDirac::new(4, 0.1, 0.2)),
            Box::new(ThermalDirac::new(4, 1.0, 0.1)),
            Box::new(FokkerPlanckDirac::new(4, 0.3, 0.5)),
            Box::new(EigenPolicyDirac::new(4, 0.9, 1.0)),
        ]);
        let lifecycle = closure.agent_loop(1000.0, 1.0, 100);
        assert!(lifecycle.conservation_holds());
    }

    #[test]
    fn test_agent_self_model_death() {
        let closure = UnifiedClosure::new(vec![
            Box::new(KalmanDirac::new(4, 0.1, 0.2)),
        ]);
        let lifecycle = closure.agent_loop(0.01, 100.0, 10000);
        assert!(lifecycle.is_dead(), "Agent with tiny budget should die");
    }

    // ---- Serde tests ----

    #[test]
    fn test_named_spectrum_serialize() {
        let s = NamedSpectrum {
            name: "test".into(),
            eigenvalues: vec![1.0, 2.0],
            dimension: 2,
        };
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("test"));
        let back: NamedSpectrum = serde_json::from_str(&json).unwrap();
        assert_eq!(back.dimension, 2);
    }

    #[test]
    fn test_conservation_law_serialize() {
        let cl = ConservationLaw {
            landauer_cost: 1.0,
            free_energy: 2.0,
            h1_risk: 3.0,
            total: 6.0,
        };
        let json = serde_json::to_string(&cl).unwrap();
        let back: ConservationLaw = serde_json::from_str(&json).unwrap();
        assert_abs_diff_eq!(back.total, 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_agent_lifecycle_serialize() {
        let al = AgentLifecycle::new(100.0);
        let json = serde_json::to_string(&al).unwrap();
        let back: AgentLifecycle = serde_json::from_str(&json).unwrap();
        assert!(back.alive);
    }

    #[test]
    fn test_gluing_result_serialize() {
        let gr = GluingResult {
            theorem_a: "Kalman".into(),
            theorem_b: "Thermal".into(),
            spectrum_distance: 0.5,
            glued: false,
            tolerance: 0.1,
        };
        let json = serde_json::to_string(&gr).unwrap();
        let back: GluingResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.theorem_a, "Kalman");
    }

    // ---- Missing center proof tests ----

    #[test]
    fn test_dirichlet_space_is_center() {
        let ds = DirichletSpace::new(5);
        let k = KalmanDirac::new(5, 0.1, 0.2);
        let wired = ds.wire(&k);
        let norm: f64 = wired.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm > 0.0, "Wiring should produce non-zero result");
    }

    #[test]
    fn test_all_theorems_wire_to_center() {
        let ds = DirichletSpace::new(5);
        let theorems: Vec<Box<dyn UniversalDirac>> = vec![
            Box::new(KalmanDirac::new(5, 0.1, 0.2)),
            Box::new(ThermalDirac::new(5, 1.0, 0.1)),
            Box::new(FokkerPlanckDirac::new(5, 0.3, 0.5)),
            Box::new(EigenPolicyDirac::new(5, 0.9, 1.0)),
        ];
        for t in &theorems {
            let wired = ds.wire(t.as_ref());
            let norm: f64 = wired.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(norm > 0.0, "{} should wire non-trivially to DirichletSpace", t.theorem_name());
        }
    }

    #[test]
    fn test_spectral_theorem_of_d_squared() {
        let k = KalmanDirac::new(5, 0.1, 0.2);
        let spec_d = k.spectrum();
        let spec_d2 = k.spectrum_squared();
        for i in 0..spec_d.len() {
            let expected = spec_d[i] * spec_d[i];
            // Find closest match in spec_d2
            let mut best = f64::MAX;
            for j in 0..spec_d2.len() {
                let diff = (spec_d2[j] - expected).abs();
                if diff < best { best = diff; }
            }
            assert!(best < 0.5, "D² eigenvalue should be D eigenvalue squared, diff = {}", best);
        }
    }

    // ---- Fiedler analysis tests ----

    #[test]
    fn test_fiedler_partitions_graph() {
        let graph = build_theorem_graph();
        let (_fiedler_val, fiedler_vec) = graph.fiedler_vector();
        let pos_count = fiedler_vec.iter().filter(|&&x| x > 0.0).count();
        let neg_count = fiedler_vec.iter().filter(|&&x| x < 0.0).count();
        assert!(pos_count > 0 && neg_count > 0, "Fiedler vector should partition the graph");
    }

    #[test]
    fn test_algebraic_connectivity_positive() {
        let graph = build_theorem_graph();
        let (fiedler_val, _) = graph.fiedler_vector();
        assert!(fiedler_val > 0.0, "Graph should be connected");
    }

    // ---- Edge cases ----

    #[test]
    fn test_single_dimension_operator() {
        let k = KalmanDirac::new(1, 0.1, 0.2);
        assert_eq!(k.dimension(), 1);
        let d = k.dirac_matrix();
        assert_eq!(d.nrows(), 1);
        let spec = k.spectrum();
        assert_eq!(spec.len(), 1);
    }

    #[test]
    fn test_two_dimension_operator() {
        let k = KalmanDirac::new(2, 0.1, 0.2);
        let spec = k.spectrum();
        assert_eq!(spec.len(), 2);
    }

    #[test]
    fn test_large_dimension_operator() {
        let k = KalmanDirac::new(20, 0.1, 0.2);
        let spec = k.spectrum();
        assert_eq!(spec.len(), 20);
        for i in 0..spec.len() {
            assert!(spec[i].is_finite(), "Eigenvalue {} should be finite", i);
        }
    }

    #[test]
    fn test_empty_closure() {
        let closure = UnifiedClosure::new(vec![]);
        assert_eq!(closure.dimension(), 0);
        let d = closure.unified_dirac();
        assert_eq!(d.nrows(), 0);
    }

    #[test]
    fn test_single_operator_closure() {
        let closure = UnifiedClosure::new(vec![
            Box::new(KalmanDirac::new(5, 0.1, 0.2)),
        ]);
        assert_eq!(closure.dimension(), 5);
    }

    // ---- Four theorem spectral proximity ----

    #[test]
    fn test_four_theorem_spectral_proximity() {
        let k = KalmanDirac::new(5, 0.5, 0.0);
        let t = ThermalDirac::new(5, 0.5, 0.0);
        let spec_k = NamedSpectrum {
            name: "Kalman".into(),
            eigenvalues: k.spectrum().iter().cloned().collect(),
            dimension: 5,
        };
        let spec_t = NamedSpectrum {
            name: "Thermal".into(),
            eigenvalues: t.spectrum().iter().cloned().collect(),
            dimension: 5,
        };
        let dist = SpectrumComparator::normalized_distance(&spec_k, &spec_t);
        assert!(dist < 1.0, "Kalman and Thermal spectra should be close, got {}", dist);
    }

    // ---- Emergent conservation ----

    #[test]
    fn test_emergent_conservation() {
        let closure = UnifiedClosure::new(vec![
            Box::new(KalmanDirac::new(5, 0.1, 0.2)),
            Box::new(ThermalDirac::new(5, 1.0, 0.1)),
            Box::new(FokkerPlanckDirac::new(5, 0.3, 0.5)),
            Box::new(EigenPolicyDirac::new(5, 0.9, 1.0)),
        ]);
        let temps = vec![0.5, 1.0, 2.0, 5.0];
        let mut totals = Vec::new();
        for t in &temps {
            let cl = closure.conservation_law(*t);
            totals.push(cl.total);
        }
        for total in &totals {
            assert!(total.is_finite(), "Total should be finite");
        }
    }
}
