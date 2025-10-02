# Product Requirements Document (PRD)

**Document ID:** FSE-RAD-VIEWFACTOR-PRD-v0.5  
**Date:** October 2, 2025  
**Product:** Local Python Tool – Radiation View Factor Validation  
**Owner:** Fire Safety Engineer  
**Tech Support:** Roger (Software Engineer Role)  

---

## 1. Function Overview

### Purpose
Create a local Python-based tool to calculate **local peak view factors** between rectangular surfaces (emitter → receiver) under fire conditions.

### Definition: Local Peak View Factor
The maximum differential (point) view factor over the receiver surface (not the area-averaged view factor). For parallel, centre-aligned emitter/receiver configurations, the local peak occurs at the receiver centre.

### Methods Implemented

**Primary Method:**
- **Adaptive Integration (1AI)** per Walton (2002), NISTIR 6925

**Secondary Methods (Cross-checking):**
- Fixed grid integration
- Monte Carlo sampling  
- Analytical formulas (where applicable)

### Tool Purpose
Methodology validation and sanity checks only - not for production fire safety calculations.

---

## 2. Users & Use Cases

### Target Users
- **Primary:** Fire Safety Engineer (developer/validator)
- **Secondary:** Software engineers supporting fire safety applications

### Use Cases

1. **Validation:** Validate 1AI implementation against NISTIR 6925 benchmarks
2. **Comparison:** Compare numerical methods to understand accuracy vs. runtime trade-offs
3. **Quality Assurance:** Confirm calculated values before embedding into other fire safety projects
4. **Research:** Investigate convergence behavior and method limitations

---

## 3. Input Data Format

### Input Methods
- Command Line Interface (CLI)
- Configuration file (YAML/JSON)

### Required Inputs

| Parameter | Description | Units | Example |
|-----------|-------------|-------|---------|
| `emitter_width` | Emitter surface width | meters | 5.1 |
| `emitter_height` | Emitter surface height | meters | 2.1 |
| `setback_distance` | Separation between surfaces | meters | 1.0 |

### Optional Inputs

| Parameter | Description | Default | Units |
|-----------|-------------|---------|-------|
| `receiver_width` | Receiver width | `emitter_width` | meters |
| `receiver_height` | Receiver height | `emitter_height` | meters |
| `angle` | Rotation angle | 0 (parallel) | degrees |
| `occluder_*` | Obstruction parameters | None | various |

### Default Geometric Assumptions
- Surfaces face towards each other
- Centers are aligned
- Receiver dimensions = emitter dimensions (unless specified)

### CLI Example
```bash
python viewfactor.py --emitter 5.1 2.1 --receiver 5.1 2.1 --setback 1.0 \
  --method adaptive --rel-tol 3e-3 --max-depth 12
```

---

## 4. Output Requirements

### Primary Output
- **Local peak view factor** (receiver) from Adaptive Integration (1AI)

### Secondary Outputs
- Fixed grid view factor
- Monte Carlo view factor (with uncertainty estimate)
- Analytical reference (when applicable)

### Output Formats

1. **Console Printout**
   ```
   === View Factor Results ===
   Adaptive (1AI):    0.702740 ± 0.000210
   Fixed Grid:        0.702156 
   Monte Carlo:       0.703021 ± 0.014052
   Analytical:        0.702740 (reference)
   ```

2. **CSV Export** (`/results/viewfactor_results_YYYYMMDD_HHMMSS.csv`)
3. **Optional Visualization** (`/results/plots/flux_map_*.png`)

---

## 5. Functional Flow

### Execution Sequence

1. **Input Processing**
   - Parse CLI arguments or config file
   - Validate geometric parameters
   - Set method-specific parameters

2. **Computation Engine**
   - **Primary:** Adaptive Integration (1AI)
   - **Secondary:** Fixed Grid + Monte Carlo + Analytical (if available)

3. **Results Processing**
   - Compare methods and calculate differences
   - Generate summary statistics
   - Export results to `/results/`

4. **Optional Visualization**
   - Flux distribution maps
   - Convergence plots
   - Save to `/results/plots/`

### Error Handling Flow
```
Input Validation → Geometry Check → Method Execution → Result Validation → Output
     ↓                ↓                  ↓                ↓              ↓
   Error Exit      Error Exit       Timeout/Limit    NaN Check      Success
```

---

## 6. Performance & Tuning (Refinement Controls)

### Design Philosophy
Allow developer to trade accuracy vs. runtime. Provide sensible defaults that can be tightened or loosened based on requirements.

### 6.1 Adaptive Integration (1AI)

| Parameter | Description | Default | Location |
|-----------|-------------|---------|----------|
| `--rel-tol` | Relative tolerance (≈99.7% accuracy) | 3e-3 | `src/adaptive.py` |
| `--abs-tol` | Absolute tolerance | 1e-6 | `src/adaptive.py` |
| `--max-depth` | Maximum recursion depth | 12 | `src/adaptive.py` |
| `--min-cell-area-frac` | Min sub-rect area fraction | 1e-8 | `src/adaptive.py` |
| `--max-cells` | Maximum number of sub-cells | 200,000 | `src/adaptive.py` |
| `--time-limit-s` | Wall-clock time limit | 60 | `src/adaptive.py` |
| `--init-grid` | Initial partition | 4×4 | `src/adaptive.py` |

**Performance Tuning:**
- **Lower Cost:** Relax `--rel-tol` (e.g., 5e-2 ≈ 95%), reduce `--max-depth`/`--max-cells`
- **Higher Accuracy:** Tighten `--rel-tol`, increase `--max-depth`/`--max-cells`

### 6.2 Fixed Grid

| Parameter | Description | Default | Location |
|-----------|-------------|---------|----------|
| `--grid-nx` | Grid points in x-direction | 100 | `src/fixed_grid.py` |
| `--grid-ny` | Grid points in y-direction | 100 | `src/fixed_grid.py` |
| `--quadrature` | Integration method | centroid | `src/fixed_grid.py` |
| `--time-limit-s` | Time limit | 60 | `src/fixed_grid.py` |

**Quadrature Options:** `centroid` | `2x2`

### 6.3 Monte Carlo

| Parameter | Description | Default | Location |
|-----------|-------------|---------|----------|
| `--samples` | Total ray samples | 200,000 | `src/montecarlo.py` |
| `--target-rel-ci` | Relative half-width of 95% CI | 0.02 (2%) | `src/montecarlo.py` |
| `--max-iters` | Maximum batch iterations | 50 | `src/montecarlo.py` |
| `--seed` | RNG seed for reproducibility | 42 | `src/montecarlo.py` |
| `--time-limit-s` | Time limit | 60 | `src/montecarlo.py` |

**Note:** Only Adaptive Integration is required to meet ±0.3% target accuracy. Fixed Grid & Monte Carlo provide reference comparisons.

---

## 7. Acceptance Criteria

### Primary Requirements

1. **Accuracy Target**
   - Adaptive Integration (1AI) reproduces benchmark cases within **±0.3% error** (≈99.7% accuracy)
   - NISTIR 6925 validation cases match within tolerance

2. **Method Consistency**
   - Secondary methods show consistent trends
   - Exact ±0.3% accuracy not required for secondary methods unless explicitly tuned

3. **Technical Requirements**
   - Compatible with Python 3.12+
   - Dependencies: numpy, scipy, matplotlib, pytest
   - No infinite loops or unbounded memory growth
   - All runs terminate with clear status: `converged`, `reached_limits`, or `failed`

### Validation Benchmarks

**Must Pass:**
- NISTIR 6925 obstructed case (when obstruction capability enabled)
- User hand-calculation cases (unobstructed, parallel configurations)
- Symmetry and geometric consistency tests

---

## 8. Non-Convergence & Hang Safeguards

### Global Safeguards (All Methods)

1. **Timeout Protection**
   - Per-solve time limits: `--time-limit-s` (default: 60s)
   - Global execution timeout for batch runs

2. **Iteration Limits**
   - `--max-depth`, `--max-cells`, `--max-iters`, `--samples`
   - Hard caps prevent runaway calculations

3. **Early Stopping Conditions**
   - Stop when tolerance or confidence interval achieved
   - Stagnation detection: improvement < 10% of tolerance for N=5 consecutive steps

4. **Numerical Stability Guards**
   - Enforce positive dimensions and nonzero setback
   - Clamp `cos θ ≥ 0` for physical validity
   - Division-by-zero protection: `EPS = 1e-12`
   - Near-contact geometry handling via high-precision or analytical approximation

5. **Deterministic Behavior**
   - Fixed RNG seed for Monte Carlo (unless overridden)
   - Reproducible results for debugging

### Exit Status Reporting

```python
class CalculationStatus:
    CONVERGED = "converged"           # Met tolerance/CI target
    REACHED_LIMITS = "reached_limits" # Hit iteration/time/memory limits  
    FAILED = "failed"                 # NaN/Inf or other numerical failure
```

**Diagnostic Information:**
- Iterations completed
- Achieved tolerance/confidence interval
- Limits encountered (time, memory, iterations)
- Convergence history

### Unit Test Coverage

- **Smoke tests:** Basic functionality verification
- **Worst-case geometries:** Near-contact, extreme aspect ratios
- **Symmetry checks:** Geometric consistency validation
- **Boundary conditions:** Edge cases and degenerate inputs

---

## 9. Error Handling

### Error Categories & Responses

#### 9.1 Invalid Input
**Trigger:** Non-positive dimensions, zero/negative setback, invalid method selection

**Response:**
```
Error: Invalid geometry (non-positive dimensions or setback). Check inputs.
Exit Code: 1
```

**Action:** Immediate termination with clear diagnostic message

#### 9.2 Non-Convergence / Limits Reached
**Trigger:** Timeout, iteration limits, memory limits reached before convergence

**Response:**
```
Warning: Reached limits before tolerance. Result may be approximate.
Status: reached_limits
Achieved tolerance: 1.2e-2 (target: 3e-3)
Exit Code: 0 (with warning flag)
```

**Action:** Return best available result with uncertainty estimate

#### 9.3 Numerical Failure
**Trigger:** NaN/Inf results, matrix singularities, geometric degeneracies

**Response:**
```
Error: Calculation failed (NaN/Inf detected). Contact Roger.
Debug info: [method=adaptive, iteration=47, cell_area=1.2e-15]
Exit Code: 2
```

**Action:** Immediate termination with diagnostic information for debugging

### Logging Strategy

- **INFO:** Normal progress updates
- **WARNING:** Non-critical issues (limits reached, approximations used)
- **ERROR:** Critical failures requiring attention
- **DEBUG:** Detailed computational diagnostics (optional verbose mode)

---

## 10. Validation Test Cases

### 10.1 NISTIR 6925 – Analytic Obstructed Test

**Reference:** Walton (2002), NISTIR 6925, Table 1 & Figure 13  
**URL:** nvlpubs.nist.gov

#### Geometry Configuration
- **Surface 1:** 1.0 × 1.0 m square (emitter)
- **Surface 2:** 1.0 × 1.0 m square (receiver), 1.0 m separation
- **Surface 3 & 4:** 0.5 × 0.5 m squares (occluders)
  - Back-to-back configuration
  - Centered on line between surfaces 1 & 2  
  - Located at 3/4 distance from surface 1 toward surface 2
- **Visibility:** Surface 1 visible from 3 only; Surface 2 visible from 4 only

#### Reference Values
- **Unobstructed** (surfaces 3 & 4 absent): F*₁,₂ = 0.19982490
- **Obstructed** (with surfaces 3 & 4): F₁,₂ = 0.11562061

#### Adaptive Integration Validation Results

| Tolerance (ε) | Calculated F₁,₂ | Error | Points | Direction |
|---------------|-----------------|-------|--------|-----------|
| 1e-3 | 0.11563653 | +0.00001592 | 25 | 1→2 |
| 1e-4 | 0.11562055 | -0.00000006 | 125 | 1→2 |
| 1e-3 | 0.11473675 | -0.00088386 | 525 | 2→1 |
| 1e-4 | 0.11526465 | -0.00035596 | 925 | 2→1 |
| 1e-5 | 0.11553235 | -0.00008826 | 2925 | 2→1 |
| 1e-6 | 0.11560305 | -0.00001756 | 8125 | 2→1 |
| 1e-7 | 0.11561626 | -0.00000435 | 18525 | 2→1 |

**Implementation Note:** Mark as `@pytest.mark.skip` until occluder functionality is implemented.

### 10.2 User Hand-Calculation Cases (Unobstructed)

**Configuration:** Parallel surfaces, centers aligned, facing each other

| Emitter (W×H) | Receiver (W×H) | Setback | Angle | Expected F | Test ID |
|---------------|----------------|---------|-------|------------|---------|
| 5.1 × 2.1 | 5.1 × 2.1 | 0.05 m | 0° | 0.998805 | UC-001 |
| 5.1 × 2.1 | 5.1 × 2.1 | 1.0 m | 0° | 0.70274 | UC-002 |
| 5.0 × 2.0 | 5.0 × 2.0 | 3.8 m | 0° | 0.17735 | UC-003 |
| 20.02 × 1.05 | 20.02 × 1.05 | 0.81 m | 0° | 0.54375 | UC-004 |
| 20.02 × 1.05 | 20.02 × 1.05 | 1.8 m | 0° | 0.27931 | UC-005 |
| 21 × 1 | 21 × 1 | 3.67 m | 0° | 0.13285 | UC-006 |

#### Acceptance Criteria
- **Adaptive method:** Within ±0.3% (≈99.7% accuracy) of reference values
- **Fixed Grid & Monte Carlo:** Consistent trends (looser tolerance unless explicitly tuned)

### 10.3 Additional Test Cases

#### Geometric Consistency Tests
- **Symmetry:** F₁₋₂ calculations should be consistent regardless of coordinate system orientation
- **Reciprocity:** A₁F₁₋₂ = A₂F₂₋₁ for unobstructed cases
- **Conservation:** Sum of view factors from closed enclosure = 1.0

#### Edge Cases
- **Near-contact:** Setback → 0 (should approach analytical limits)
- **Far-field:** Large setback distances (should approach point-source behavior)
- **Extreme aspect ratios:** Very wide or very tall surfaces

---

## 11. Glossary

### Core Concepts

**View Factor (F):** The fraction of radiation leaving an emitter surface that directly reaches a receiver surface, accounting for geometric configuration and orientation.

**Local Peak View Factor:** The maximum pointwise (differential) view factor occurring anywhere on the receiver surface. For parallel, center-aligned configurations, this typically occurs at the receiver center.

**Area-Averaged View Factor:** The integral average of pointwise view factors over the entire receiver surface area.

### Computational Methods

**Adaptive Integration (1AI):** A numerical method that recursively refines the emitter surface subdivision until a specified error tolerance is met. Based on Walton (2002) NISTIR 6925.

**Fixed Grid Integration:** Uniform partitioning of the emitter surface with regular quadrature points. Accuracy scales with grid density (O(N²) for N×N grid).

**Monte Carlo Method:** Statistical sampling approach using random ray tracing. Accuracy scales as O(1/√N) where N is the number of samples.

**Analytical Solution:** Closed-form mathematical expressions available for simplified geometries (e.g., parallel rectangles, point sources).

### Technical Terms

**Setback Distance:** The perpendicular separation between emitter and receiver surfaces.

**Occluder/Obstruction:** Intermediate surfaces that block direct line-of-sight between emitter and receiver.

**Convergence Tolerance:** The maximum acceptable relative error between successive iterations before terminating the adaptive algorithm.

**Stagnation Detection:** Algorithm safeguard that stops iteration when improvement falls below a threshold for consecutive steps.

### Fire Safety Context

**Emitter Surface:** The fire source or heated surface radiating thermal energy.

**Receiver Surface:** The target surface receiving thermal radiation (e.g., building facade, structural element).

**Thermal Radiation:** Electromagnetic energy transfer in the infrared spectrum, following Stefan-Boltzmann law and geometric view factor relationships.

---

## Appendices

### A. Implementation Priority

**Phase 1 (MVP):**
- Adaptive Integration (1AI) - unobstructed cases
- Fixed Grid method
- Basic CLI interface
- Core validation test cases (UC-001 through UC-006)

**Phase 2 (Enhanced):**
- Monte Carlo method
- Configuration file input (YAML/JSON)
- Results visualization and plotting
- Performance optimization

**Phase 3 (Advanced):**
- Occluder/obstruction support
- NISTIR 6925 obstructed validation
- Parallel processing capabilities
- Advanced geometric configurations

### B. Dependencies & Environment

**Required Python Packages:**
```
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
pytest>=7.0.0
pyyaml>=6.0.0  # for config file support
```

**Development Tools:**
```
black>=23.0.0      # code formatting
flake8>=6.0.0      # linting
mypy>=1.0.0        # type checking
pytest-cov>=4.0.0 # coverage reporting
```

### C. File Structure

```
viewfactor_tool/
├── README.md
├── requirements.txt
├── setup.py
├── viewfactor.py              # Main CLI entry point
├── src/
│   ├── __init__.py
│   ├── adaptive.py            # Adaptive Integration (1AI)
│   ├── fixed_grid.py          # Fixed grid method
│   ├── montecarlo.py          # Monte Carlo method
│   ├── analytical.py          # Analytical solutions
│   ├── geometry.py            # Geometric utilities
│   ├── validation.py          # Input validation
│   └── utils.py               # Common utilities
├── tests/
│   ├── test_adaptive.py
│   ├── test_fixed_grid.py
│   ├── test_montecarlo.py
│   ├── test_validation_cases.py
│   └── conftest.py
├── results/                   # Output directory
│   └── plots/                 # Visualization outputs
└── docs/
    ├── API.md
    ├── VALIDATION.md
    └── PERFORMANCE.md
```

---

**Document Status:** Draft v0.5  
**Next Review:** Upon implementation milestone completion  
**Approval Required:** Fire Safety Engineer (Owner)
