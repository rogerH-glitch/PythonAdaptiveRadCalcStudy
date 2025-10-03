# Radiation View Factor Validation Tool

A local Python tool for calculating **local peak view factors** between rectangular surfaces under fire conditions. This tool implements multiple numerical methods for validation and comparison purposes.

## Overview

This tool calculates the maximum differential (point) view factor over receiver surfaces, which is critical for fire safety engineering applications. For parallel, centre-aligned emitter/receiver configurations, the local peak typically occurs at the receiver centre.

## Default Geometric Assumptions

- **Surface Orientation**: Emitter and receiver face towards each other
- **Alignment**: Surface centres are aligned along the separation axis  
- **Receiver Dimensions**: Default to emitter dimensions unless explicitly specified
- **Coordinate System**: Right-hand coordinate system with standard orientations

## Features

### Calculation Methods

1. **Adaptive Integration (1AI)** - Primary method per NISTIR 6925 (Walton, 2002)
2. **Fixed Grid Integration** - Uniform subdivision with regular quadrature
3. **Monte Carlo Sampling** - Statistical ray tracing with uncertainty estimation
4. **Analytical Solutions** - Closed-form expressions where available

### Key Capabilities

- **High Accuracy**: Adaptive method targets ±0.3% accuracy (99.7% confidence)
- **Performance**: Single calculations complete within 3 seconds
- **Validation**: Built-in comparison against NISTIR 6925 benchmarks
- **Flexibility**: Multiple output formats (console, CSV, JSON, YAML)
- **Safety-Critical**: Comprehensive validation and error checking

## Quickstart (Windows / PowerShell)

```powershell
# One-time per machine:
.\scripts\setup.ps1

# Run tests:
python -m pytest -q

# Run validation suite (CSV + plots into ./results):
python main.py --cases DOCS/validation_cases.yaml --outdir results --plot
```

## Installation

### Prerequisites

- Python 3.12 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

- `numpy` - Numerical computations
- `scipy` - Scientific computing utilities  
- `matplotlib` - Plotting and visualisation
- `pyyaml` - YAML configuration support

### Development Dependencies

```bash
pip install -r requirements-dev.txt
```

- `pytest` - Testing framework
- `coverage` - Code coverage analysis

### Basic Usage

```bash
# Calculate view factor for 5.1m × 2.1m surfaces with 1.0m setback
python main.py --emitter 5.1 2.1 --setback 1.0 --method adaptive

# Compare all methods
python main.py --emitter 20.02 1.05 --setback 0.81 --method all

# Different receiver size
python main.py --emitter 5.1 2.1 --receiver 4.0 1.8 --setback 2.0
```

### Configuration File

```bash
# Use YAML configuration
python main.py --config config.yaml --output results.csv
```

### Method-Specific Parameters

```bash
# Adaptive method with custom tolerance
python main.py --emitter 5.1 2.1 --setback 1.0 --method adaptive --rel-tol 1e-4

# Fixed grid with higher resolution
python main.py --emitter 5.1 2.1 --setback 1.0 --method fixed_grid --grid-nx 200 --grid-ny 200

# Monte Carlo with more samples
python main.py --emitter 5.1 2.1 --setback 1.0 --method montecarlo --samples 500000
```

## Peak Locator System

### Overview

The tool includes an advanced **peak locator system** that finds the exact location on the receiver where the point view factor is maximized. This is crucial for general geometries where the peak may not occur at the receiver center.

### Key Insight

⚠️ **Important**: The local-peak view factor does NOT necessarily occur at the receiver center. This is only true for the special case of parallel, concentric (center-aligned) emitter/receiver with no occluders. For off-center offsets, rotations, aspect-ratio extremes, or occluders, the local peak shifts.

### Peak Search Modes

#### 1. Center Mode (Default)
```bash
# Fast path for concentric parallel cases
python main.py --method analytical --emitter 5.1 2.1 --setback 1.0 --rc-mode center
```
- **Use case**: Concentric parallel geometries
- **Performance**: Fastest (single evaluation)
- **Accuracy**: Exact for center-aligned cases

#### 2. Grid Mode
```bash
# Coarse grid sampling for exploration
python main.py --method analytical --emitter 5.1 2.1 --setback 1.0 --rc-mode grid --rc-grid-n 21
```
- **Use case**: Quick exploration of view factor distribution
- **Performance**: Fast (N×N evaluations)
- **Accuracy**: Limited by grid resolution

#### 3. Search Mode
```bash
# Full coarse-to-fine optimization
python main.py --method analytical --emitter 5.1 2.1 --setback 1.0 --rc-mode search --rc-grid-n 21 --rc-search-multistart 8
```
- **Use case**: General geometries, non-concentric cases
- **Performance**: Slower but most accurate
- **Accuracy**: Finds true local peak with high precision

### Peak Locator Parameters

```bash
# Grid resolution for coarse sampling
--rc-grid-n 21                    # Default: 21 (21×21 grid)

# Search optimization parameters
--rc-search-rel-tol 3e-3          # Relative improvement tolerance
--rc-search-max-iters 200         # Max local optimizer iterations
--rc-search-multistart 8          # Number of multi-start seeds
--rc-search-time-limit-s 10.0     # Wall clock time limit

# Analytical method grid density
--analytical-nx 240               # Emitter grid Nx
--analytical-ny 240               # Emitter grid Ny
```

### Peak Locator Output

The peak locator provides detailed information about the search process:

```
==================================================
CALCULATION RESULTS
==================================================
Method: Analytical
Local Peak View Factor: 0.70275158
Peak Location: (0.000, 0.000) m
RC Mode: search
Calculation Time: 0.526 seconds

Search Details:
  Evaluations: 121
  Search Time: 0.526 s
  Seeds Used: 8
==================================================
```

### Heatmap Visualization

When using `--plot` with grid or search modes, the tool generates heatmap visualizations:

```bash
# Generate heatmap showing view factor distribution
python main.py --method analytical --emitter 5.1 2.1 --setback 1.0 --rc-mode grid --plot
```

**Plot Features**:
- **Left panel**: Geometry overview showing emitter/receiver positions
- **Right panel**: View factor heatmap with peak location marked
- **Peak indicator**: Red star showing exact peak location
- **Color scale**: Viridis colormap for clear visualization

### Example Use Cases

#### Concentric Parallel (Peak at Center)
```bash
python main.py --method analytical --emitter 5.1 2.1 --receiver 5.1 2.1 --setback 1.0 --rc-mode search
# Result: Peak at (0.000, 0.000) - receiver center
```

#### Offset Receiver (Peak Shifts)
```bash
python main.py --method analytical --emitter 5.1 2.1 --receiver 3.0 1.5 --setback 1.0 --rc-mode search
# Result: Peak shifts toward the nearer edge
```

#### Different Methods
```bash
# All methods support peak location
python main.py --method adaptive --emitter 5.1 2.1 --setback 1.0 --rc-mode search
python main.py --method fixedgrid --emitter 5.1 2.1 --setback 1.0 --rc-mode search
python main.py --method montecarlo --emitter 5.1 2.1 --setback 1.0 --rc-mode search
```

### Analytical (Point) Evaluator

The analytical backend computes the **point** view factor by integrating the kernel over the emitter for a specified receiver point `(rx, ry)`.  
Defaults: parallel, concentric (angle=0). Use `--analytical-nx/--analytical-ny` to trade accuracy for time.  
When using `rc_mode=search`, the peak locator probes multiple `(x,y)` on the receiver with this point evaluator.

## Output Examples

### Console Output

```
=== View Factor Results ===
    adaptive: 0.702740 ± 0.000210 (1.234s) ✓
  fixed_grid: 0.702156 ± 0.000000 (0.856s) ✓
  montecarlo: 0.703021 ± 0.014052 (2.145s) ✓
  analytical: 0.702740 ± 0.000000 (0.001s) ✓

Method comparison:
  Mean value: 0.702664
  Max difference: 0.000584 (0.1%)
```

### CSV Output

```csv
timestamp,method,view_factor,uncertainty,converged,iterations,computation_time_s
2025-10-02T14:30:15,adaptive,0.70274000,0.00021000,True,156,1.234
2025-10-02T14:30:16,fixed_grid,0.70215600,0.00000000,True,10000,0.856
```

## Validation Test Cases

The tool includes validation against established benchmarks:

### NISTIR 6925 Reference Cases

- **Obstructed Case**: Unit squares with 0.5×0.5 occluders
- **Reference Value**: F₁,₂ = 0.11562061 (analytical)
- **Tolerance**: Within ±0.3% for adaptive method

### User Hand-Calculation Cases

| Emitter (W×H) | Setback | Expected F | Test ID |
|---------------|---------|------------|---------|
| 5.1 × 2.1 m   | 0.05 m  | 0.998805   | UC-001  |
| 5.1 × 2.1 m   | 1.0 m   | 0.70274    | UC-002  |
| 5.0 × 2.0 m   | 3.8 m   | 0.17735    | UC-003  |

## Testing

### Run All Tests

```bash
pytest
```

### Run Smoke Tests Only

```bash
pytest tests/test_smoke.py -v
```

### Expected Output

```
tests/test_smoke.py::test_import_all_modules PASSED
tests/test_smoke.py::test_rectangle_creation PASSED  
tests/test_smoke.py::test_view_factor_result_creation PASSED
tests/test_smoke.py::test_calculator_instantiation PASSED
tests/test_smoke.py::test_geometry_validation PASSED
```

## Project Structure

```
├── main.py                 # Main entry point
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── pyproject.toml        # Project configuration
├── src/                  # Source code
│   ├── __init__.py
│   ├── cli.py           # Command-line interface
│   ├── geometry.py      # Geometric utilities
│   ├── io_yaml.py       # YAML I/O utilities
│   ├── analytical.py    # Analytical solutions
│   ├── fixed_grid.py    # Fixed grid method
│   ├── adaptive.py      # Adaptive integration
│   └── montecarlo.py    # Monte Carlo method
├── tests/               # Test suite
│   ├── __init__.py
│   └── test_smoke.py    # Smoke tests
├── docs/                # Documentation
│   └── validation_cases.yaml
└── results/             # Output directory
    └── .gitkeep
```

## Fire Safety Engineering Context

### Applications

- **Building Separation Analysis**: Calculate radiation between facades
- **Fire Exposure Assessment**: Determine heat flux on target surfaces  
- **Code Compliance**: Validate separation distances per fire safety codes
- **Research**: Investigate radiation heat transfer in fire scenarios

### Physical Interpretation

View factors represent the geometric relationship between surfaces for radiation heat transfer:

- **F = 0**: No geometric view (surfaces don't "see" each other)
- **F = 1**: Complete view (rare, only for specific geometries)
- **Typical Range**: 0.01 to 0.8 for most fire safety applications

### Engineering Units

- **Dimensions**: Metres (m)
- **View Factors**: Dimensionless (0 ≤ F ≤ 1)
- **Heat Flux**: kW/m² (when combined with fire intensity)

## Performance Requirements

- **Single Calculation**: ≤ 3 seconds per case
- **Accuracy Target**: ±0.3% for adaptive method
- **Memory Usage**: Reasonable limits with safety bounds
- **Convergence**: Guaranteed termination with timeout protection

## Error Handling

The tool includes comprehensive error handling:

- **Geometry Validation**: Positive areas, reasonable dimensions
- **Numerical Stability**: Division-by-zero protection, NaN/Inf detection
- **Performance Limits**: Timeout protection, iteration caps
- **User Feedback**: Clear error messages with suggested fixes

## Contributing

When contributing to this project:

1. Follow the established code structure and naming conventions
2. Include comprehensive tests for new functionality
3. Validate against known solutions where possible
4. Consider fire safety engineering use cases
5. Maintain performance requirements and safety-critical reliability

## License

[Add appropriate license information]

---

**Note**: This tool is designed for validation and research purposes in fire safety engineering. Always validate results against established methods and consult with qualified fire protection engineers for critical applications.
