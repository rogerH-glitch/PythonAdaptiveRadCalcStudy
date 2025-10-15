# Plotting Principles

## Core Rule: Plots are diagnostic truth, not decoration.

All 2D/3D drawings derive only from computed geometry and fields.

### What this means:

- **No coordinate tweaks** to match expectations; if a plot "looks wrong", the math/geometry is wrong
- **No visual massaging** to make plots "look better" or "more intuitive"
- **Direct reflection** of inputs and mathematical outputs
- **Heatmap star must match** the field actually plotted

### Why this matters:

- Plots are diagnostic tools for validating calculations
- Visual adjustments can hide bugs in the underlying mathematics
- Faithful representation enables accurate interpretation of results
- Consistency between computed values and visual representation is essential

### Implementation:

- All plotting functions must render computed geometry/fields as-is
- No scaling, offsetting, or "correction" of coordinates for visual appeal
- Peak markers must correspond exactly to computed peak locations
- Color scales must represent actual data values without adjustment

### Enforcement:

- Code banners in plotting modules remind developers of this principle
- Tests verify the principle is maintained
- Code reviews should check for any visual "adjustments" or "corrections"
