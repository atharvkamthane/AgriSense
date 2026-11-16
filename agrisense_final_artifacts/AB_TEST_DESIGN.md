# AgriSense Simulated A/B Testing Specification
## Overview
Transparent assumption-based scenario simulation comparing a generic across-the-board input reduction against a cluster-targeted input reduction strategy.

## Key Design Parameters
- **Control Group (Generic_Control)**: Uniform 5% reduction across both fertilizer and pesticide.
- **Treatment Group (Segment_Treatment)**: Targeted 20% reduction applied only to inputs where the segment median is in the 75th percentile.
- **Yield Assumption**: Held unchanged (0% uplift) to avoid unsupported yield increase claims.
- **Simulated Efficiency Gain**: -2.38% reduction in combined inputs per tonne of yield vs generic control.

## Methodological Note
The dataset represents aggregated historical observations (Crop-Season-State-Year), not individual farm/field experiments. Results illustrate optimization potential under defined agronomic scenarios rather than causal field trial measurements.
