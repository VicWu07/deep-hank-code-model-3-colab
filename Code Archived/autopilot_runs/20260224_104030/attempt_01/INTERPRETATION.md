# Diagnostics Interpretation Guide

## loss_curves.png
- Good: residual_mse and shape_loss trend down over training steps.
- Bad: flat or rising curves indicate weak progress or instability.
- Implication: fix training quality first before trusting downstream diagnostics.

## residual_heatmap_nlow.png / residual_heatmap_nhigh.png
- Good: most cells are cool with low log10(|R|).
- Bad: hot regions indicate local HJB residual violations in (a, z).
- Implication: target those regions with more training or different sampling.

## w_clearing.png
- Good: histogram mass concentrated near zero.
- Bad: long right tail means wage-clearing errors in sampled states.
- Implication: equilibrium price map may be inconsistent in parts of state space.

## policy_slices.png
- Good: W(a) and c(a) are smooth and generally increasing in a.
- Bad: non-monotone kinks or decreasing segments signal shape issues.
- Implication: revisit shape regularization and optimization settings.

## sim_paths.png / asset_hist.png
- Good: bounded aggregate paths, sensible moments, and plausible final asset distribution.
- Bad: exploding series, drift, or boundary-mass pileups imply dynamic instability.
- Implication: simulation diagnostics reflect whether learned policy generalizes in time.

Machine readers should use JSON artifacts and diagnostics.json only; do not parse image files.

<!-- amp-managed -->