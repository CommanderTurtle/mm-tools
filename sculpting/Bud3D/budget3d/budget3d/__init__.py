"""Budget3D: Idol and Idol-turbo (single image -> 3D Gaussian splats) at any token budget.

The package layout:

    octree      the token layouts: the Sobol-to-octree bijection, salience, exact-budget layouts
    upstream    Idol's unmodified code (see IDOL_ROOT)
    models/     Idol's denoiser on an adaptive token set, the added modules, Idol-turbo's experts, the frozen
                encoders, loading released and trained weights
    sampling/   the 50-step guided sampler, the 5-step Idol-turbo variant, optimised inference
    training/   the layout curriculum, the teacher rollouts, the loss, the data, what the two trainers share
    render/     a small Gaussian-splat renderer for galleries and evaluation

Importing ``budget3d`` is cheap; the submodules that need Idol's code import it on first use (see ``upstream``).
"""
__version__ = "1.0.0"
