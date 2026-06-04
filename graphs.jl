using Plots

# -----------------------------------------------------------------------
# Timing data: fill in your measured values here.
# Each row is [steps=50, steps=100, steps=200] in milliseconds.
# -----------------------------------------------------------------------
timesteps = [50, 100, 200]

cuda_ms   = [0.208, 0.408, 0.790]   # your CUDA numbers (100 and 200 steps TBD)
openmp_ms = [11.629, 20.023, 36.848]  # group member's OpenMP numbers
hip_ms = [0.79, 1.26, 2.31]   # group member's HIP numbers

# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------
p = plot(
    timesteps, cuda_ms;
    label       = "CUDA",
    markershape = :circle,
    linewidth   = 2,
    xlabel      = "Timesteps",
    ylabel      = "Time (ms)",
    title       = "2D Heat Equation: Runtime vs Timesteps\n(kappa = 0.20, forward Euler)",
    legend      = :topleft,
    grid        = true,
)
plot!(timesteps, openmp_ms; label = "OpenMP", markershape = :square,  linewidth = 2)
plot!(timesteps, hip_ms;    label = "HIP",    markershape = :diamond, linewidth = 2)

savefig(p, "heat2d_timing.png")
println("Saved heat2d_timing.png")
display(p)
