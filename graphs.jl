using Plots

# -----------------------------------------------------------------------
# Timing data: fill in your measured values here.
# Each row is [steps=50, steps=100, steps=200] in milliseconds.
# -----------------------------------------------------------------------
timesteps = [50, 100, 200]

cuda_ms   = [0.208, NaN, NaN]   # your CUDA numbers (100 and 200 steps TBD)
openmp_ms = [45.0, 90.5, 181.2]  # group member's OpenMP numbers
hip_ms    = [4.1,  8.3,  16.9]   # group member's HIP numbers

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
