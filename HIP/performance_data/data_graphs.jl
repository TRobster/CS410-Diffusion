using CSV, DataFrames, Plots

data_dir = "HIP/performance_data/profiling_data"
out_dir  = "HIP/performance_data/profiling_graphs"
mkpath(out_dir)

config_files = ["config_1", "config_2", "config_3"]
labels       = ["Stride (s=4)", "Dimension (block=1024)", "Adaptive"]
shapes       = [:circle, :square, :diamond]

dfs = [CSV.read(joinpath(data_dir, c * ".csv"), DataFrame) for c in config_files]

# SIMD_UTILIZATION and MemUnitStalled are fractions — convert to percent
for df in dfs
    df.SIMD_UTILIZATION .*= 100
    df.MemUnitStalled   .*= 100
end

dims = dfs[1].Dim

metrics = [
    ("SIMD_UTILIZATION", "SIMD Utilization (%)",      "simd"),
    ("OccupancyPercent", "Occupancy (%)",              "occupancy"),
    ("MemUnitStalled",   "Memory Unit Stalled (%)",    "mem_stalled"),
]

for (col, ylabel_str, prefix) in metrics

    # --- Individual plot per config ---
    for (i, (df, label)) in enumerate(zip(dfs, labels))
        p = plot(
            df.Dim, df[!, col];
            label       = label,
            markershape = :circle,
            markersize  = 6,
            linewidth   = 2,
            color       = :steelblue,
            xlabel      = "Grid Dimension (N)",
            ylabel      = ylabel_str,
            title       = "$ylabel_str by Dimension — $label",
            legend      = :bottomright,
            grid        = true,
            xticks      = (dims, string.(dims)),
            xscale      = :log2,
        )
        out_path = joinpath(out_dir, "$(prefix)_config_$(i).png")
        savefig(p, out_path)
        println("Saved: $out_path")
    end

    # --- Comparison plot: all three configs ---
    p_cmp = plot(;
        xlabel  = "Grid Dimension (N)",
        ylabel  = ylabel_str,
        title   = "$ylabel_str — All Configs Comparison",
        legend  = :bottomright,
        grid    = true,
        xticks  = (dims, string.(dims)),
        xscale  = :log2,
    )
    for (df, label, shape) in zip(dfs, labels, shapes)
        plot!(p_cmp, df.Dim, df[!, col];
              label       = label,
              markershape = shape,
              markersize  = 6,
              linewidth   = 2)
    end
    out_path = joinpath(out_dir, "$(prefix)_comparison.png")
    savefig(p_cmp, out_path)
    println("Saved: $out_path")
end

println("\nDone — profiling graphs written to $out_dir")

# -----------------------------------------------------------------------
# Timing graphs
# -----------------------------------------------------------------------

timing_dir   = "HIP/performance_data/timing_data"
timing_files = ["config_1_times.txt", "config_2_times.txt", "config_3_times.txt"]

# Parse "Mean for NxN: VALUE ms [optional suffix]" from each file
function parse_timing(path)
    dims_out = Int[]
    times_out = Float64[]
    for line in eachline(path)
        m = match(r"Mean for (\d+) x \d+: ([\d.]+) ms", line)
        if m !== nothing
            push!(dims_out, parse(Int, m[1]))
            push!(times_out, parse(Float64, m[2]))
        end
    end
    return dims_out, times_out
end

timing_data = [parse_timing(joinpath(timing_dir, f)) for f in timing_files]
t_dims = timing_data[1][1]

# Individual plot per config
for (i, ((tdims, times), label)) in enumerate(zip(timing_data, labels))
    p = plot(
        tdims, times;
        label       = label,
        markershape = :circle,
        markersize  = 6,
        linewidth   = 2,
        color       = :steelblue,
        xlabel      = "Grid Dimension (N)",
        ylabel      = "Mean Execution Time (ms)",
        title       = "Execution Time by Dimension — $label",
        legend      = :topleft,
        grid        = true,
        xticks      = (t_dims, string.(t_dims)),
        xscale      = :log2,
    )
    out_path = joinpath(out_dir, "timing_config_$(i).png")
    savefig(p, out_path)
    println("Saved: $out_path")
end

# Comparison plot
p_t = plot(;
    xlabel  = "Grid Dimension (N)",
    ylabel  = "Mean Execution Time (ms)",
    title   = "Execution Time — All Configs Comparison",
    legend  = :topleft,
    grid    = true,
    xticks  = (t_dims, string.(t_dims)),
    xscale  = :log2,
)
for ((tdims, times), label, shape) in zip(timing_data, labels, shapes)
    plot!(p_t, tdims, times;
          label       = label,
          markershape = shape,
          markersize  = 6,
          linewidth   = 2)
end
out_path = joinpath(out_dir, "timing_comparison.png")
savefig(p_t, out_path)
println("Saved: $out_path")

println("\nDone — all graphs written to $out_dir")
