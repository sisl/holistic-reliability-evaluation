using Plots; default(fontfamily="Computer Modern", framestyle=:box)
using CSV
using YAML
using DataFrames
using OrderedCollections
using GLM
using LaTeXStrings
using Statistics
using Random

function string_to_color(s::String)
    h = hash(s) # Hash the input string

    # Extract the RGB color components using the modulo operation
    r = mod1(h, 256)
    g = mod1(h >> 8, 256)
    b = mod1(h >> 16, 256)

    return RGB(r / 255, g / 255, b / 255)
end

function load_results(results_dir; self_trained=false, phase="train")
    evaluation_results = OrderedDict()
    datasets = readdir(results_dir)
    for d in datasets
        evaluation_results[d] = OrderedDict()
        algorithms = readdir(joinpath(results_dir, d))
        for a in algorithms
            evaluation_results[d][a] = OrderedDict()
            if self_trained
                seed_dir = joinpath(results_dir, d, a, phase, d)
            else
                seed_dir = joinpath(results_dir, d, a)
            end
            seeds = []
            try 
                seeds = readdir(seed_dir) 
            catch
                @warn "Failed to read dir: $(seed_dir)"
                continue
            end
            for s in seeds
                results = CSV.read(joinpath(seed_dir, s, "metrics.csv"), DataFrame)
                params = YAML.load_file(joinpath(seed_dir, s, "hparams.yaml"))
                if "test_performance" in names(results)
                    evaluation_results[d][a][s] = (results, params)
                end
            end
        end
    end
    return evaluation_results
end

function get_perf(dataset_results, sym, alg)
    vals = Float64[]
    for ver in keys(dataset_results[alg])
        v = dataset_results[alg][ver][1][!,sym]
        nonmissing = .! ismissing.(v)
        if sum(nonmissing) > 1
            @warn "Multiple values for $sym"
        end
        val = v[findlast(nonmissing)]

        # Push back the value
        push!(vals, val)
    end
    return vals
end

function get_all_by_alg(dataset_results, sym)
    # Convert to a symbol if needed
    if sym isa String
        sym = Symbol(sym)
    end

    algs = []
    vals = []
    for alg in keys(dataset_results)
        push!(algs, alg)
        push!(vals, get_perf(dataset_results, sym, alg))
    end
    vals, algs
end

function correlation(dataset_results, symx, symy)
    xvals, xalgs = get_all_by_alg(dataset_results, symx)
    yvals, yalgs = get_all_by_alg(dataset_results, symy)
    @assert xalgs == yalgs

    # Flatten for computing the linear fit
    allx = vcat(xvals...)
    ally = vcat(yvals...)

    # double check that the metrics have appropriate variation
    if std(allx) == 0 && std(ally) == 0
        return 1.0
    elseif std(allx) == 0 || std(ally) == 0
        return 0.0
    end

    return cor(allx, ally)
end


function plot_relationship(dataset_results, symx, symy; p=plot(), normx=nothing, normy=nothing, label_alg=false, show_yeqx=false, scale_yeqx=true)
    xvals, xalgs = get_all_by_alg(dataset_results, symx)
    yvals, yalgs = get_all_by_alg(dataset_results, symy)
    @assert xalgs == yalgs
    algs = xalgs

    if !isnothing(normx)
        xnorms, _ = get_all_by_alg(dataset_results, normx)
        xvals = [x ./ norm for (x, norm) in zip(xvals, xnorms)]
    end
    if !isnothing(normy)
        ynorms, _ = get_all_by_alg(dataset_results, normy)
        yvals = [y ./ norm for (y, norm) in zip(yvals, ynorms)]
    end
    
    # Flatten for computing the linear fit
    allx = vcat(xvals...)
    ally = vcat(yvals...)

    xmin = minimum(allx)
    ymin = minimum(ally)

    xmax = maximum(allx)
    ymax = maximum(ally)

    # Compute the r2 value using GLM.jl
    model = lm(@formula(y ~ x), DataFrame(x=allx, y=ally))
    R = r2(model)

    # Plot the linear fit
    for (x, y, alg) in zip(xvals, yvals, algs)
        scatter!(p, x, y, color=string_to_color(alg), label=label_alg ? alg : "", markerstrokecolor=:auto)
    end
    if show_yeqx
        if scale_yeqx
            minval = max(xmin, ymin)
            maxval = min(xmax, ymax)
        else
            minval, maxval = 0,1
        end
        plot!(p, minval:0.001:maxval, minval:0.001:maxval, color=:black, linestyle=:dash, alpha=0.5, label="\$y=x\$")
    end

    plot!(p, allx, allx .* coef(model)[2] .+ coef(model)[1], color=:black, label="\$R^2\$ = $(round(R, digits=2))")
end

function make_hribbon_shape(ys, values, ribbon)
    # make the ribbon as a shape to fill in
    rib_min = values .- ribbon[1] # the lower edge of the ribbon
    rib_max = values .+ ribbon[2] # the upper edge of the ribbon
    ys = [ys; [ys[end]; ys[end]]; reverse(ys); [ys[1]; ys[1]]]
    xs = [rib_max; [rib_max[end],rib_min[end]]; reverse(rib_min); [rib_min[1]; rib_max[1]]]
    return xs, ys
end

function single_metric_comp(results, metric, metric_name, baseline_algs, comparison_algs; ytick_label=true)
    # Compute the baseline params
    baseline = vcat([get_perf(results, metric, alg) for alg in baseline_algs]...)
    μbaseline = mean(baseline)
    σbaseline = std(baseline)

    N = length(comparison_algs)
    ys = [-0.1, 0.1*N]
    xs = μbaseline*ones(length(ys))
    ribbons = σbaseline*ones(length(ys))
    ribbon_xs, ribbon_ys = make_hribbon_shape(ys, xs, ribbons)
    p = plot(xs, ys, linestyle=:dash)
    plot!(ribbon_xs, ribbon_ys, fill=true, linewidth=0, fillalpha = 0.5,  fillcolor=:lightblue,label=nothing)
    plot!(size=(500,200), legend=false, xlabel=metric_name, framestyle=nothing, bottom_margin=5Plots.mm)
    if ytick_label
        yticks!(0:0.1:0.1*(N-1), collect(keys(comparison_algs)))
    else
        yticks!(0:0.1:0.1*(N-1), fill("", N))
    end
    for (i,(name, algs)) in enumerate(comparison_algs)
        vs = vcat([get_perf(results, metric, c) for c in algs]...)
        scatter!(vs, (i-1)*0.1*ones(length(vs)), alpha=0.7, markersize=7, markerstrokewidth=0)
    end
    p
end
