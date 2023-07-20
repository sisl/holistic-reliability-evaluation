using Plots, StatsPlots 

Plots.reset_defaults()
pgfplotsx()
default(
    thickness_scaling=1/2,
    palette=:seaborn_dark,
    titlefont=32,
    legendfontsize=24,
    guidefontsize=24,
    tickfontsize=24,
    colorbartickfontsizes=24,
    framestyle=:box,
    grid=true,
    linewidth=2,
    markersize=8,
    legend=:topleft,
    left_margin=5Plots.mm,
    size=(400,350),
    legend_font_halign=:left,
    fg_legend = RGBA(0,0,0,0.5),
)

using Plots.PlotMeasures
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
                try
                    results = CSV.read(joinpath(seed_dir, s, "metrics.csv"), DataFrame)
                    params = YAML.load_file(joinpath(seed_dir, s, "hparams.yaml"); dicttype=OrderedDict{Any, Any})
                    # if "test_performance" in names(results)
                    evaluation_results[d][a][s] = (results, params)
                    # end
                catch e
                    print("error: ", e)
                    @warn "Failed to read results for $(d) $(a) $(s)"
                end
            end
        end
    end
    return evaluation_results
end

function merge_results(r1, r2; augment_alg_name=nothing)
    r = deepcopy(r1)
    for d in keys(r2)
        if !(d in keys(r))
            r[d] = deepcopy(r2[d])
        else
            for a in keys(r2[d])
                if !(a in keys(r[d]))
                    r[d][a] = deepcopy(r2[d][a])
                else
                    if !isnothing(augment_alg_name)
                        r[d]["$a-$augment_alg_name"] = deepcopy(r2[d][a])
                    else
                        for s in keys(r2[d][a])
                            if !(s in keys(r[d][a]))
                                r[d][a][s] = deepcopy(r2[d][a][s])
                            else
                                @warn "Duplicate result for $(d) $(a) $(s)"
                            end
                        end
                    end
                end
            end
        end
    end
    return r
end

function get_perf(dataset_results, sym, alg; version=nothing, requires=nothing)
    versions = []
    vals = Float64[]
    for ver in keys(dataset_results[alg])
        if !(sym in names(dataset_results[alg][ver][1]))
            @warn "===============================  Symbol $sym not found for $alg"
            continue
        end
        v = dataset_results[alg][ver][1][!,sym]
        nonmissing = .! ismissing.(v)
        if sum(nonmissing) > 1
            @warn "Multiple values for $sym"
        end
        val = v[findlast(nonmissing)]

        # TODO: Find a better way to filter outliers
        # if sym=="val_performance" && val < 0.3
        #     println("found 1")
        #     continue
        # end

        # Push back the value
        push!(versions, ver)
        push!(vals, val)
    end
    if !isnothing(version)
        if !(version in versions)
            @warn "Version $version not found for $alg. available versions are: $versions"
            return nothing
        end
        return vals[findfirst(versions .== version)]
    end
    return vals
end

function get_all_by_alg(dataset_results, sym; requires=nothing)
    algs = []
    vals = []
    for alg in keys(dataset_results)
        if !isnothing(requires)
            if !(requires in keys(dataset_results[alg]))
                continue
            end
        end
        push!(algs, alg)
        push!(vals, get_perf(dataset_results, sym, alg))
    end
    vals, algs
end

function correlation(allx, ally)
    if std(allx) == 0 && std(ally) == 0
        return 1.0
    elseif std(allx) == 0 || std(ally) == 0
        return 0.0
    end

    return cor(allx, ally)
end

function adjusted_correlation(allx, ally, groups)
    unique_groups = unique(groups)
    zs = [Float64.(groups .== g) for g in unique_groups]

    fx = Term(:x) ~ sum(term.([Symbol("z$i") for i=1:length(zs)]))
    fy = Term(:y) ~ sum(term.([Symbol("z$i") for i=1:length(zs)]))
    pairs = [:x=>allx, :y=>ally, [Symbol("z$i") => zs[i] for i=1:length(zs)]...]
    data =  DataFrame(pairs...)
    
    modelx = lm(fx, data)
    modely = lm(fy, data)
    
    x_adjusted = allx .- (coef(modelx)[1] .+ sum([coef(modelx)[1+i] .* zs[i] for i=1:length(zs)]))
    y_adjusted = ally .- (coef(modely)[1] .+ sum([coef(modely)[1+i] .* zs[i] for i=1:length(zs)]))
    
    cor(x_adjusted, y_adjusted)
    # adjusted_model = lm(@formula(y ~ x), DataFrame(x=x_adjusted, y=y_adjusted))

end

function correlation(dataset_results, symx, symy)
    xvals, xalgs = get_all_by_alg(dataset_results, symx)
    yvals, yalgs = get_all_by_alg(dataset_results, symy)
    @assert xalgs == yalgs

    # Flatten for computing the linear fit
    allx = vcat(xvals...)
    ally = vcat(yvals...)

    # double check that the metrics have appropriate variation
    correlation(allx, ally)
end

function scatter_fit(allx, ally; p=plot(), show_yeqx=false, scale_yeqx=true)

    xmin = minimum(allx)
    ymin = minimum(ally)

    xmax = maximum(allx)
    ymax = maximum(ally)

    # Compute the r2 value using GLM.jl
    model = lm(@formula(y ~ x), DataFrame(x=allx, y=ally))
    R = r2(model)

    scatter!(p, allx, ally, color=:black, alpha=0.3, markerstrokealpha=0, label="")
    if show_yeqx
        if scale_yeqx
            minval = max(xmin, ymin)
            maxval = min(xmax, ymax)
        else
            minval, maxval = 0,1
        end
        plot!(p, minval:0.001:maxval, minval:0.001:maxval, color=:black, linestyle=:dash, alpha=0.5, label=L"$y=x$")
    end

    plot!(p, allx, allx .* coef(model)[2] .+ coef(model)[1], color=:black, label=L"$R^2$ = %$(round(R, digits=2))")
end

function make_hribbon_shape(ys, values, ribbon_bounds)
    # make the ribbon as a shape to fill in
    rib_min = ribbon_bounds[1] # the lower edge of the ribbon
    rib_max = ribbon_bounds[2] # the upper edge of the ribbon
    ys = [ys; [ys[end]; ys[end]]; reverse(ys); [ys[1]; ys[1]]]
    xs = [rib_max; [rib_max[end],rib_min[end]]; reverse(rib_min); [rib_min[1]; rib_max[1]]]
    return xs, ys
end

function single_metric_comp(results, metric, metric_name, baseline_algs, comparison_algs; ytick_label=true, kwargs...)
    p = plot(;kwargs...)
    N = length(comparison_algs)
    ys = [-0.1, 0.1*N]

    # If comparing to a baseline
    if !isnothing(baseline_algs)
        baseline = []
        for alg in baseline_algs
            try
                baseline = [baseline; get_perf(results, metric, alg)]
            catch
                @warn "Failed to get baseline for $alg"
            end
        end
        # baseline = vcat([get_perf(results, metric, alg) for alg in baseline_algs]...)
        μbaseline = mean(baseline)
        baselinemin, baselinemax = extrema(baseline)
        xs = μbaseline*ones(length(ys))
        ribbon_mins = baselinemin*ones(length(ys))
        ribbon_maxs = baselinemax*ones(length(ys))
        ribbon_xs, ribbon_ys = make_hribbon_shape(ys, xs, [ribbon_mins, ribbon_maxs])
        plot!(p, xs, ys, linestyle=:dash)
        plot!(ribbon_xs, ribbon_ys, fill=true, linewidth=0, fillalpha = 0.3, fillcolor=:lightblue, color=:lightblue, alpha=0, label=nothing)
    end
    
    plot!(size=(500,200), legend=false, xlabel=metric_name, bottom_margin=5Plots.mm)
    if ytick_label
        yticks!(0:0.1:0.1*(N-1), collect(keys(comparison_algs)))
    else
        yticks!(0:0.1:0.1*(N-1), fill("", N))
    end
    for (i,(name, algs)) in enumerate(comparison_algs)
        vs = []
        for alg in algs
            try
                vs = [vs; get_perf(results, metric, alg)]
            catch
                @warn "Failed to get results for $alg"
            end
        end
        # vs = vcat([get_perf(results, metric, c) for c in algs]...)
        scatter!(vs, (i-1)*0.1*ones(length(vs)), alpha=0.5, markersize=7, markerstrokealpha=0)
    end
    p
end


function plot_correlations(arrs, names; ylabels=true, annotate_rval=true, groups=nothing, kwargs...)
    N = length(names)
    fn = (x,y) -> y < x ? NaN : correlation(arrs[x], arrs[y])
    if !isnothing(groups)
        fn = (x,y) -> y < x ? NaN : adjusted_correlation(arrs[x], arrs[y], groups)
    end
    
    p = heatmap(1:N, 1:N, fn, cmap=:PiYG, clims=(-1,1), xrotation=45; colorbar=true, alpha=1, kwargs...)
    if annotate_rval
        for x in 1:N
            for y in x:N
                if x == y
                    continue
                end
                annotate!(x, y, (string(round(fn(x,y)^2, digits=2)), 28))
            end
        end
    end
    xticks!(0.5:N-0.5,  names)
    if ylabels
        yticks!(1:N, names)
    else
        yticks!(1:N, fill("", N))
    end
    p
end
