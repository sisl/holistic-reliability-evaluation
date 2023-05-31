using Plots; default(fontfamily="Computer Modern", framestyle=:box)
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

    scatter!(p, allx, ally, color=:black, alpha=0.3, markerstroke=:auto, label="")
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


function plot_relationship(dataset_results, symx, symy; p=plot(), normx=nothing, normy=nothing, label_alg=false, show_yeqx=false, scale_yeqx=true)
    xvals, xalgs = get_all_by_alg(dataset_results, symx, requires=symy)
    yvals, yalgs = get_all_by_alg(dataset_results, symy, requires=symx)
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

function make_hribbon_shape(ys, values, ribbon_bounds)
    # make the ribbon as a shape to fill in
    rib_min = ribbon_bounds[1] # the lower edge of the ribbon
    rib_max = ribbon_bounds[2] # the upper edge of the ribbon
    ys = [ys; [ys[end]; ys[end]]; reverse(ys); [ys[1]; ys[1]]]
    xs = [rib_max; [rib_max[end],rib_min[end]]; reverse(rib_min); [rib_min[1]; rib_max[1]]]
    return xs, ys
end

function single_metric_comp(results, metric, metric_name, baseline_algs, comparison_algs; ytick_label=true)
    p = plot()
    N = length(comparison_algs)
    ys = [-0.1, 0.1*N]

    # If comparing to a baseline
    if !isnothing(baseline_algs)
        baseline = vcat([get_perf(results, metric, alg) for alg in baseline_algs]...)
        μbaseline = mean(baseline)
        baselinemin, baselinemax = extrema(baseline)
        xs = μbaseline*ones(length(ys))
        ribbon_mins = baselinemin*ones(length(ys))
        ribbon_maxs = baselinemax*ones(length(ys))
        ribbon_xs, ribbon_ys = make_hribbon_shape(ys, xs, [ribbon_mins, ribbon_maxs])
        plot!(p, xs, ys, linestyle=:dash)
        plot!(ribbon_xs, ribbon_ys, fill=true, linewidth=0, fillalpha = 0.5, fillcolor=:lightblue, color=:lightblue, alpha=0.5, label=nothing)
    end
    
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


function plot_correlations(arrs, names; annotate_rval =true, kwargs...)
    N = length(names)
    fn(x,y) = y < x ? NaN : correlation(arrs[x], arrs[y])
    heatmap(1:N, 1:N, fn, cmap=:PiYG, clims=(-1,1), xrotation=45; colorbar=true, kwargs...)
    if annotate_rval
        for x in 1:N
            for y in x:N
                if x == y
                    continue
                end
                annotate!(x, y, (string(round(fn(x,y)^2, digits=2)), 10))
            end
        end
    end
    xticks!(1:N, names)
    yticks!(1:N, names)
end
