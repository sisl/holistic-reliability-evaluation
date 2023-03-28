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

function load_results(results_dir)
    evaluation_results = OrderedDict()
    datasets = readdir(results_dir)
    for d in datasets
        evaluation_results[d] = OrderedDict()
        algorithms = readdir(joinpath(results_dir, d))
        for a in algorithms
            evaluation_results[d][a] = OrderedDict()
            seeds = readdir(joinpath(results_dir, d, a))
            for s in seeds
                results = CSV.read(joinpath(results_dir, d, a, s, "metrics.csv"), DataFrame)
                params = YAML.load_file(joinpath(results_dir, d, a, s, "hparams.yaml"))
                evaluation_results[d][a][s] = (results, params)
            end
        end
    end
    return evaluation_results
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
        push!(vals, Float64[])
        for ver in keys(dataset_results[alg])
            # Get the value by checking the first and second row and using the one that is not missing
            val = val = dataset_results[alg][ver][1][1, sym]
            ismissing(val) && (val = dataset_results[alg][ver][1][2, sym])

            # Push back the value
            push!(vals[end], val)
        end
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