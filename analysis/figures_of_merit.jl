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


function plot_relationship(dataset_results, symx, symy; p=plot(), normx=nothing, normy=nothing)
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

    # Compute the r2 value using GLM.jl
    model = lm(@formula(y ~ x), DataFrame(x=allx, y=ally))
    R = r2(model)

    # Plot the linear fit
    for (x, y, alg) in zip(xvals, yvals, algs)
        scatter!(p, x, y, color=string_to_color(alg), label=alg)
    end

    plot!(p, allx, allx .* coef(model)[2] .+ coef(model)[1], color=:black, label="\$R^2\$ = $(round(R, digits=2))")

end

function domain_shift_performance_relationships(dataset)
    # Plot Domain shifted performance vs in-distribution performance 
    p1 = plot_relationship(results[dataset], "val_performance", "$dataset-val_performance", p=plot(title="DS-Val vs ID Performance", xlabel="ID-Val Performance", ylabel="DS-Val Performance"))
    p2 = plot_relationship(results[dataset], "val_performance", "$dataset-test_performance", p=plot(title="DS-Test vs ID Performance", xlabel="ID-Val Performance", ylabel="DS-Test Performance"))

    # Plot Domain shifted performance vs other domain shifted performance
    p3 = plot_relationship(results[dataset], "$dataset-val_performance", "$dataset-test_performance", p=plot(title="DS-Test vs DS-Val Performance", xlabel="DS-Val Performance", ylabel="DS-Test Performance"))

    # Plot Domain shifted performance vs corruption performance
    p4 = plot_relationship(results[dataset], "$dataset-id_val-corruption1_val_performance", "$dataset-val_performance", p=plot(title="DS-Val vs C1-Val Performance", xlabel="C1-Val Performance", ylabel="DS-Val Performance"))
    p5 = plot_relationship(results[dataset], "$dataset-id_val-corruption1_val_performance", "$dataset-test_performance", p=plot(title="DS-Test vs C1-Val Performance", xlabel="C1-Val Performance", ylabel="DS-Test Performance"))
    p6 = plot_relationship(results[dataset], "$dataset-id_test-corruption1_test_performance", "$dataset-val_performance", p=plot(title="DS-Val vs C1-Test Performance", xlabel="C1-Test Performance", ylabel="DS-Val Performance"))
    p7 = plot_relationship(results[dataset], "$dataset-id_test-corruption1_test_performance", "$dataset-test_performance", p=plot(title="DS-Test vs C1-Test Performance", xlabel="C1-Test Performance", ylabel="DS-Test Performance"))

    # Plot the corruption performances against each other
    p8 = plot_relationship(results[dataset], "$dataset-id_val-corruption1_val_performance", "$dataset-id_test-corruption1_test_performance", p=plot(title="C1-Test vs C1-Val Performance", xlabel="C1-Val Performance", ylabel="C1-Test Performance"))

    plot(p1, p2, p3, p4, p5, p6, p7, p8, layout=(8, 1), size=(600, 400*8), left_margin=25Plots.mm)
end

function domain_shift_robustness_relationships(dataset)
    # Plot Domain shifted robustness vs in-distribution performance 
    p1 = plot_relationship(results[dataset], "val_performance", "$dataset-val_performance", p=plot(title="DS-Val vs ID Performance", xlabel="ID-Val Performance", ylabel="DS-Val Robustness"), normy="val_performance")
    p2 = plot_relationship(results[dataset], "val_performance", "$dataset-test_performance", p=plot(title="DS-Test vs ID Performance", xlabel="ID-Val Performance", ylabel="DS-Test Robustness"), normy="val_performance")

    # Plot Domain shifted robustness vs other domain shifted performance
    p3 = plot_relationship(results[dataset], "$dataset-val_performance", "$dataset-test_performance", p=plot(title="DS-Test vs DS-Val Performance", xlabel="DS-Val Robustness", ylabel="DS-Test Robustness"), normx="val_performance", normy="val_performance")

    # Plot Domain shifted robustness vs corruption performance
    p4 = plot_relationship(results[dataset], "$dataset-id_val-corruption1_val_performance", "$dataset-val_performance", p=plot(title="DS-Val vs C1-Val Robustness", xlabel="C1-Val Robustness", ylabel="DS-Val Robustness"), normx="val_performance", normy="val_performance")
    p5 = plot_relationship(results[dataset], "$dataset-id_val-corruption1_val_performance", "$dataset-test_performance", p=plot(title="DS-Test vs C1-Val Robustness", xlabel="C1-Val Robustness", ylabel="DS-Test Robustness"), normx="val_performance", normy="val_performance")
    p6 = plot_relationship(results[dataset], "$dataset-id_test-corruption1_test_performance", "$dataset-val_performance", p=plot(title="DS-Val vs C1-Test Robustness", xlabel="C1-Test Robustness", ylabel="DS-Val Robustness"), normx="val_performance", normy="val_performance")
    p7 = plot_relationship(results[dataset], "$dataset-id_test-corruption1_test_performance", "$dataset-test_performance", p=plot(title="DS-Test vs C1-Test Robustness", xlabel="C1-Test Robustness", ylabel="DS-Test Robustness"), normx="val_performance", normy="val_performance")

    # Plot the corruption robustness against each other
    p8 = plot_relationship(results[dataset], "$dataset-id_val-corruption1_val_performance", "$dataset-id_test-corruption1_test_performance", p=plot(title="C1-Test vs C1-Val Robustness", xlabel="C1-Val Robustness", ylabel="C1-Test Robustness"), normx="val_performance", normy="val_performance")

    plot(p1, p2, p3, p4, p5, p6, p7, p8, layout=(8, 1), size=(600, 400*8), left_margin=25Plots.mm)
end

function domain_shift_calibration_relationships(dataset)
    # Plot Domain shifted calibration vs in-distribution performance 
    p1 = plot_relationship(results[dataset], "$dataset-id_val_calibration", "$dataset-val_calibration", p=plot(title="DS-Val vs ID Calibration", xlabel="ID-Val Calibration", ylabel="DS-Val Calibration"))
    p2 = plot_relationship(results[dataset], "$dataset-id_val_calibration", "$dataset-test_calibration", p=plot(title="DS-Test vs ID Calibration", xlabel="ID-Val Calibration", ylabel="DS-Test Calibration"))

    # Plot Domain shifted calibration vs other domain shifted performance
    p3 = plot_relationship(results[dataset], "$dataset-val_calibration", "$dataset-test_calibration", p=plot(title="DS-Test vs DS-Val Calibration", xlabel="DS-Val Calibration", ylabel="DS-Test Calibration"))

    # Plot Domain shifted calibration vs corruption performance
    p4 = plot_relationship(results[dataset], "$dataset-id_val-corruption1_val_calibration", "$dataset-val_calibration", p=plot(title="DS-Val vs C1-Val Calibration", xlabel="C1-Val Calibration", ylabel="DS-Val Calibration"))
    p5 = plot_relationship(results[dataset], "$dataset-id_val-corruption1_val_calibration", "$dataset-test_calibration", p=plot(title="DS-Test vs C1-Val Calibration", xlabel="C1-Val Calibration", ylabel="DS-Test Calibration"))
    p6 = plot_relationship(results[dataset], "$dataset-id_test-corruption1_test_calibration", "$dataset-val_calibration", p=plot(title="DS-Val vs C1-Test Calibration", xlabel="C1-Test Calibration", ylabel="DS-Val Calibration"))
    p7 = plot_relationship(results[dataset], "$dataset-id_test-corruption1_test_calibration", "$dataset-test_calibration", p=plot(title="DS-Test vs C1-Test Calibration", xlabel="C1-Test Calibration", ylabel="DS-Test Calibration"))

    # Plot the corruption calibration against each other
    p8 = plot_relationship(results[dataset], "$dataset-id_val-corruption1_val_calibration", "$dataset-id_test-corruption1_test_calibration", p=plot(title="C1-Test vs C1-Val Calibration", xlabel="C1-Val Calibration", ylabel="C1-Test Robustness"))

    plot(p1, p2, p3, p4, p5, p6, p7, p8, layout=(8, 1), size=(600, 400*8), left_margin=25Plots.mm)
end

function plot_metric_correlations(dataset_results; prefix="val", title="")
    syms = ["$(prefix)_performance","$(prefix)_robustness","$(prefix)_security","$(prefix)_calibration","$(prefix)_ood_detection"]
    names = ["Performance", "Robustness", "Security", "Calibration", "Fault Detection"]

    N = length(syms)
    fn(x,y) = y < x ? NaN : correlation(dataset_results, syms[x], syms[y])
    heatmap(1:N, 1:N, fn, cmap=:PiYG, clims=(-1,1), title=title)
    xticks!(1:N, names)
    yticks!(1:N, names)
end

function parallel_coord_mean_std(results, datasets, dataset_names, syms, names)
    p = plot()
    for (d, name) in zip(datasets, dataset_names)
        all_metrics = [vcat(get_all_by_alg(results[d], s)[1]...) for s in syms(d)]

        series = [mean(m) for m in all_metrics]
        errors = [std(m) for m in all_metrics]
        plot!(p,series, yerror=errors, label=name, markerstrokecolor=:auto )
        xticks!(1:length(names), names)
    end
    p
end

function unit_range(vec; dims)
    minvals = minimum(vec, dims=dims)
    maxvals = maximum(vec, dims=dims)
    v = (vec .- minvals) ./ (maxvals .- minvals)
    v[.! isfinite.(v)] .= 0.
    v
end

function parallel_coord_by_alg(dataset_results, syms_fn, names; prefix="val", normalize=false)
    metrics = [get_all_by_alg(dataset_results, sym)[1] for sym in syms_fn(prefix)]
    algs = get_all_by_alg(dataset_results, syms_fn(prefix)[1])[2]
    p = plot()

    # Compute the mean metric across the seeds
    series = zeros(length(names), length(algs))
    for i in 1:length(algs)
        seeds = hcat([[m[i][j] for m in metrics] for j in 1:length(metrics[1][i])]...)
        series[:,i] = mean(seeds, dims=2)
    end

    # Normalize the series
    if normalize
        series = unit_range(series, dims=2)
    end

    # Plot the (possibly normalized) results
    for (i,a) in enumerate(algs)
        plot!(p, mean(series[:,i], dims=2), label=a, linecolor=string_to_color(a), legend = :outertopleft)
    end
    xticks!(1:length(names), names)
    p
end

function scatter_hre_scores(dataset_results; prefix="val")
    hres, algs = get_all_by_alg(dataset_results, "$(prefix)_hre_score")
    p = plot()
    for (i,a) in enumerate(algs)
        c = string_to_color(a)
        scatter!(p, fill(i, length(hres[i])), hres[i], color=c, markerstrokecolor=c, label="")
    end
    xticks!(1:length(algs), String.(algs), xrotation = 90 )
    p
end

results_dir = "evaluation_results"
results = load_results(results_dir)
datasets = ["camelyon17", "iwildcam", "fmow", "rxrx1"]
dataset_names = ["Camelyon17", "iWildCam", "FMoW", "RxRx1"]

## Demonstrate the drop in performance and calibration due to domain shifts
# Performance
syms_fn(d) = ["val_performance", "$d-val_performance", "$d-test_performance", "$d-id_val-corruption1_val_performance", "$d-id_test-corruption1_test_performance"]
axis_labels = ["ID Performance", "DS-Val Performance", "DS-Test Performance", "DS-C1-Val Performance", "DS-C1-Test Performance"]
parallel_coord_mean_std(results, datasets, dataset_names, syms_fn, axis_labels)

# Calibration
syms_fn(d) = ["$d-id_val_calibration", "$d-val_calibration", "$d-test_calibration", "$d-id_val-corruption1_val_calibration", "$d-id_test-corruption1_test_calibration"]
axis_labels = ["ID Calibration", "DS-Val Calibration", "DS-Test Calibration", "DS-C1-Val Calibration", "DS-C1-Test Calibration"]
parallel_coord_mean_std(results, datasets, dataset_names, syms_fn, axis_labels)

## Compare the correlation between metrics
p1 = plot_metric_correlations(results["camelyon17"], title="Metric Correlations on Camelyon17 (Val)")
p2 = plot_metric_correlations(results["camelyon17"], prefix="test", title="Metric Correlations on Camelyon17 (Test)")

p3 = plot_metric_correlations(results["iwildcam"], title="Metric Correlations on iWildCam (Val)")
p4 = plot_metric_correlations(results["iwildcam"], prefix="test", title="Metric Correlations on iWildCam (Test)")

p5 = plot_metric_correlations(results["fmow"], title="Metric Correlations on FMOW (Val)")
p6 = plot_metric_correlations(results["fmow"], prefix="test", title="Metric Correlations on FMOW (Test)")

p7 = plot_metric_correlations(results["rxrx1"], title="Metric Correlations on RxRx1 (Val)")
p8 = plot_metric_correlations(results["rxrx1"], prefix="test", title="Metric Correlations on RxRx1 (Test)")

plot(p1, p2, p3, p4, p5, p6, p7, p8, layout=(4, 2), size=(1200, 400*4))


## Compare individual algorithms across metrics (one plot per dataset)
syms_fn(prefix) = ["$(prefix)_performance","$(prefix)_robustness","$(prefix)_security","$(prefix)_calibration","$(prefix)_ood_detection"]
names = ["Performance", "Robustness", "Security", "Calibration", "Fault Detection"]
p1 = parallel_coord_by_alg(results["camelyon17"], syms_fn, names, prefix="val")
p2 = parallel_coord_by_alg(results["camelyon17"], syms_fn, names, prefix="test")

p3 = parallel_coord_by_alg(results["iwildcam"], syms_fn, names, prefix="val")
p4 = parallel_coord_by_alg(results["iwildcam"], syms_fn, names, prefix="test")

p5 = parallel_coord_by_alg(results["fmow"], syms_fn, names, prefix="val")
p6 = parallel_coord_by_alg(results["fmow"], syms_fn, names, prefix="test")

p7 = parallel_coord_by_alg(results["rxrx1"], syms_fn, names, prefix="val")
p8 = parallel_coord_by_alg(results["rxrx1"], syms_fn, names, prefix="test")
plot(p1, p2, p3, p4, p5, p6, p7, p8, layout=(4, 2), size=(2000, 400*4))

## Compare HRE scores for each
p1 = scatter_hre_scores(results["camelyon17"]; prefix="val")
p2 = scatter_hre_scores(results["camelyon17"]; prefix="test")

p3 = scatter_hre_scores(results["iwildcam"]; prefix="val")
p4 = scatter_hre_scores(results["iwildcam"]; prefix="test")

p5 = scatter_hre_scores(results["fmow"]; prefix="val")
p6 = scatter_hre_scores(results["fmow"]; prefix="test")

p7 = scatter_hre_scores(results["rxrx1"]; prefix="val")
p8 = scatter_hre_scores(results["rxrx1"]; prefix="test")
plot(p1, p2, p3, p4, p5, p6, p7, p8, layout=(4, 2), size=(600*2, 400*4))
savefig("hre_plots.pdf")

## Plot some individual relationships of interest
# Sanity check of val vs test performance
p1 = plot_relationship(results["camelyon17"], "val_performance", "test_performance", p=plot(title="ID-Test vs ID-Val Performance", xlabel="ID-Val Performance", ylabel="ID-Test Performance"))
p2 = plot_relationship(results["iwildcam"], "val_performance", "test_performance", p=plot(title="ID-Test vs ID-Val Performance", xlabel="ID-Val Performance", ylabel="ID-Test Performance"))
p3 = plot_relationship(results["fmow"], "val_performance", "test_performance", p=plot(title="ID-Test vs ID-Val Performance", xlabel="ID-Val Performance", ylabel="ID-Test Performance"))
p4 = plot_relationship(results["rxrx1"], "val_performance", "test_performance", p=plot(title="ID-Test vs ID-Val Performance", xlabel="ID-Val Performance", ylabel="ID-Test Performance"))
plot(p1, p2, p3, p4, layout=(1, 4), size=(4*600, 400))

# Performance and robustness
p1 = plot_relationship(results["camelyon17"], "val_performance", "test_robustness", p=plot(title="Test Robustness vs ID-Val Performance", xlabel="ID-Val Performance", ylabel="Test Robustness"))
p2 = plot_relationship(results["iwildcam"], "val_performance", "test_robustness", p=plot(title="Test Robustness vs ID-Val Performance", xlabel="ID-Val Performance", ylabel="Test Robustness"))
p3 = plot_relationship(results["fmow"], "val_performance", "test_robustness", p=plot(title="Test Robustness vs ID-Val Performance", xlabel="ID-Val Performance", ylabel="Test Robustness"))
p4 = plot_relationship(results["rxrx1"], "val_performance", "test_robustness", p=plot(title="Test Robustness vs ID-Val Performance", xlabel="ID-Val Performance", ylabel="Test Robustness"))
plot(p1, p2, p3, p4, layout=(1, 4), size=(4*600, 400))

# Performance and calibration
p1 = plot_relationship(results["camelyon17"], "val_performance", "test_calibration", p=plot(title="Test Calibration vs ID-Val Performance", xlabel="ID-Val Performance", ylabel="Test Calibration"))
p2 = plot_relationship(results["iwildcam"], "val_performance", "test_calibration", p=plot(title="Test Calibration vs ID-Val Performance", xlabel="ID-Val Performance", ylabel="Test Calibration"))
p3 = plot_relationship(results["fmow"], "val_performance", "test_calibration", p=plot(title="Test Calibration vs ID-Val Performance", xlabel="ID-Val Performance", ylabel="Test Calibration"))
p4 = plot_relationship(results["rxrx1"], "val_performance", "test_calibration", p=plot(title="Test Calibration vs ID-Val Performance", xlabel="ID-Val Performance", ylabel="Test Calibration"))
plot(p1, p2, p3, p4, layout=(1, 4), size=(4*600, 400))

# Performance and security
p1 = plot_relationship(results["camelyon17"], "val_performance", "test_security", p=plot(title="Test Security vs ID-Val Performance", xlabel="ID-Val Performance", ylabel="Test Security"))
p2 = plot_relationship(results["iwildcam"], "val_performance", "test_security", p=plot(title="Test Security vs ID-Val Performance", xlabel="ID-Val Performance", ylabel="Test Security"))
p3 = plot_relationship(results["fmow"], "val_performance", "test_security", p=plot(title="Test Security vs ID-Val Performance", xlabel="ID-Val Performance", ylabel="Test Security"))
p4 = plot_relationship(results["rxrx1"], "val_performance", "test_security", p=plot(title="Test Security vs ID-Val Performance", xlabel="ID-Val Performance", ylabel="Test Security"))
plot(p1, p2, p3, p4, layout=(1, 4), size=(4*600, 400))

# Performance and fault detection
p1 = plot_relationship(results["camelyon17"], "val_performance", "test_ood_detection", p=plot(title="Test Fault Detection vs ID-Val Performance", xlabel="ID-Val Performance", ylabel="Test Fault Detection"))
p2 = plot_relationship(results["iwildcam"], "val_performance", "test_ood_detection", p=plot(title="Test Fault Detection vs ID-Val Performance", xlabel="ID-Val Performance", ylabel="Test Fault Detection"))
p3 = plot_relationship(results["fmow"], "val_performance", "test_ood_detection", p=plot(title="Test Fault Detection vs ID-Val Performance", xlabel="ID-Val Performance", ylabel="Test Fault Detection"))
p4 = plot_relationship(results["rxrx1"], "val_performance", "test_ood_detection", p=plot(title="Test Fault Detection vs ID-Val Performance", xlabel="ID-Val Performance", ylabel="Test Fault Detection"))
plot(p1, p2, p3, p4, layout=(1, 4), size=(4*600, 400))



## Relationship of performance and calibration metrics to different domain shifts
# Relationship of domain shift performance to in-distribution performance for a variety of datasets
p1 = domain_shift_performance_relationships("camelyon17")
p2 = domain_shift_performance_relationships("iwildcam")
p3 = domain_shift_performance_relationships("fmow")
p4 = domain_shift_performance_relationships("rxrx1")
plot(p1,p2,p3,p4, size=(600*4, 400*8), layout=(1, 4))

# Relationship of domain shift robustness to in-distribution performance for a variety of datasets
p1 = domain_shift_robustness_relationships("camelyon17")
p2 = domain_shift_robustness_relationships("iwildcam")
p3 = domain_shift_robustness_relationships("fmow")
p4 = domain_shift_robustness_relationships("rxrx1")
plot(p1,p2,p3,p4, size=(600*4, 400*8), layout=(1, 4))

# Compare the relationship between calibrations across datasets
p1 = domain_shift_calibration_relationships("camelyon17")
p2 = domain_shift_calibration_relationships("iwildcam")
p3 = domain_shift_calibration_relationships("fmow")
p4 = domain_shift_calibration_relationships("rxrx1")
plot(p1,p2,p3,p4, size=(600*4, 400*8), layout=(1, 4))