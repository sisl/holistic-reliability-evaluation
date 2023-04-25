include("plotting_utils.jl")

function domain_shift_performance_relationships(dataset, datasetname)
    # Plot Domain shifted performance vs in-distribution performance 
    # p1 = plot_relationship(results[dataset], "val_performance", "$dataset-val_performance", p=plot(title="DS-Val vs ID Performance", xlabel="ID-Val Performance", ylabel="DS-Val Performance"))
    p2 = plot_relationship(results[dataset], "val_performance", "$dataset-test_performance", p=plot(title=datasetname, xlabel="ID-Val Performance", ylabel="DS-Test Performance"), show_yeqx=true, scale_yeqx=false)

    # Plot Domain shifted performance vs other domain shifted performance
    p3 = plot_relationship(results[dataset], "$dataset-val_performance", "$dataset-test_performance", p=plot(title="", xlabel="DS-Val Performance", ylabel="DS-Test Performance"), show_yeqx=true, scale_yeqx=false)

    # Plot Domain shifted performance vs corruption performance
    # p4 = plot_relationship(results[dataset], "$dataset-id_val-corruption1_val_performance", "$dataset-val_performance", p=plot(title="DS-Val vs C1-Val Performance", xlabel="C1-Val Performance", ylabel="DS-Val Performance"))
    p5 = plot_relationship(results[dataset], "$dataset-id_val-corruption1_val_performance", "$dataset-test_performance", p=plot(title="", xlabel="C1-Val Performance", ylabel="DS-Test Performance"), show_yeqx=true, scale_yeqx=false)
    # p6 = plot_relationship(results[dataset], "$dataset-id_test-corruption1_test_performance", "$dataset-val_performance", p=plot(title="DS-Val vs C1-Test Performance", xlabel="C1-Test Performance", ylabel="DS-Val Performance"))
    # p7 = plot_relationship(results[dataset], "$dataset-id_test-corruption1_test_performance", "$dataset-test_performance", p=plot(title="DS-Test vs C1-Test Performance", xlabel="C1-Test Performance", ylabel="DS-Test Performance"))

    # Plot the corruption performances against each other
    p8 = plot_relationship(results[dataset], "$dataset-id_val-corruption1_val_performance", "$dataset-id_test-corruption1_test_performance", p=plot(title="", xlabel="C1-Val Performance", ylabel="C1-Test Performance"), show_yeqx=true, scale_yeqx=false)

    plot( p2, p3, p5, p8, layout=(4, 1), size=(600, 400*4), left_margin=25Plots.mm)
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

function domain_shift_calibration_relationships(dataset, dataset_name)
    # Plot Domain shifted calibration vs in-distribution performance 
    # p1 = plot_relationship(results[dataset], "$dataset-id_val_calibration", "$dataset-val_calibration", p=plot(title="DS-Val vs ID Calibration", xlabel="ID-Val Calibration", ylabel="DS-Val Calibration"))
    p2 = plot_relationship(results[dataset], "$dataset-id_val_calibration", "$dataset-test_calibration", p=plot(title="dataset_name", xlabel="ID-Val Calibration", ylabel="DS-Test Calibration"))

    # Plot Domain shifted calibration vs other domain shifted calibration
    p3 = plot_relationship(results[dataset], "$dataset-val_calibration", "$dataset-test_calibration", p=plot(title="", xlabel="DS-Val Calibration", ylabel="DS-Test Calibration"))

    # Plot Domain shifted calibration vs corruption performance
    # p4 = plot_relationship(results[dataset], "$dataset-id_val-corruption1_val_calibration", "$dataset-val_calibration", p=plot(title="DS-Val vs C1-Val Calibration", xlabel="C1-Val Calibration", ylabel="DS-Val Calibration"))
    p5 = plot_relationship(results[dataset], "$dataset-id_val-corruption1_val_calibration", "$dataset-test_calibration", p=plot(title="", xlabel="C1-Val Calibration", ylabel="DS-Test Calibration"))
    # p6 = plot_relationship(results[dataset], "$dataset-id_test-corruption1_test_calibration", "$dataset-val_calibration", p=plot(title="DS-Val vs C1-Test Calibration", xlabel="C1-Test Calibration", ylabel="DS-Val Calibration"))
    # p7 = plot_relationship(results[dataset], "$dataset-id_test-corruption1_test_calibration", "$dataset-test_calibration", p=plot(title="DS-Test vs C1-Test Calibration", xlabel="C1-Test Calibration", ylabel="DS-Test Calibration"))

    # Plot the corruption calibration against each other
    p8 = plot_relationship(results[dataset], "$dataset-id_val-corruption1_val_calibration", "$dataset-id_test-corruption1_test_calibration", p=plot(title="", xlabel="C1-Val Calibration", ylabel="C1-Test Robustness"))

    plot(p2,p3,p5,p8, layout=(4, 1), size=(600, 400*4), left_margin=25Plots.mm)
end

function plot_metric_correlations(dataset_results; prefix="val", title="", colorbar=false)
    syms = ["$(prefix)_performance","$(prefix)_robustness","$(prefix)_security","$(prefix)_calibration","$(prefix)_ood_detection"]
    names = ["Performance", "Robustness", "Security", "Calibration", "Fault Detection"]

    N = length(syms)
    fn(x,y) = y < x ? NaN : correlation(dataset_results, syms[x], syms[y])
    heatmap(1:N, 1:N, fn, cmap=:PiYG, clims=(-1,1), xrotation=45; title, colorbar)
    xticks!(1:N, names)
    yticks!(1:N, names)
end

function parallel_coord_mean_std(results, datasets, dataset_names, syms, names, p=plot())
    for (d, name) in zip(datasets, dataset_names)
        all_metrics = [vcat(get_all_by_alg(results[d], s)[1]...) for s in syms(d)]

        series = [mean(m) for m in all_metrics]
        errors = [std(m) for m in all_metrics]
        scatter!(p,series, yerror=errors, linestyle=:solid, label=name, markerstrokecolor=:auto, alpha=0.8)
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

function scatter_scores(dataset_results; metric="val_hre_score", p=plot())
    hres, algs = get_all_by_alg(dataset_results, metric)
    for (i,a) in enumerate(algs)
        c = string_to_color(a)
        scatter!(p, fill(i, length(hres[i])), hres[i], color=c, markerstrokecolor=c, label="")
    end
    xticks!(1:length(algs), String.(algs), xrotation = 90 )
    p
end



# results_dir = "results/evaluation_results"
results_dir = "results/evaluation_results"
results = load_results(results_dir)
datasets = ["camelyon17", "iwildcam", "fmow", "rxrx1"]
dataset_names = ["Camelyon17", "iWildCam", "fMoW", "RxRx1"]

finetune_results_dir = "/mnt/data/acorso/results/"
fine_tune_results = load_results(finetune_results_dir, self_trained=true)

# Put all the results together
all_results = deepcopy(results)
ft_results = Dict("iwildcam" => Dict())
# Combine dictionaries
for (k, v) in fine_tune_results["iwildcam-train"]
    all_results["iwildcam"][k] = v
    ft_results["iwildcam"][k] = v
end


## Demonstrate the drop in performance and calibration due to domain shifts
# Performance
syms_fn(d) = ["val_performance", "$d-val_performance", "$d-test_performance", "$d-id_val-corruption1_val_performance", "$d-id_test-corruption1_test_performance"]
axis_labels = ["ID Performance", "DS-Val Performance", "DS-Test Performance", "DS-C1-Val Performance", "DS-C1-Test Performance"]
parallel_coord_mean_std(results, datasets, dataset_names, syms_fn, axis_labels, plot(title="Performance Drop Under Domain Shift", ylabel="Performance", legend=:outertopright, xrotation=45, bottommargin=10Plots.mm, dpi=300))
savefig("analysis/figures/performance_drop.png")

# fine tune
p = parallel_coord_mean_std(results,  ["iwildcam"], ["iWildCam"], syms_fn, axis_labels, plot(title="Performance Drop Under Domain Shift", ylabel="Performance", legend=:outertopright, xrotation=45, bottommargin=10Plots.mm, dpi=300))
parallel_coord_mean_std(ft_results, ["iwildcam"], ["iWildCam-FT"], syms_fn, axis_labels, p)
savefig("analysis/figures/ft-performance_drop.png")

# Calibration
syms_fn(d) = ["$d-id_val_calibration", "$d-val_calibration", "$d-test_calibration", "$d-id_val-corruption1_val_calibration", "$d-id_test-corruption1_test_calibration"]
axis_labels = ["ID Calibration", "DS-Val Calibration", "DS-Test Calibration", "DS-C1-Val Calibration", "DS-C1-Test Calibration"]
parallel_coord_mean_std(results, datasets, dataset_names, syms_fn, axis_labels, plot(title="Calibration Drop Under Domain Shift", ylabel="Calibration", legend=:outertopright, xrotation=45, bottommargin=10Plots.mm, dpi=300, markershape=:circle))
savefig("analysis/figures/calibration_drop.png")

#fine tune
p = parallel_coord_mean_std(results,  ["iwildcam"], ["iWildCam"], syms_fn, axis_labels, plot(title="Calibration Drop Under Domain Shift", ylabel="Calibration", legend=:outertopright, xrotation=45, bottommargin=10Plots.mm, dpi=300, markershape=:circle))
parallel_coord_mean_std(ft_results, ["iwildcam"], ["iWildCam-FT"], syms_fn, axis_labels, p)
savefig("analysis/figures/ft-calibration_drop.png")


## Compare the correlation between metrics
p1 = plot_metric_correlations(results["camelyon17"], title="Metric Correlations on Camelyon17 (Val)")
p2 = plot_metric_correlations(results["camelyon17"], prefix="test", title="Metric Correlations on Camelyon17 (Test)")

p3 = plot_metric_correlations(results["iwildcam"], title="Metric Correlations on iWildCam (Val)")
p4 = plot_metric_correlations(results["iwildcam"], prefix="test", title="Metric Correlations on iWildCam (Test)")
plot_metric_correlations(fine_tune_results["iwildcam-train"], prefix="test")

p5 = plot_metric_correlations(results["fmow"], title="Metric Correlations on FMOW (Val)")
p6 = plot_metric_correlations(results["fmow"], prefix="test", title="Metric Correlations on FMOW (Test)")

p7 = plot_metric_correlations(results["rxrx1"], title="Metric Correlations on RxRx1 (Val)", colorbar=true)
p8 = plot_metric_correlations(results["rxrx1"], prefix="test", title="Metric Correlations on RxRx1 (Test)", colorbar=true)

plot(p1, p3, p5, p7, p2, p4, p6, p8, layout=grid(2,4, widths=(0.215,0.215,0.215,0.355)), size=(1800,800), bottommargin=10Plots.mm, dpi=300)
savefig("analysis/figures/metric_correlations.png")

## Plot some individual relationships of interest
# Sanity check of val vs test performance
p1 = plot_relationship(results["camelyon17"], "val_performance", "test_performance", p=plot(title="Camelyon17", xlabel="ID Val Performance", ylabel="ID Test Performance"), show_yeqx=true, scale_yeqx=false)
p2 = plot_relationship(results["iwildcam"], "val_performance", "test_performance", p=plot(title="iWildCam", xlabel="ID Val Performance", ylabel="ID Test Performance"), show_yeqx=true, scale_yeqx=false)
p3 = plot_relationship(results["fmow"], "val_performance", "test_performance", p=plot(title="fMoW", xlabel="ID Val Performance", ylabel="ID Test Performance"), show_yeqx=true, scale_yeqx=false)
p4 = plot_relationship(results["rxrx1"], "val_performance", "test_performance", p=plot(title="RxRx1", xlabel="ID Val Performance", ylabel="ID Test Performance"), show_yeqx=true, scale_yeqx=false)
plot(p1, p2, p3, p4, layout=(1, 4), size=(4*400, 400), dpi=300, left_margin=10Plots.mm, bottom_margin=10Plots.mm)
savefig("analysis/figures/test_vs_val_accuracy.png")

## FT comparison
p1 = plot_relationship(results["iwildcam"], "val_performance", "test_performance", p=plot(title="iWildCam", xlabel="ID Val Performance", ylabel="ID Test Performance"), show_yeqx=true, scale_yeqx=false)
p2 = plot_relationship(ft_results["iwildcam"], "val_performance", "test_performance", p=plot(title="iWildCam", xlabel="ID Val Performance", ylabel="ID Test Performance"), show_yeqx=true, scale_yeqx=false)


# Performance vs ds performance
p1 = plot_relationship(results["camelyon17"], "test_performance", "test_ds_performance", p=plot(title="Camelyon17", xlabel="ID Test Performance", ylabel="DS Test Performance"), show_yeqx=true, scale_yeqx=false)
p2 = plot_relationship(results["iwildcam"], "test_performance", "test_ds_performance", p=plot(title="iWildCam", xlabel="ID Test Performance", ylabel="DS Test Performance"), show_yeqx=true, scale_yeqx=false)
p3 = plot_relationship(results["fmow"], "test_performance", "test_ds_performance", p=plot(title="fMoW", xlabel="ID Test Performance", ylabel="DS Test Performance"), show_yeqx=true, scale_yeqx=false)
p4 = plot_relationship(results["rxrx1"], "test_performance", "test_ds_performance", p=plot(title="RxRx1", xlabel="ID Test Performance", ylabel="DS Test Performance"), show_yeqx=true, scale_yeqx=false)
plot(p1, p2, p3, p4, layout=(1, 4), size=(4*400, 400), dpi=300, left_margin=10Plots.mm, bottom_margin=10Plots.mm)
savefig("analysis/figures/ds_vs_id_performance")

p1 = plot_relationship(results["iwildcam"], "test_performance", "test_ds_performance", p=plot(title="iWildCam", xlabel="ID Test Performance", ylabel="DS Test Performance"), show_yeqx=true, scale_yeqx=false)
p2 = plot_relationship(ft_results["iwildcam"], "test_performance", "test_ds_performance", p=plot(title="iWildCam", xlabel="ID Test Performance", ylabel="DS Test Performance"), show_yeqx=true, scale_yeqx=false)

# DS calibration vs ID calibration
p1 = plot_relationship(results["camelyon17"], "camelyon17-id_val_calibration", "camelyon17-test_calibration", p=plot(title="Camelyon17", xlabel="ID Test Calibration", ylabel="DS Test Calibration"), show_yeqx=true, scale_yeqx=false)
p2 = plot_relationship(results["iwildcam"], "iwildcam-id_val_calibration", "iwildcam-test_calibration", p=plot(title="iWildCam", xlabel="ID Test Calibration", ylabel="DS Test Calibration"), show_yeqx=true, scale_yeqx=false)
p3 = plot_relationship(results["fmow"], "fmow-id_val_calibration", "fmow-test_calibration", p=plot(title="fMoW", xlabel="ID Test Calibration", ylabel="DS Test Calibration"), show_yeqx=true, scale_yeqx=false)
p4 = plot_relationship(results["rxrx1"], "rxrx1-id_val_calibration", "rxrx1-test_calibration", p=plot(title="RxRx1", xlabel="ID Test Calibration", ylabel="DS Test Calibration"), show_yeqx=true, scale_yeqx=false)
plot(p1, p2, p3, p4, layout=(1, 4), size=(4*400, 400), dpi=300, left_margin=10Plots.mm, bottom_margin=10Plots.mm)
savefig("analysis/figures/ds_vs_id_calibration")

p1 = plot_relationship(results["iwildcam"], "iwildcam-id_val_calibration", "iwildcam-test_calibration", p=plot(title="iWildCam", xlabel="ID Test Calibration", ylabel="DS Test Calibration"), show_yeqx=true, scale_yeqx=false)
p2 = plot_relationship(ft_results["iwildcam"], "iwildcam-id_val_calibration", "iwildcam-test_calibration", p=plot(title="iWildCam", xlabel="ID Test Calibration", ylabel="DS Test Calibration"), show_yeqx=true, scale_yeqx=false)


# Performance and robustness
p1 = plot_relationship(results["camelyon17"], "test_performance", "test_robustness", p=plot(title="Camelyon17", xlabel="ID Test Performance", ylabel="Test Robustness"))
p2 = plot_relationship(results["iwildcam"], "test_performance", "test_robustness", p=plot(title="iWildCam", xlabel="ID Test Performance", ylabel="Test Robustness"))
p3 = plot_relationship(results["fmow"], "test_performance", "test_robustness", p=plot(title="fMoW", xlabel="ID Test Performance", ylabel="Test Robustness"))
p4 = plot_relationship(results["rxrx1"], "test_performance", "test_robustness", p=plot(title="RxRx1", xlabel="ID Test Performance", ylabel="Test Robustness"))
plot(p1, p2, p3, p4, layout=(1, 4), size=(4*400, 400), dpi=300, left_margin=10Plots.mm, bottom_margin=10Plots.mm)

p1 = plot_relationship(results["iwildcam"], "test_performance", "test_robustness", p=plot(title="iWildCam", xlabel="ID Test Performance", ylabel="Test Robustness"))
p2 = plot_relationship(ft_results["iwildcam"], "test_performance", "test_robustness", p=plot(title="iWildCam", xlabel="ID Test Performance", ylabel="Test Robustness"))


p1 = plot_relationship(results["iwildcam"], "test_robustness", "test_calibration", p=plot(title="iWildCam", xlabel="Test Robustness", ylabel="Test Calibration"))
p2 = plot_relationship(ft_results["iwildcam"], "test_robustness", "test_calibration", p=plot(title="iWildCam", xlabel="Test Robustness", ylabel="Test Calibration"))



# Performance and calibration
p1 = plot_relationship(results["camelyon17"], "test_performance", "test_calibration", p=plot(title="Camelyon17", xlabel="ID Test Performance", ylabel="Test Calibration"))
p2 = plot_relationship(results["iwildcam"], "test_performance", "test_calibration", p=plot(title="iWildCam", xlabel="ID Test Performance", ylabel="Test Calibration"))
p3 = plot_relationship(results["fmow"], "test_performance", "test_calibration", p=plot(title="fMoW", xlabel="ID Test Performance", ylabel="Test Calibration"))
p4 = plot_relationship(results["rxrx1"], "test_performance", "test_calibration", p=plot(title="RxRx1", xlabel="ID Test Performance", ylabel="Test Calibration"))
plot(p1, p2, p3, p4, layout=(1, 4), size=(4*400, 400), dpi=300, left_margin=10Plots.mm, bottom_margin=10Plots.mm)
savefig("analysis/figures/calibration_vs_performance.png")

p1 = plot_relationship(results["iwildcam"], "test_performance", "test_calibration", p=plot(title="iWildCam", xlabel="ID Test Performance", ylabel="Test Calibration"))
p2 = plot_relationship(ft_results["iwildcam"], "test_performance", "test_calibration", p=plot(title="iWildCam", xlabel="ID Test Performance", ylabel="Test Calibration"))


# Performance and security
p1 = plot_relationship(results["camelyon17"], "test_performance", "test_security", p=plot(title="Camelyon17", xlabel="ID Test Performance", ylabel="Test Security"))
p2 = plot_relationship(results["iwildcam"], "test_performance", "test_security", p=plot(title="iWildCam", xlabel="ID Test Performance", ylabel="Test Security"))
p3 = plot_relationship(results["fmow"], "test_performance", "test_security", p=plot(title="fMoW", xlabel="ID Test Performance", ylabel="Test Security"))
p4 = plot_relationship(results["rxrx1"], "test_performance", "test_security", p=plot(title="RxRx1", xlabel="ID Test Performance", ylabel="Test Security"))
plot(p1, p2, p3, p4, layout=(1, 4), size=(4*400, 400), dpi=300, left_margin=10Plots.mm, bottom_margin=10Plots.mm)
savefig("analysis/figures/security_vs_perf.png")

p1 = plot_relationship(results["iwildcam"], "test_performance", "test_security", p=plot(title="iWildCam", xlabel="ID Test Performance", ylabel="Test Security"))
p2 = plot_relationship(ft_results["iwildcam"], "test_performance", "test_security", p=plot(title="iWildCam", xlabel="ID Test Performance", ylabel="Test Security"))


# Performance and fault detection
p1 = plot_relationship(results["camelyon17"], "test_performance", "test_ood_detection", p=plot(title="Camelyon17", xlabel="ID Test Performance", ylabel="Test Fault Detection"))
p2 = plot_relationship(results["iwildcam"], "test_performance", "test_ood_detection", p=plot(title="iWildCam", xlabel="ID Test Performance", ylabel="Test Fault Detection"))
p3 = plot_relationship(results["fmow"], "test_performance", "test_ood_detection", p=plot(title="fMoW", xlabel="ID Test Performance", ylabel="Test Fault Detection"))
p4 = plot_relationship(results["rxrx1"], "test_performance", "test_ood_detection", p=plot(title="RxRx1", xlabel="ID Test Performance", ylabel="Test Fault Detection"))
plot(p1, p2, p3, p4, layout=(1, 4), size=(4*450, 450), dpi=300, left_margin=10Plots.mm, bottom_margin=10Plots.mm)
savefig("analysis/figures/fault_detection_vs_perf.png")

p1 = plot_relationship(results["iwildcam"], "test_performance", "test_ood_detection", p=plot(title="iWildCam", xlabel="ID Test Performance", ylabel="Test Fault Detection"))
p2 = plot_relationship(ft_results["iwildcam"], "test_performance", "test_ood_detection", p=plot(title="iWildCam", xlabel="ID Test Performance", ylabel="Test Fault Detection"))


# Check val and test hre performance
p1 = plot_relationship(results["camelyon17"], "val_hre_score", "test_hre_score", p=plot(title="Camelyon17", xlabel="HRE (Val)", ylabel="HRE (Test)"), show_yeqx=true, scale_yeqx=false)
p2 = plot_relationship(results["iwildcam"], "val_hre_score", "test_hre_score", p=plot(title="iWildCam", xlabel="HRE (Val)", ylabel="HRE (Test)"), show_yeqx=true, scale_yeqx=false)
p3 = plot_relationship(results["fmow"], "val_hre_score", "test_hre_score", p=plot(title="fMoW", xlabel="HRE (Val)", ylabel="HRE (Test)"), show_yeqx=true, scale_yeqx=false)
p4 = plot_relationship(results["rxrx1"], "val_hre_score", "test_hre_score", p=plot(title="RxRx1", xlabel="HRE (Val)", ylabel="HRE (Test)"), show_yeqx=true, scale_yeqx=false)
plot(p1, p2, p3, p4, layout=(1, 4), size=(4*400, 400), dpi=300, left_margin=10Plots.mm, bottom_margin=10Plots.mm)
savefig("analysis/figures/test_hre_vs_val_hre.png")

p1 = plot_relationship(results["iwildcam"], "val_hre_score", "test_hre_score", p=plot(title="iWildCam", xlabel="HRE (Val)", ylabel="HRE (Test)"), show_yeqx=true, scale_yeqx=false)
p2 = plot_relationship(ft_results["iwildcam"], "val_hre_score", "test_hre_score", p=plot(title="iWildCam", xlabel="HRE (Val)", ylabel="HRE (Test)"), show_yeqx=true, scale_yeqx=false)



## Compare individual algorithms across metrics (one plot per dataset)
# syms_fn(prefix) = ["$(prefix)_performance","$(prefix)_robustness","$(prefix)_security","$(prefix)_calibration","$(prefix)_ood_detection"]
# names = ["Performance", "Robustness", "Security", "Calibration", "Fault Detection"]
# p1 = parallel_coord_by_alg(results["camelyon17"], syms_fn, names, prefix="val")
# p2 = parallel_coord_by_alg(results["camelyon17"], syms_fn, names, prefix="test")

# p3 = parallel_coord_by_alg(results["iwildcam"], syms_fn, names, prefix="val")
# p4 = parallel_coord_by_alg(results["iwildcam"], syms_fn, names, prefix="test")

# p5 = parallel_coord_by_alg(results["fmow"], syms_fn, names, prefix="val")
# p6 = parallel_coord_by_alg(results["fmow"], syms_fn, names, prefix="test")

# p7 = parallel_coord_by_alg(results["rxrx1"], syms_fn, names, prefix="val")
# p8 = parallel_coord_by_alg(results["rxrx1"], syms_fn, names, prefix="test")
# plot(p1, p2, p3, p4, p5, p6, p7, p8, layout=(4, 2), size=(2000, 400*4))

## Compare HRE scores for each
# p1 = scatter_scores(results["camelyon17"]; metric="val_hre_score", p=plot(ylabel="HRE Score (Val)"))
# p2 = scatter_scores(results["camelyon17"]; metric="test_hre_score", p=plot(ylabel="HRE Score (Test)"))

# p3 = scatter_scores(results["iwildcam"]; metric="val_hre_score", p=plot(ylabel="HRE Score (Val)"))
# p4 = scatter_scores(results["iwildcam"]; metric="test_hre_score", p=plot(ylabel="HRE Score (Test)"))

# p5 = scatter_scores(results["fmow"]; metric="val_hre_score", p=plot(ylabel="HRE Score (Val)"))
# p6 = scatter_scores(results["fmow"]; metric="test_hre_score", p=plot(ylabel="HRE Score (Test)"))

# p7 = scatter_scores(results["rxrx1"]; metric="val_hre_score", p=plot(ylabel="HRE Score (Val)"))
# p8 = scatter_scores(results["rxrx1"]; metric="test_hre_score", p=plot(ylabel="HRE Score (Test)"))
# plot(p1, p2, p3, p4, p5, p6, p7, p8, layout=(4, 2), size=(600*2, 400*4))
# savefig("analysis/figures/hre_plots.png")

function scatter_plots(metric, metric_name)
    p1 = scatter_scores(results["camelyon17"]; metric, p=plot(ylabel=metric_name, title="Cameylon17 $metric_name"))
    p2 = scatter_scores(results["iwildcam"]; metric, p=plot(ylabel=metric_name, title="iWildCam $metric_name"))
    p3 = scatter_scores(results["fmow"]; metric, p=plot(ylabel=metric_name, title="fMoW $metric_name"))
    p4 = scatter_scores(results["rxrx1"]; metric, p=plot(ylabel=metric_name, title="RxRx1 $metric_name"))
    plot(p1, p2, p3, p4, layout=grid(1, 4, widths=(0.28, 0.28, 0.28, 0.16)), size=(400*4, 400), bottom_margin=30Plots.mm, left_margin=10Plots.mm, dpi=300)
end

scatter_plots("val_performance", "Performance (Val)")
savefig("analysis/figures/val_performance_plots.png")

metric, metric_name = "val_performance", "Performance (Val)"
p = scatter_scores(all_results["iwildcam"]; metric, p=plot(ylabel=metric_name, title="iWildCam $metric_name"))

scatter_plots("test_performance", "Performance (Test)")
savefig("analysis/figures/test_performance_plots.png")

metric, metric_name = "test_performance", "Performance (Test)"
p = scatter_scores(all_results["iwildcam"]; metric, p=plot(ylabel=metric_name, title="iWildCam $metric_name"))

scatter_plots("val_ds_performance", "DS Performance (Val)")
savefig("analysis/figures/val_ds_performance_plots.png")

metric, metric_name = "val_ds_performance", "DS Performance (Val)"
p = scatter_scores(all_results["iwildcam"]; metric, p=plot(ylabel=metric_name, title="iWildCam $metric_name"))

scatter_plots("test_ds_performance", "DS Performance (Test)")
savefig("analysis/figures/test_ds_performance_plots.png")

metric, metric_name = "test_ds_performance", "DS Performance (Test)"
p = scatter_scores(all_results["iwildcam"]; metric, p=plot(ylabel=metric_name, title="iWildCam $metric_name"))

scatter_plots("val_robustness", "Robustness (Val)")
savefig("analysis/figures/val_robustness_plots.png")

metric, metric_name = "val_robustness", "Robustness (Val)"
p = scatter_scores(all_results["iwildcam"]; metric, p=plot(ylabel=metric_name, title="iWildCam $metric_name"))

scatter_plots("test_robustness", "Robustness (Test)")
savefig("analysis/figures/test_robustness_plots.png")

metric, metric_name = "test_robustness", "Robustness (Test)"
p = scatter_scores(all_results["iwildcam"]; metric, p=plot(ylabel=metric_name, title="iWildCam $metric_name"))


scatter_plots("val_security", "Security (Val)")
savefig("analysis/figures/val_security_plots.png")

metric, metric_name = "val_security", "Security (Val)"
p = scatter_scores(all_results["iwildcam"]; metric, p=plot(ylabel=metric_name, title="iWildCam $metric_name"))

scatter_plots("test_security", "Security (Test)")
savefig("analysis/figures/test_security_plots.png")

metric, metric_name = "test_security", "Security (Test)"
p = scatter_scores(all_results["iwildcam"]; metric, p=plot(ylabel=metric_name, title="iWildCam $metric_name"))


scatter_plots("val_calibration", "Calibration (Val)")
savefig("analysis/figures/val_calibration_plots.png")

metric, metric_name = "val_calibration", "Calibration (Val)"
p = scatter_scores(all_results["iwildcam"]; metric, p=plot(ylabel=metric_name, title="iWildCam $metric_name"))


scatter_plots("test_calibration", "Calibration (Test)")
savefig("analysis/figures/test_calibration_plots.png")

metric, metric_name = "test_calibration", "Calibration (Test)"
p = scatter_scores(all_results["iwildcam"]; metric, p=plot(ylabel=metric_name, title="iWildCam $metric_name"))


scatter_plots("val_ood_detection", "Fault Detection (Val)")
savefig("analysis/figures/val_fault_detection_plots.png")

metric, metric_name = "val_ood_detection", "Fault Detection (Val)"
p = scatter_scores(all_results["iwildcam"]; metric, p=plot(ylabel=metric_name, title="iWildCam $metric_name"))

scatter_plots("test_ood_detection", "Fault Detection  (Test)")
savefig("analysis/figures/test_fault_detection_plots.png")

metric, metric_name = "test_ood_detection", "Fault Detection  (Test)"
p = scatter_scores(all_results["iwildcam"]; metric, p=plot(ylabel=metric_name, title="iWildCam $metric_name"))


scatter_plots("val_hre_score", "HRE Score (Val)")
savefig("analysis/figures/val_hre_score_plots.png")

metric, metric_name = "val_hre_score", "HRE Score (Val)"
p = scatter_scores(all_results["iwildcam"]; metric, p=plot(ylabel=metric_name, title="iWildCam $metric_name"))


scatter_plots("test_hre_score", "HRE Score (Test)")
savefig("analysis/figures/test_hre_score_plots.png")

metric, metric_name = "test_hre_score", "HRE Score (Test)"
p = scatter_scores(all_results["iwildcam"]; metric, p=plot(ylabel=metric_name, title="iWildCam $metric_name"))


## Plot the dataset specific performance of each model
p1 = scatter_scores(results["camelyon17"]; metric="camelyon17-test_performance", p=plot(ylabel="Performance", title="Cameylon17 (Test)"))
p2 = scatter_scores(results["iwildcam"]; metric="iwildcam-test_performance", p=plot(ylabel="Performance", title="iWildCam (Test)"))
p3 = scatter_scores(results["fmow"]; metric="fmow-test_performance", p=plot(ylabel="Performance", title="fMoW (Test)"))
p4 = scatter_scores(results["rxrx1"]; metric="rxrx1-test_performance", p=plot(ylabel="Performance", title="RxRx1 (Test)"))
plot(p1, p2, p3, p4, layout=grid(1, 4, widths=(0.28, 0.28, 0.28, 0.16)), size=(400*4, 400), bottom_margin=30Plots.mm, left_margin=10Plots.mm, dpi=300)
savefig("analysis/figures/wilds_test_performance.png")


p = scatter_scores(all_results["iwildcam"]; metric="iwildcam-test_performance", p=plot(ylabel="Performance", title="iWildCam (Test)"))



scatter_scores(all_results["iwildcam"]; metric="iwildcam-test_performance", p=plot(ylabel="Performance", title="iWildCam (Test)"))

scatter_scores(all_results["iwildcam"]; metric="test_hre_score", p=plot(ylabel="Performance", title="iWildCam (Test)"))
scatter_scores(all_results["iwildcam"]; metric="test_robustness", p=plot(ylabel="Performance", title="iWildCam (Test)"))
scatter_scores(all_results["iwildcam"]; metric="test_ds_performance", p=plot(ylabel="Performance", title="iWildCam (Test)"))
scatter_scores(all_results["iwildcam"]; metric="iwildcam-test_performance", p=plot(ylabel="Performance", title="iWildCam (Test)"))
scatter_scores(all_results["iwildcam"]; metric="iwildcam-id_test-corruption1_test_performance", p=plot(ylabel="Performance", title="iWildCam Corruption1(Test)"))

p1 = scatter_scores(results["camelyon17"]; metric="camelyon17-val_performance", p=plot(ylabel="Performance", title="Cameylon17 (Val)"))
p2 = scatter_scores(results["iwildcam"]; metric="iwildcam-val_performance", p=plot(ylabel="Performance", title="iWildCam (Val)"))
p3 = scatter_scores(results["fmow"]; metric="fmow-val_performance", p=plot(ylabel="Performance", title="fMoW (Val)"))
p4 = scatter_scores(results["rxrx1"]; metric="rxrx1-val_performance", p=plot(ylabel="Performance", title="RxRx1 (Val)"))
plot(p1, p2, p3, p4, layout=grid(1, 4, widths=(0.28, 0.28, 0.28, 0.16)), size=(400*4, 400), bottom_margin=30Plots.mm, left_margin=10Plots.mm, dpi=300)
savefig("analysis/figures/wilds_val_performance.png")

## Val HRE with finetuning
p1 = scatter_scores(results["camelyon17"]; metric="val_hre_score", p=plot(ylabel="HRE Score (Val)", title="Cameylon17 HRE Score (Val)"))
scatter!([16,16], [0.6704, 0.6402], color=string_to_color("finetune"), label="", markerstrokecolor=:auto)
hres, algs = get_all_by_alg(results["camelyon17"], "val_hre_score")
xticks!(1:length(algs)+1, [String.(algs)..., "ViT Finetune"], xrotation = 90)

p2 = scatter_scores(results["iwildcam"]; metric="val_hre_score", p=plot(ylabel="HRE Score (Val)", title="iWildCam HRE Score (Val)"))
scatter!([14,14,14], [0.701, 0.703, 0.698], color=string_to_color("finetune"), label="", markerstrokecolor=:auto)
hres, algs = get_all_by_alg(results["iwildcam"], "val_hre_score")
xticks!(1:length(algs)+1, [String.(algs)..., "ViT Finetune"], xrotation = 90)

p3 = scatter_scores(results["fmow"]; metric="val_hre_score", p=plot(ylabel="HRE Score (Val)", title="fMoW HRE Score (Val)"))
scatter!([16,16,16], [0.6223, 0.6176, 0.6133], color=string_to_color("finetune"), label="", markerstrokecolor=:auto)
hres, algs = get_all_by_alg(results["fmow"], "val_hre_score")
xticks!(1:length(algs)+1, [String.(algs)..., "ViT Finetune"], xrotation = 90)

p4 = scatter_scores(results["rxrx1"]; metric="val_hre_score", p=plot(ylabel="HRE Score (Val)", title="RxRx1 HRE Score (Val)"))
scatter!([5,5], [0.5156, 0.5081], color=string_to_color("finetune"), label="", markerstrokecolor=:auto, dpi=300)
hres, algs = get_all_by_alg(results["rxrx1"], "val_hre_score")
xticks!(1:length(algs)+1, [String.(algs)..., "ViT Finetune"], xrotation = 90)

plot(p1, p2, p3, p4, layout=grid(1, 4, widths=(0.28, 0.28, 0.28, 0.16)), size=(400*4, 400), bottom_margin=30Plots.mm, left_margin=10Plots.mm, dpi=300)
savefig("analysis/figures/val_hre_with_finetune.png")

## Relationship of performance and calibration metrics to different domain shifts
# Relationship of domain shift performance to in-distribution performance for a variety of datasets
p1 = domain_shift_performance_relationships("camelyon17", "Cameylon17")
p2 = domain_shift_performance_relationships("iwildcam", "iWildCam")
p3 = domain_shift_performance_relationships("fmow", "fMoW")
p4 = domain_shift_performance_relationships("rxrx1", "RxRx1")
plot(p1,p2,p3,p4, size=(600*4, 400*4), layout=(1, 4), bottom_margin=15Plots.mm, dpi=300)
savefig("analysis/figures/ds_performance_selection.png")

# # Relationship of domain shift robustness to in-distribution performance for a variety of datasets
# p1 = domain_shift_robustness_relationships("camelyon17")
# p2 = domain_shift_robustness_relationships("iwildcam")
# p3 = domain_shift_robustness_relationships("fmow")
# p4 = domain_shift_robustness_relationships("rxrx1")
# plot(p1,p2,p3,p4, size=(600*4, 400*8), layout=(1, 4))

# Compare the relationship between calibrations across datasets
p1 = domain_shift_calibration_relationships("camelyon17", "Cameylon17")
p2 = domain_shift_calibration_relationships("iwildcam", "iWildCam")
p3 = domain_shift_calibration_relationships("fmow", "fMoW")
p4 = domain_shift_calibration_relationships("rxrx1", "RxRx1")
plot(p1,p2,p3,p4, size=(600*4, 400*4), layout=(1, 4), bottom_margin=10Plots.mm, dpi=300)
savefig("analysis/figures/calibration_selection.png")