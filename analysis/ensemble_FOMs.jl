include("plotting_utils.jl")

function plot_all_metrics_cumulative(dataset_ensemble_results, dataset_single_results, prefix, Nmodels)
    index = 1
    if prefix=="test"
        index = 2
    end
    ensemble_performances = [dataset_ensemble_results[join(["erm_$j" for j in 0:i], "_")]["version_0"][1][index, "$(prefix)_performance"] for i=0:Nmodels-1]
    ensemble_robustness = [dataset_ensemble_results[join(["erm_$j" for j in 0:i], "_")]["version_0"][1][index, "$(prefix)_robustness"] for i=0:Nmodels-1]
    ensemble_ds_performance = [dataset_ensemble_results[join(["erm_$j" for j in 0:i], "_")]["version_0"][1][index, "$(prefix)_ds_performance"] for i=0:Nmodels-1]
    ensemble_calibration = [dataset_ensemble_results[join(["erm_$j" for j in 0:i], "_")]["version_0"][1][index, "$(prefix)_calibration"] for i=0:Nmodels-1]
    ensemble_ood_detection = [dataset_ensemble_results[join(["erm_$j" for j in 0:i], "_")]["version_0"][1][index, "$(prefix)_ood_detection"] for i=0:Nmodels-1]
    # ensemble_hre_score = [dataset_ensemble_results[join(["erm_$j" for j in 0:i], "_")]["version_0"][1][index, "$(prefix)_hre_score"] for i=0:Nmodels-1]
    ensemble_hre_score = mean([ensemble_performances, ensemble_robustness, ensemble_calibration, ensemble_ood_detection])

    singlemodel_performances = [dataset_single_results["erm"]["version_$i"][1][index, "$(prefix)_performance"] for i=0:Nmodels-1]
    singlemodel_robustness = [dataset_single_results["erm"]["version_$i"][1][index, "$(prefix)_robustness"] for i=0:Nmodels-1]
    singlemodel_ds_performance = [dataset_single_results["erm"]["version_$i"][1][index, "$(prefix)_ds_performance"] for i=0:Nmodels-1]
    singlemodel_calibration = [dataset_single_results["erm"]["version_$i"][1][index, "$(prefix)_calibration"] for i=0:Nmodels-1]
    singlemodel_ood_detection = [dataset_single_results["erm"]["version_$i"][1][index, "$(prefix)_ood_detection"] for i=0:Nmodels-1]
    # singlemodel_hre_score = [dataset_single_results["erm"]["version_$i"][1][index, "$(prefix)_hre_score"] for i=0:Nmodels-1]
    singlemodel_hre_score = mean([singlemodel_performances, singlemodel_robustness, singlemodel_calibration, singlemodel_ood_detection])

    p1 = plot(1:Nmodels, ensemble_performances, label="Ensemble", title="Performance vs Model")
    plot!(1:Nmodels, singlemodel_performances, label="Single model")

    p2 = plot(1:Nmodels, ensemble_robustness, label="Ensemble", title="Robustness vs Model")
    plot!(1:Nmodels, singlemodel_robustness, label="Single model")

    p3 = plot(1:Nmodels, ensemble_ds_performance, label="Ensemble", title="DS Performance vs Model")
    plot!(1:Nmodels, singlemodel_ds_performance, label="Single model")

    p4 = plot(1:Nmodels, ensemble_calibration, label="Ensemble", title="Calibration vs Model")
    plot!(1:Nmodels, singlemodel_calibration, label="Single model")

    p5 = plot(1:Nmodels, ensemble_ood_detection, label="Ensemble", title="Fault Detection vs Model")
    plot!(1:Nmodels, singlemodel_ood_detection, label="Single model")

    p6 = plot(1:Nmodels, ensemble_hre_score, label="Ensemble", title="HRE Score vs Model")
    plot!(1:Nmodels, singlemodel_hre_score, label="Single model")

    plot(p1, p2, p3, p4, p5, p6, layout=(3,2), size=(1000, 800))
end

## compare randomized ensembles with single models
function compare_metrics(sm, greedy, random, dataset, dataset_name)
    perfs = vcat(get_all_by_alg(sm[dataset], "test_performance")[1]...)
    robs = vcat(get_all_by_alg(sm[dataset], "test_robustness")[1]...)
    calibs = vcat(get_all_by_alg(sm[dataset], "test_calibration")[1]...)
    oods = vcat(get_all_by_alg(sm[dataset], "test_ood_detection")[1]...)
    hres = mean([perfs, robs, calibs, oods])

    ens_perfs = vcat(get_all_by_alg(greedy[dataset], "test_performance")[1]...)
    ens_robs = vcat(get_all_by_alg(greedy[dataset], "test_robustness")[1]...)
    ens_calibs = vcat(get_all_by_alg(greedy[dataset], "test_calibration")[1]...)
    ens_oods = vcat(get_all_by_alg(greedy[dataset], "test_ood_detection")[1]...)
    ens_hres = mean([ens_perfs, ens_robs, ens_calibs, ens_oods])

    rens_perfs = vcat(get_all_by_alg(random[dataset], "test_performance")[1]...)
    rens_robs = vcat(get_all_by_alg(random[dataset], "test_robustness")[1]...)
    rens_calibs = vcat(get_all_by_alg(random[dataset], "test_calibration")[1]...)
    rens_oods = vcat(get_all_by_alg(random[dataset], "test_ood_detection")[1]...)
    rens_hres = mean([rens_perfs, rens_robs, rens_calibs, rens_oods])

    scatter(perfs, hres, title = dataset_name, label="Single models", xlabel="ID Performance", ylabel="HRE", markerstrokecolor=:auto, alpha=0.7)
    scatter!(ens_perfs, ens_hres, label="Greedy Ensemble", color=3, markerstrokecolor=:auto, alpha=0.7)
    scatter!(rens_perfs, rens_hres, label="Random Ensemble", color=4, markerstrokecolor=:auto, alpha=0.7)
end

cumulative_ensemble = load_results("old_results/ensemble_results/cumulative")
single_model_results = load_results("old_results/evaluation_results")

plot_all_metrics_cumulative(cumulative_ensemble["camelyon17"], single_model_results["camelyon17"], "test", 10)
plot_all_metrics_cumulative(cumulative_ensemble["iwildcam"], single_model_results["iwildcam"], "test", 3)
plot_all_metrics_cumulative(cumulative_ensemble["fmow"], single_model_results["fmow"], "test", 3)
plot_all_metrics_cumulative(cumulative_ensemble["rxrx1"], single_model_results["rxrx1"], "test", 3)


greedy_ensemble = load_results("old_results/ensemble_results/greedy")
random_ensemble = load_results("old_results/ensemble_results/random")

dataset = "camelyon17"
x = "val_performance"
y = "test_hre_score"

scatter(perfs, hres, label="single models")

cum_x = [v["version_0"][1][1, x] for (k,v) in cumulative_ensemble[dataset]]
cum_y = [v["version_0"][1][2, y] for (k,v) in cumulative_ensemble[dataset]]

scatter(cum_x, cum_y, label="Cumulative Ensemble", title=string(y, " vs ", x))

greedy_x = [v["version_0"][1][1, x] for (k,v) in greedy_ensemble[dataset]]
greedy_y = [v["version_0"][1][2, y] for (k,v) in greedy_ensemble[dataset]]

scatter!(greedy_x, greedy_y, label="Greedy Ensemble")

random_x = [v["version_0"][1][1, x] for (k,v) in random_ensemble[dataset]]
random_y = [v["version_0"][1][2, y] for (k,v) in random_ensemble[dataset]]

scatter!(random_x, random_y, label="Random Ensemble")

p1 = compare_metrics(single_model_results, greedy_ensemble, random_ensemble, "camelyon17", "Camelyon17")
p2 = compare_metrics(single_model_results, greedy_ensemble, random_ensemble, "iwildcam", "iWildCam")
p3 = compare_metrics(single_model_results, greedy_ensemble, random_ensemble, "fmow", "fMoW")
p4 = compare_metrics(single_model_results, greedy_ensemble, random_ensemble, "rxrx1", "RxRx1")
plot(p1, p2, p3, p4, layout=(1,4), size=(400*4, 400), left_margin=10Plots.mm, bottom_margin=10Plots.mm, dpi=300)

savefig("analysis/figures/ensemble_comparison.png")


# Compare performance vs model size:

function single_metric_comp_ensemble(results, baseline_results, metric, metric_name, baseline_algs; ytick_label=true)
    index = 1
    if occursin("test",  metric)
        index = 2
    end
    
    num_models = [sum(isdigit(c) for c in replace(k, "v2"=>"")) for (k,v) in results]
    metrics = [v["version_0"][1][index, metric] for (k,v) in results]
    
    p = plot()
    N = maximum(num_models)
    ys = [-0.1, 0.1*N]

    # If comparing to a baseline
    if !isnothing(baseline_algs)
        baseline = vcat([get_perf(baseline_results, metric, alg) for alg in baseline_algs]...)
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
        yticks!(0:0.1:0.1*(N-1), string.(collect(1:N)))
    else
        yticks!(0:0.1:0.1*(N-1), fill("", N))
    end

    
    for nm in sort(unique(num_models))
        alg_i = num_models .== nm
        scatter!(metrics[alg_i], (nm-1)*0.1*ones(sum(alg_i)), alpha=0.7, markersize=7, markerstrokewidth=0)
    end
    yflip!(true)
    p
end

datasets = ["iwildcam", "fmow", "camelyon17",]#, "rxrx1"]
dataset_names = ["iWildCam", "fMoW", "Camelyon17"]#, "RxRx1"]

performance_metrics(prefix="test") = ["$(prefix)_performance", "$(prefix)_robustness", "$(prefix)_security", "$(prefix)_calibration", "$(prefix)_ood_detection", "$(prefix)_hre_score"]
performance_metric_names = ["ID Performance", "DS Robustness", "Adv Robustness", "Calibration", "OOD Detection", "HRE score"]
ranges = [(0.5, 1), (0.7,1.0), (0,0.75), (0.2,1), (0.4,1), (0.3,1)]
baseline_algs = ["erm", "erm-v2"]

function hre_plot(ensemble_results, baseline_results, prefix="test")
    plots = []
    for (i, (metric, mectric_name, range)) in enumerate(zip(performance_metrics(prefix), performance_metric_names, ranges))
        p = single_metric_comp_ensemble(ensemble_results, baseline_results,  metric, mectric_name, baseline_algs, ytick_label=i==1)
        plot!(xlims=range)

        push!(plots, p)
    end
    num_models = [sum(isdigit(c) for c in replace(k, "v2"=>"")) for (k,v) in ensemble_results]

    plot(plots..., layout=(1,6), size=(1200,100 + 25*maximum(num_models)), bottom_margin=10Plots.mm, dpi=300)
end

for d in datasets
    hre_plot(random_ensemble[d], results[d])
    savefig("analysis/paper_figures/random_ensembling_$d.pdf")
end

function plot_metric_by_num_model(dataset, dataset_name, metric, metric_name)
    index = 1
    if occursin("test",  metric)
        index = 2
    end
    num_models = [sum(isdigit(c) for c in replace(k, "v2"=>"")) for (k,v) in random_ensemble[dataset]]
    hres = [v["version_0"][1][index, metric] for (k,v) in random_ensemble[dataset]]

    num_models_greedy = [sum(isdigit(c) for c in replace(k, "v2"=>"")) for (k,v) in greedy_ensemble[dataset]]
    hres_greedy = [v["version_0"][1][index, metric] for (k,v) in greedy_ensemble[dataset]]

    scatter(num_models, hres, label="Random Ensemble", color=4, alpha=0.7, title=dataset_name, xlabel="Number of Models", ylabel=metric_name, markerstrokecolor=:auto)
    scatter!(num_models_greedy, hres_greedy, label="Greedy Ensemble", color=3, alpha=0.7, markerstrokecolor=:auto)
end

p1 = plot_metric_by_num_model("camelyon17", "Camelyon17", "test_hre_score", "HRE (Test)")
p2 = plot_metric_by_num_model("iwildcam", "iWildCam", "test_hre_score", "HRE (Test)")
p3 = plot_metric_by_num_model("fmow", "fMoW", "test_hre_score", "HRE (Test)")
p4 = plot_metric_by_num_model("rxrx1", "RxRx1", "test_hre_score", "HRE (Test)")
plot(p1, p2, p3, p4, layout=(1,4), size=(400*4, 400), left_margin=10Plots.mm, bottom_margin=10Plots.mm, dpi=300)
savefig("analysis/figures/ensemble_size.png")

random_ensemble