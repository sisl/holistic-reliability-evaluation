include("plotting_utils.jl")

# Plotting defaults
pszw = 400
pszh = 350

#region: Intialize some important names and defaults
datasets = ["iwildcam", "fmow", "camelyon17"]#, "rxrx1"]
dataset_names = ["iWildCam", "FMoW", "Camelyon17"]#, "RxRx1"]

performance_metrics(prefix="test") = ["$(prefix)_performance", "$(prefix)_robustness", "$(prefix)_security", "$(prefix)_calibration", "$(prefix)_ood_detection", "$(prefix)_hre_score"]
performance_metric_names = ["ID Performance", "DS Robustness", "Adv Robustness", "Calibration", "OOD Detection", "HR score"]
ranges = [(0.5, 1), (0.6,1.0), (0,0.75), (0.2,1), (0.5,1), (0.3,1)]
baseline_algs = ["erm", "erm-v2"]

function hre_plot(results, comparison_algs, prefix="test")
    plots = []
    for (i, (metric, mectric_name, range)) in enumerate(zip(performance_metrics(prefix), performance_metric_names, ranges))
        p = single_metric_comp(results, metric, mectric_name, baseline_algs, comparison_algs, ytick_label=i==1, left_margin=-8mm, tickfontsize=18)
        plot!(xlims=range)

        push!(plots, p)
    end
    plot(plots..., layout=(1,6), size=(800,100 + 25*length(comparison_algs)),)
end

function comparison_all_datasets(x, y, xlabel, ylabel, c; lb=0, ub=1, title="", legend=false)
    p = plot([lb,ub], [lb,ub], color=:black, linestyle=:dash, xlabel=xlabel, ylabel=ylabel, label="", title=title, legend=legend)

    for ((di, dataset), dname) in zip(enumerate(datasets), dataset_names)
        xi = x[c .== di]
        yi = y[c .== di]
        scatter!(p,xi,yi, markercolor=di, label=dname, alpha=0.3, markerstrokealpha=0)
    end
    p
end
#endregion

#region: Load the results (and combine in useful ways for plotting)
results_dir = "results/"
wilds_pretrained = load_results(string(results_dir, "wilds_pretrained"))
wilds_pretrained_calibrated = load_results(string(results_dir, "wilds_pretrained_calibrated"))
self_trained = load_results(string(results_dir, "self_trained"))
self_trained_calibrated = load_results(string(results_dir, "self_trained_calibrated"))

# compute the number of models
function print_totals()
    total = 0
    for d in datasets
        for alg in keys(wilds_pretrained_calibrated[d])
            total += length(keys(wilds_pretrained_calibrated[d][alg]))
        end
    end
    print("Total WILDS pretrained: ", total)

    total = 0
    for d in datasets
        for alg in keys(self_trained[d])
            total += length(keys(self_trained[d][alg]))
        end
    end
    print("Total Self-Trained models: ", total)
end
print_totals()

results = merge_results(wilds_pretrained, self_trained, augment_alg_name="self")
results_calibrated = merge_results(wilds_pretrained_calibrated, self_trained_calibrated, augment_alg_name="self")
#endregion

#region: Compare erm models (pretrained from wilds vs the ones we trained)
# results = merge_results(wilds_pretrained, self_trained, augment_alg_name="self")
# comparison_algs = OrderedDict("ERM-Wilds" => ["erm"],
#                               "ERM-V2-Wilds" => ["erm-v2"],
#                               "ERM-Us" => ["erm-self"])
# hre_plot(results["iwildcam"], comparison_algs)
#endregion

#region: Measure performance drops and justifying multiple datasets metrics
all_id_acc = Float64[]
all_synth_acc = Float64[]
all_test_acc = Float64[]
all_val_acc = Float64[]
all_ds_acc = Float64[]
all_id_cal = Float64[]
all_synth_cal = Float64[]
all_test_cal = Float64[]
all_val_cal = Float64[]
all_ds_cal = Float64[]
dataset_indices = Int64[]
self_vs_wilds = Symbol[]

# Loop through the intersection of keys of results and results_calibrated
for (di, dataset) in enumerate(datasets)
    println("dataset: ", dataset, " index: ", di)
    if !(dataset in keys(results))
        continue
    end

    for key in keys(results[dataset])
        for version in keys(results[dataset][key])

            # Get the scores
            try
                id_accuracy =  get_perf(results[dataset], "test_performance", key, version=version)
                synth_accuracy = get_perf(results[dataset], "$dataset-id_test-corruption1_test_performance", key, version=version)
                test_accuracy = get_perf(results[dataset], "$dataset-test_performance", key, version=version)
                val_accuracy = get_perf(results[dataset], "$dataset-val_performance", key, version=version)
                ds_accuracy = get_perf(results[dataset], "test_ds_performance", key, version=version)

                id_calibration = get_perf(results[dataset], "$dataset-id_test_calibration", key, version=version)
                synth_calibration = get_perf(results[dataset], "$dataset-id_test-corruption1_test_calibration", key, version=version)
                test_calibration = get_perf(results[dataset], "$dataset-test_calibration", key, version=version)
                val_calibration = get_perf(results[dataset], "$dataset-val_calibration", key, version=version)
                ds_calibration = (synth_calibration + test_calibration) / 2

                push!(all_id_acc, id_accuracy)
                push!(all_synth_acc, synth_accuracy)
                push!(all_test_acc, test_accuracy)
                push!(all_val_acc, val_accuracy)
                push!(all_ds_acc, ds_accuracy)

                push!(all_id_cal, id_calibration)
                push!(all_synth_cal, synth_calibration)
                push!(all_test_cal, test_calibration)
                push!(all_val_cal, val_calibration)
                push!(all_ds_cal, ds_calibration)
                
                push!(dataset_indices, di)

                #TODO: Ignore ensembles for this
                if key in keys(self_trained[dataset]) && !(key in keys(wilds_pretrained[dataset]))
                    push!(self_vs_wilds, :self)
                elseif key in keys(wilds_pretrained[dataset]) && !(key in keys(self_trained[dataset])) && !(occursin("Ensemble",key))
                    push!(self_vs_wilds, :wilds)
                elseif occursin("Ensemble",key)
                    push!(self_vs_wilds, :ensemble)
                else
                    @error "What happened?"
                end
            catch e
                println("error: ", e)
                println("Skipping key: ", key, " version: ", version)
            end
        end
    end
end

wi = self_vs_wilds .== :wilds

# Plot the performance drop/accuracy on the line for the wilds only models. This motivates the need for domain shifted datasets
p1 = comparison_all_datasets(all_id_acc[wi], all_val_acc[wi], "ID Accuracy", "Val Accuracy", dataset_indices[wi], legend=:topleft)
p2 = comparison_all_datasets(all_id_acc[wi], all_test_acc[wi], "ID Accuracy", "Test Accuracy", dataset_indices[wi], legend=false)
p3 = comparison_all_datasets(all_id_acc[wi], all_synth_acc[wi], "ID Accuracy", "C1 Accuracy", dataset_indices[wi], legend=false)
plot(p1, p2, p3, layout=(1,3), size=(pszw*3, pszh))
savefig("analysis/paper_figures/performance_drop.pdf")

# Plot the performance comparison between the different shifted datasets - Shows that one does not predict the other necessarily
p1 = comparison_all_datasets(all_val_acc[wi], all_test_acc[wi], "Val Accuracy", "Test Accuracy", dataset_indices[wi], legend=:topleft)
p2 = comparison_all_datasets(all_synth_acc[wi], all_test_acc[wi], "C1 Accuracy", "Test Accuracy", dataset_indices[wi], legend=false)
plot(p1, p2, layout=(1,2), size=(pszw*2, pszh))
savefig("analysis/paper_figures/ds_performance_comparisons.pdf")


# Plot calibration drops
p1 = comparison_all_datasets(all_id_cal[wi], all_val_cal[wi], "ID Calibration", "Val Calibration", dataset_indices[wi], legend=:topleft)
p2 = comparison_all_datasets(all_id_cal[wi], all_test_cal[wi], "ID Calibration", "Test Calibration", dataset_indices[wi], legend=false)
p3 = comparison_all_datasets(all_id_cal[wi], all_synth_cal[wi], "ID Calibration", "C1 Calibration", dataset_indices[wi], legend=false)
plot(p1, p2, p3, layout=(1,3), size=(pszw*3, pszh))
savefig("analysis/paper_figures/calibration_drop.pdf")

p1 = comparison_all_datasets(all_val_cal[wi], all_test_cal[wi], "Val Calibration", "Test Calibration", dataset_indices[wi], legend=:topleft)
p2 = comparison_all_datasets(all_synth_cal[wi], all_test_cal[wi], "C1 Calibration", "Test Calibration", dataset_indices[wi], legend=false)
plot(p1, p2, layout=(1,2), size=(pszw*2, pszh))
savefig("analysis/paper_figures/ds_calibration_comparisons.pdf")

#endregion

#region: Relationship between OOD detection strategies

all_oodd_energybased = Float64[]
all_oodd_maxsoftmax = Float64[]
all_oodd_maxlogit = Float64[]
all_oodd_odin = Float64[]
dataset_indices = Int64[]
self_vs_wilds = Symbol[]

for (di, dataset) in enumerate(datasets)
    println("dataset: ", dataset, " index: ", di)
    if !(dataset in keys(results))
        continue
    end
    # Loop through the intersection of keys of results and results_calibrated
    for key in keys(results[dataset])
        for version in keys(results[dataset][key])

            # Get the scores
            try
                oodd_energybased =  get_perf(results[dataset], "test_ood_detection_EnergyBased", key, version=version)
                oodd_maxsoftmax = get_perf(results[dataset], "test_ood_detection_MaxSoftmax", key, version=version)
                oodd_maxlogit = get_perf(results[dataset], "test_ood_detection_MaxLogit", key, version=version)
                oodd_odin = get_perf(results[dataset], "test_ood_detection_ODIN", key, version=version)

                push!(all_oodd_energybased, oodd_energybased)
                push!(all_oodd_maxsoftmax, oodd_maxsoftmax)
                push!(all_oodd_maxlogit, oodd_maxlogit)
                push!(all_oodd_odin, oodd_odin)
                
                push!(dataset_indices, di)
                if key in keys(self_trained[dataset]) && !(key in keys(wilds_pretrained[dataset]))
                    push!(self_vs_wilds, :self)
                elseif key in keys(wilds_pretrained[dataset]) && !(key in keys(self_trained[dataset]))
                    push!(self_vs_wilds, :wilds)
                else
                    @error "What happened?"
                end
            catch
                println("Skipping key: ", key, " version: ", version)
            end
        end
    end
end


wi = self_vs_wilds .== :wilds

p1 = comparison_all_datasets(all_oodd_energybased[wi], all_oodd_maxsoftmax[wi], "Energy Based", "Max Softmax", dataset_indices[wi], lb=0.5, ub=1)
p2 = comparison_all_datasets(all_oodd_energybased[wi], all_oodd_maxlogit[wi], "Energy Based", "Max Logit", dataset_indices[wi], lb=0.5, ub=1, legend=:topleft)
p3 = comparison_all_datasets(all_oodd_energybased[wi], all_oodd_odin[wi], "Energy Based", "ODIN", dataset_indices[wi], lb=0.5, ub=1, legend=false)
p4 = comparison_all_datasets(all_oodd_maxsoftmax[wi], all_oodd_maxlogit[wi], "Max Softmax", "Max Logit", dataset_indices[wi], lb=0.5, ub=1)
p5 = comparison_all_datasets(all_oodd_maxsoftmax[wi], all_oodd_odin[wi], "Max Softmax", "ODIN", dataset_indices[wi], lb=0.5, ub=1)
p6 = comparison_all_datasets(all_oodd_maxlogit[wi], all_oodd_odin[wi], "Max Logit", "ODIN", dataset_indices[wi], lb=0.5, ub=1)

# plot(p1, p2, p3, p4, p5, p6, layout=(2,3), size=(1000, 600), dpi=300, legend=false, left_margin=5mm, bottom_margin=5mm)

plot_correlations([all_oodd_energybased, all_oodd_maxsoftmax, all_oodd_maxlogit, all_oodd_odin], ["Energy Based", "Max Softmax", "Max Logit", "ODIN"], title="OOD Detection Correlations", size=(pszw*0.93, pszh))
savefig("analysis/paper_figures/ood_detection_correlation.pdf")

plot(p2, p3, layout=(1,2), size=(pszw*2,pszh))
savefig("analysis/paper_figures/ood_detection_scatter.pdf")

#endregion

#region: Relationship between metrics
plots = []
for (d, dname) in zip(datasets, dataset_names)
    perf = vcat([get_perf(results[d], "test_performance", c) for c in keys(results[d])]...)
    ds_perf = vcat([get_perf(results[d], "test_ds_performance", c) for c in keys(results[d])]...)
    ds_rob = vcat([get_perf(results[d], "test_robustness", c) for c in keys(results[d])]...)
    adv_rob = vcat([get_perf(results[d], "test_security", c) for c in keys(results[d])]...)
    cal = vcat([get_perf(results[d], "test_calibration", c) for c in keys(results[d])]...)
    ood = vcat([get_perf(results[d], "test_ood_detection", c) for c in keys(results[d])]...)
    
    p = plot_correlations([perf, ds_rob, adv_rob, cal, ood], ["ID Performance", "DS Robustness", "Adv Robustness", "Calibration", "OOD Detection"], title=dname, annotate_rval=true, colorbar=d==datasets[end], ylabels=d==datasets[1])
    push!(plots, p)
end
plot(plots..., layout=grid(1, 3, widths=[0.333, 0.333, 0.333]), size=(pszw*2.8, pszh))
savefig("analysis/paper_figures/metric_correlations.pdf")

# Adversarial Robustness vs. ID Perf
plots = []
legends = [:topright, :topright, :bottomleft]
for (d, dname, legend) in zip(datasets, dataset_names, legends)
    perf = vcat([get_perf(results[d], "test_performance", c) for c in keys(results[d])]...)
    adv = vcat([get_perf(results[d], "test_security", c) for c in keys(results[d])]...)
    p = scatter_fit(perf, adv, p=plot(xlabel="ID Performance", ylabel="Adv Robustness", title=dname; legend))
    push!(plots, p)
end
plot(plots..., layout = (1, 3), size=(pszw*3, pszh))
savefig("analysis/paper_figures/adv_robustness_vs_id_perf.pdf")

# Adversarial Performance vs. ID Perf
plots = []
legends = [:topright, :topleft, :bottomleft]
for (d, dname, legend) in zip(datasets, dataset_names, legends)
    perf = vcat([get_perf(results[d], "test_performance", c) for c in keys(results[d])]...)
    adv = vcat([get_perf(results[d], "test_security", c) for c in keys(results[d])]...)
    p = scatter_fit(perf, adv.*perf, p=plot(xlabel="ID Performance", ylabel="Adv Performance", title=dname; legend))
    push!(plots, p)
end
plot(plots..., layout = (1, 3), size=(pszw*3, pszh))
savefig("analysis/paper_figures/adv_performance_vs_id_perf.pdf")

# Calibration vs. ID Performance
plots = []
legends  = [:bottomright, :bottomright, :topleft]
for (d, dname, legend) in zip(datasets, dataset_names, legends)
    perf = vcat([get_perf(results[d], "test_performance", c) for c in keys(results[d])]...)
    cal = vcat([get_perf(results[d], "test_calibration", c) for c in keys(results[d])]...)
    p = scatter_fit(perf, cal, p=plot(xlabel="ID Performance", ylabel="Calibration", title=dname; legend))
    push!(plots, p)
end
plot(plots..., layout = (1, 3), size=(pszw*3, pszh))
savefig("analysis/paper_figures/calibration_vs_id_perf.pdf")

# DS Robustness vs Calibration
plots = []
legends  = [:bottomright, :bottomright, :topleft]
for (d, dname, legend) in zip(datasets, dataset_names, legends)
    ds_rob = vcat([get_perf(results[d], "test_robustness", c) for c in keys(results[d])]...)
    cal = vcat([get_perf(results[d], "test_calibration", c) for c in keys(results[d])]...)
    p = scatter_fit(ds_rob, cal, p=plot(xlabel="DS Robustness", ylabel="Calibration", title=dname; legend))
    push!(plots, p)
end
plot(plots..., layout = (1, 3), size=(pszw*3, pszh))
savefig("analysis/paper_figures/cal_vs_ds_robustness.pdf")

# DS Robustness vs ID Performance
plots = []
legends  = [:bottomright, :bottomright, :bottomleft]
for (d, dname, legend) in zip(datasets, dataset_names, legends)
    perf = vcat([get_perf(results[d], "test_performance", c) for c in keys(results[d])]...)
    ds_rob = vcat([get_perf(results[d], "test_robustness", c) for c in keys(results[d])]...)
    p = scatter_fit(perf, ds_rob, p=plot(xlabel="ID Performance", ylabel="DS Robustness", title=dname; legend))
    push!(plots, p)
end
plot(plots..., layout = (1, 3), size=(pszw*3, pszh))
savefig("analysis/paper_figures/ds_robustness_vs_id_perf.pdf")


# OOD Detection vs ID Performance
plots = []
legends  = [:topleft, :topleft, :topleft]
for (d, dname, legend) in zip(datasets, dataset_names, legends)
    perf = vcat([get_perf(results[d], "test_performance", c) for c in keys(results[d])]...)
    ood = vcat([get_perf(results[d], "test_ood_detection", c) for c in keys(results[d])]...)
    p = scatter_fit(perf, ood, p=plot(xlabel="ID Performance", ylabel="OOD Detection", title=dname; legend))
    push!(plots, p)
end
plot(plots..., layout = (1, 3), size=(pszw*3, pszh))
savefig("analysis/paper_figures/ood_detection_vs_id_perf.pdf")

#endregion

#region: Effect of data augmentation
no_data_augs = ["no_data_augs"]
randaug = ["erm-augment", "randaug"]
augmix = ["augmix"]
adv = ["adversarial_sweep"]
comparison_algs = OrderedDict("RandAug" => randaug,
                              "AugMix" => augmix,
                            #   "None" => no_data_augs,
                              "\\phantom{iir}Adversarial" => adv)

randaug = ["erm-augment"]                  
comparison_algs_c17 = OrderedDict("RandAug" => randaug,
                              "AugMix" => augmix,
                              #   "None" => no_data_augs,\
                              "\\phantom{iir}Adversarial" => adv)

for d in datasets
    if d == "camelyon17"
        hre_plot(results[d], comparison_algs_c17)
        savefig("analysis/paper_figures/dataset_augmentation_$d.pdf")
    else
        hre_plot(results[d], comparison_algs)
        savefig("analysis/paper_figures/dataset_augmentation_$d.pdf")
    end
end
#endregion

#region: Effect of Loss functions
irm = ["irm"]
coral = ["deepCORAL"]
dro = ["groupDRO"]
comparison_algs = OrderedDict("IRM" => irm,
                              "CORAL" => coral,
                              "\\phantom{tp}GroupDRO" => dro)

for d in datasets
    hre_plot(results[d], comparison_algs)
    savefig("analysis/paper_figures/invariant_loss_$d.pdf")
end
#endregion

#region: Leveraging unlabeled data

for d in datasets
    coral = ["deepCORAL-Coarse"] # domain invariance
    afn = ["AFN"] # domain invariance
    dann = ["DANN"] # domain invariance
    if d in ["fmow", "camelyon17"]
        noisystudent = ["NoisyStudent-trainunlabeled", "NoisyStudent-testunlabeled", "NoisyStudent-valunlabeled"]
    else
        noisystudent = ["NoisyStudent-extraunlabeled"] # self-training
    end
    psuedolabels = ["PseudoLabels"] # Self-training
    fixmatch = ["FixMatch"] # Self-training
    swav = ["SwAV"] # Contrastive pretraining
    comparison_algs = OrderedDict("CORAL" => coral,
                                "AFN" => afn,
                                "DANN" => dann,
                                "NoisyStudent" => noisystudent,
                                "PseudoLabels" => psuedolabels,
                                "FixMatch" => fixmatch,
                                "SwAV" => swav)

    hre_plot(results[d], comparison_algs)
    savefig("analysis/paper_figures/unlabeled_data_$d.pdf")
end
#endregion

#region: Effect of pretraining algorithm
sup_algs = ["torchvision_vit_b_16_IMAGENET1K_V1", "torchvision_vit_l_16_IMAGENET1K_V1"]
swag_algs = ["torchvision_vit_b_16_IMAGENET1K_SWAG_LINEAR_V1", "torchvision_vit_l_16_IMAGENET1K_SWAG_LINEAR_V1", "torchvision_vit_h_14_IMAGENET1K_SWAG_LINEAR_V1"]
clip_algs = ["open_clip_vit_b_16_openai", "open_clip_vit_l_14_openai", "open_clip_vit_h_14_laion2b_s32b_b79k"]
mae_algs = ["mae_vit_b_16_DEFAULT", "mae_vit_l_16_DEFAULT", "mae_vit_h_14_DEFAULT"]

comparison_algs = OrderedDict(
                              "\\phantom{iiir}Supervised" => sup_algs,
                              "SWAG" => swag_algs,
                              "CLIP" => clip_algs,
                              "MAE" => mae_algs)

for d in datasets
    hre_plot(results[d], comparison_algs)
    savefig("analysis/paper_figures/pretraining_$d.pdf")
end
#endregion

#region: Effect of model type
convnext = ["torchvision_convnext_large_IMAGENET1K_V1"]
efficientnet = ["torchvision_efficientnet_v2_l_IMAGENET1K_V1"]
maxvit_t = ["torchvision_maxvit_t_IMAGENET1K_V1"]
swin_v2 = ["torchvision_swin_v2_b_IMAGENET1K_V1"]
vit_b = ["torchvision_vit_b_16_IMAGENET1K_V1"]
vit_l = ["torchvision_vit_l_16_IMAGENET1K_V1"]
comparison_algs = OrderedDict("ConvNext" => convnext,
                              "\\phantom{iii}EfficientNet" => efficientnet,
                                "MaxViT" => maxvit_t,
                                "SWIN-V2" => swin_v2,
                                "ViT-B" => vit_b,
                                "ViT-L" => vit_l)

for d in datasets
    hre_plot(results[d], comparison_algs)
    savefig("analysis/paper_figures/model_type_$d.pdf")
end
#endregion

#region: Effect of model size
vit_b = ["mae_vit_b_16_DEFAULT", "torchvision_vit_b_16_IMAGENET1K_SWAG_LINEAR_V1", "torchvision_vit_b_16_IMAGENET1K_V1", "open_clip_vit_b_16_openai"]
vit_l = ["torchvision_vit_l_16_IMAGENET1K_V1", "torchvision_vit_l_16_IMAGENET1K_SWAG_LINEAR_V1", "open_clip_vit_l_14_openai", "mae_vit_l_16_DEFAULT"]
vit_h = ["open_clip_vit_h_14_laion2b_s32b_b79k", "mae_vit_h_14_DEFAULT", "torchvision_vit_h_14_IMAGENET1K_SWAG_LINEAR_V1"]
comparison_algs = OrderedDict(
                              "\\phantom{iiiiiiiiip}ViT-H" => vit_b,
                              "ViT-L" => vit_l,
                              "ViT-H" => vit_h)


for d in datasets
    hre_plot(results[d], comparison_algs)
    savefig("analysis/paper_figures/model_size_$d.pdf")
end
#endregion

#region: Ensembling
comparison_algs = OrderedDict("1 Model" => ["Ensemble_1"],
                              "2 Models" => ["Ensemble_2"],
                              "3 Models" => ["Ensemble_3"],
                              "4 Models" => ["Ensemble_4"],
                              "\\phantom{iiiiiii}5 Models" => ["Ensemble_5"],
                             )


for d in datasets
    hre_plot(results[d], comparison_algs)
    savefig("analysis/paper_figures/ensembles_wilds_$d.pdf")
end


#endregion

#region: Effect of temperature scaling on calibration, adv robustness and ood detection

# Loop through every result and get the calibrated version
cal_scores_uncalibrated = Float64[]
cal_scores_calibrated = Float64[]
ood_scores_uncalibrated = Float64[]
ood_scores_calibrated = Float64[]
adv_scores_uncalibrated = Float64[]
adv_scores_calibrated = Float64[]

dataset_indices = Int64[]

for (di, dataset) in enumerate(datasets)
    println("dataset: ", dataset, " index: ", di)
    if !(dataset in keys(results)) || !(dataset in keys(results_calibrated))
        continue
    end
    # Loop through the intersection of keys of results and results_calibrated
    for key in intersect(keys(results[dataset]), keys(results_calibrated[dataset]))
        for version in intersect(keys(results[dataset][key]), keys(results_calibrated[dataset][key]))
            println("key:", key, "version: ", version)
            # Get the scores
            try
                calibration_uncalibrated =  get_perf(results[dataset], "test_calibration", key, version=version)
                calibration_calibrated = get_perf(results_calibrated[dataset], "test_calibration", key, version=version)

                if calibration_calibrated <= calibration_uncalibrated
                    println("==========> Weird circumstance for key: ", key, " version: ", version)
                end
                ood_uncalibrated = get_perf(results[dataset], "test_ood_detection", key, version=version)
                ood_calibrated = get_perf(results_calibrated[dataset], "test_ood_detection", key, version=version)
                adv_uncalibrated = get_perf(results[dataset], "test_security", key, version=version)
                adv_calibrated = get_perf(results_calibrated[dataset], "test_security", key, version=version)

                push!(cal_scores_uncalibrated, calibration_uncalibrated)
                push!(cal_scores_calibrated, calibration_calibrated)
                push!(ood_scores_uncalibrated, ood_uncalibrated)
                push!(ood_scores_calibrated, ood_calibrated)
                push!(adv_scores_uncalibrated, adv_uncalibrated)
                push!(adv_scores_calibrated, adv_calibrated)
                push!(dataset_indices, di)
            catch
                println("Skipping key: ", key, " version: ", version)
            end
        end
    end
end

p1 = comparison_all_datasets(adv_scores_uncalibrated, adv_scores_calibrated,"Original", "Temperature-Scaled", dataset_indices, title="Adversarial Robustness", legend=:topleft)
p2 = comparison_all_datasets(cal_scores_uncalibrated, cal_scores_calibrated,"Original", "Temperature-Scaled", dataset_indices, title="Calibration", legend=false)
p3 = comparison_all_datasets(ood_scores_uncalibrated, ood_scores_calibrated,"Original", "Temperature-Scaled", dataset_indices, title="OOD Detection", legend=false)

plot(p1,p2,p3, layout=(1,3), size=(pszw*3,pszh))
savefig("analysis/paper_figures/temperature_scaling.pdf")

#endregion

#region: Model selection (val vs test performance)
# val_perf = Float64[]
# test_perf = Float64[]
# val_ds_robustness = Float64[]
# test_ds_robustness = Float64[]
# val_adv_robustness = Float64[]
# test_adv_robustness = Float64[]
# val_calibration = Float64[]
# test_calibration = Float64[]
# val_ood = Float64[]
# test_ood = Float64[]
# val_hre = Float64[]
# test_hre = Float64[]

# dataset_indices = Int64[]

# for (di, dataset) in enumerate(datasets)
#     println("dataset: ", dataset, " index: ", di)
#     if !(dataset in keys(results))
#         continue
#     end

#     # Loop through the intersection of keys of results and results_calibrated
#     for key in keys(results[dataset])
#         if ! (key in keys(results_calibrated[dataset]))
#             @warn "Skipping key: $key"
#             continue
#         end
#         for version in intersect(keys(results[dataset][key]), keys(results_calibrated[dataset][key]))
#             println("key:", key, "version: ", version)
#             # Get the scores
#             try
#                 vp =  get_perf(results[dataset], "val_performance", key, version=version)
#                 tp =  get_perf(results[dataset], "test_performance", key, version=version)

#                 vds =  get_perf(results[dataset], "val_robustness", key, version=version)
#                 tds =  get_perf(results[dataset], "test_robustness", key, version=version)

#                 vsec =  get_perf(results[dataset], "val_security", key, version=version)
#                 tsec =  get_perf(results[dataset], "test_security", key, version=version)

#                 vcal =  get_perf(results[dataset], "val_calibration", key, version=version)
#                 tcal =  get_perf(results[dataset], "test_calibration", key, version=version)

#                 vood =  get_perf(results[dataset], "val_ood_detection", key, version=version)
#                 tood =  get_perf(results[dataset], "test_ood_detection", key, version=version)

#                 vhre =  get_perf(results[dataset], "val_hre_score", key, version=version)
#                 thre =  get_perf(results[dataset], "test_hre_score", key, version=version)

#                 push!(val_perf, vp)
#                 push!(test_perf, tp)
#                 push!(val_ds_robustness, vds)
#                 push!(test_ds_robustness, tds)
#                 push!(val_adv_robustness, vsec)
#                 push!(test_adv_robustness, tsec)
#                 push!(val_calibration, vcal)
#                 push!(test_calibration, tcal)
#                 push!(val_ood, vood)
#                 push!(test_ood, tood)
#                 push!(val_hre, vhre)
#                 push!(test_hre, thre)
                
#                 push!(dataset_indices, di)
#             catch
#                 println("Skipping key: ", key, " version: ", version)
#             end
#         end
#     end
# end

# p1 = comparison_all_datasets(val_perf, test_perf, "Val", "Test", dataset_indices, title="ID Performance", legend=true)
# p2 = comparison_all_datasets(val_ds_robustness, test_ds_robustness, "Val", "Test", dataset_indices, title="DS Robustness", legend=false)
# p3 = comparison_all_datasets(val_adv_robustness, test_adv_robustness, "Val", "Test", dataset_indices, title="Adv Robustness", legend=false)
# p4 = comparison_all_datasets(val_calibration, test_calibration, "Val", "Test", dataset_indices, title="Calibration", legend=false)
# p5 = comparison_all_datasets(val_ood, test_ood, "Val", "Test", dataset_indices, title="OOD Detection", legend=false)
# p6 = comparison_all_datasets(val_hre, test_hre, "Val", "Test", dataset_indices, title="HRE Score", legend=false)

# plot(p1,p2,p3,p4,p5,p6, layout=(2,3), size=(1000,600), dpi=300, left_margin=5mm, bottom_margin=5mm)
# savefig("analysis/paper_figures/model_selection.png")

#endregion