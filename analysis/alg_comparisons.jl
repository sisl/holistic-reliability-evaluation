include("plotting_utils.jl")

#region: Intialize some important names and defaults
datasets = ["camelyon17", "iwildcam", "fmow", "rxrx1"]
dataset_names = ["Camelyon17", "iWildCam", "fMoW", "RxRx1"]

performance_metrics(prefix="test") = ["$(prefix)_performance", "$(prefix)_robustness", "$(prefix)_security", "$(prefix)_calibration", "$(prefix)_ood_detection", "$(prefix)_hre_score"]
performance_metric_names = ["ID Performance", "DS Robustness", "Adv Robustness", "Calibration", "OOD Detection", "HRE score"]
ranges = [(0.65, 1), (0.7,1.0), (0,0.75), (0,1), (0,1), (0.3,1)]
baseline_algs = ["erm", "erm-v2"]

function hre_plot(results, comparison_algs, prefix="test")
    plots = []
    for (i, (metric, mectric_name, range)) in enumerate(zip(performance_metrics(prefix), performance_metric_names, ranges))
        p = single_metric_comp(results, metric, mectric_name, baseline_algs, comparison_algs, ytick_label=i==1)
        plot!(xlims=range)

        push!(plots, p)
    end
    plot(plots..., layout=(1,6), size=(1200,100 + 25*length(comparison_algs)), bottom_margin=10Plots.mm, dpi=300)
end

function comparison_all_datasets(x, y, xlabel, ylabel, c, lb=0, ub=1)
    p = plot([lb,ub], [lb,ub], color=:black, linestyle=:dash, xlabel=xlabel, ylabel=ylabel, label="")

    scatter!(p, x, y, alpha=0.5, markerstrokewidth=0, markercolor=c, label="")
    
    for ((di, dataset), dname) in zip(enumerate(datasets), dataset_names)
        scatter!(p,[],[], markercolor=di, label=dname, alpha=0.5, markerstrokewidth=0, legend=true)
    end
    p
end
#endregion

#region: Load the results (and combine in useful ways for plotting)
results_dir = "new_results/"

wilds_pretrained = load_results(string(results_dir, "wilds_pretrained"))
wilds_pretrained_calibrated = load_results(string(results_dir, "wilds_pretrained_calibrated"))
self_trained = load_results(string(results_dir, "self_trained"))
self_trained_calibrated = load_results(string(results_dir, "self_trained_calibrated"))

results = merge_results(wilds_pretrained, self_trained, augment_alg_name="self")
results_calibrated = merge_results(wilds_pretrained_calibrated, self_trained_calibrated, augment_alg_name="self")
#endregion

#region: Compare erm models (pretrained from wilds vs the ones we trained)
results = merge_results(wilds_pretrained, self_trained, augment_alg_name="self")
comparison_algs = OrderedDict("ERM-Wilds" => ["erm"],
                              "ERM-V2-Wilds" => ["erm-v2"],
                              "ERM-Us" => ["erm-self"])
hre_plot(results["iwildcam"], comparison_algs)
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

comparison_all_datasets(cal_scores_uncalibrated, cal_scores_calibrated,"Original Calibration", "Temperature Scaled Calibration", dataset_indices)
comparison_all_datasets(adv_scores_uncalibrated, adv_scores_calibrated,"Original Adv Robustness", "Temperature Scaled Adv Robustness", dataset_indices)
comparison_all_datasets(ood_scores_uncalibrated, ood_scores_calibrated,"Original OOD Detection", "Temperature Scaled OOD Detection", dataset_indices)
#endregion

#region: Relationship between OOD detection strategies

all_oodd_energybased = Float64[]
all_oodd_maxsoftmax = Float64[]
all_oodd_maxlogit = Float64[]
all_oodd_odin = Float64[]
dataset_indices = Int64[]

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
            catch
                println("Skipping key: ", key, " version: ", version)
            end
        end
    end
end



p1 = comparison_all_datasets(all_oodd_energybased, all_oodd_maxsoftmax, "Energy Based", "Max Softmax", dataset_indices)
p2 = comparison_all_datasets(all_oodd_energybased, all_oodd_maxlogit, "Energy Based", "Max Logit", dataset_indices)
p3 = comparison_all_datasets(all_oodd_energybased, all_oodd_odin, "Energy Based", "ODIN", dataset_indices)
p4 = comparison_all_datasets(all_oodd_maxsoftmax, all_oodd_maxlogit, "Max Softmax", "Max Logit", dataset_indices)
p5 = comparison_all_datasets(all_oodd_maxsoftmax, all_oodd_odin, "Max Softmax", "ODIN", dataset_indices)
p6 = comparison_all_datasets(all_oodd_maxlogit, all_oodd_odin, "Max Logit", "ODIN", dataset_indices)


plot(p1, p2, p3, p4, p5, p6, layout=(2,3), size=(1000, 600), dpi=300, legend=false, left_margin=5mm, bottom_margin=5mm)

plot_correlations([all_oodd_energybased, all_oodd_maxsoftmax, all_oodd_maxlogit, all_oodd_odin], ["EnergyBased", "MaxSoftmax", "MaxLogit", "ODIN"], title="OOD Detection Correlations")



#endregion

#region: Justifying robustness metrics
dataset = "fmow"
all_id_acc = Float64[]
all_synth_acc = Float64[]
all_test_acc = Float64[]

# Loop through the intersection of keys of results and results_calibrated
for key in keys(results[dataset])
    for version in keys(results[dataset][key])

        # Get the scores
        try
            id_accuracy =  get_perf(results[dataset], "test_performance", key, version=version)
            synth_accuracy = get_perf(results[dataset], "$dataset-id_test-corruption1_test_performance", key, version=version)
            test_accuracy = get_perf(results[dataset], "$dataset-test_performance", key, version=version)
            ds_accuracy = get_perf(results[dataset], "test_ds_performance", key, version=version)

            push!(all_id_acc, id_accuracy)
            push!(all_synth_acc, synth_accuracy)
            push!(all_test_acc, test_accuracy)
            push!(all_ds_acc, ds_accuracy)
            
            # push!(dataset_indices, di)
        catch
            println("Skipping key: ", key, " version: ", version)
        end
    end
end



p1 = scatter(all_id_acc, all_synth_acc, xlabel="ID Accuracy", ylabel="C1 Accuracy", title="C1 Accuracy vs ID Accuracy", legend=false)
plot!([0,1], [0,1], color=:black, linestyle=:dash, label="")

p2 = scatter(all_id_acc, all_test_acc, xlabel="ID Accuracy", ylabel="Test Accuracy", title="Test Accuracy vs ID Accuracy", legend=false)
plot!([0,1], [0,1], color=:black, linestyle=:dash, label="")

p3 = scatter(all_synth_acc, all_ds_acc, xlabel="C1 Accuracy", ylabel="Test Accuracy", title="Test Accuracy vs C1 Accuracy", legend=false)
plot!([0,1], [0,1], color=:black, linestyle=:dash, label="")

p1 = scatter(all_id_acc, all_ds_acc, xlabel="ID Accuracy", ylabel="DS Accuracy", title="DS Accuracy vs ID Accuracy", legend=false)
plot!([0,1], [0,1], color=:black, linestyle=:dash, label="")


# contour(0:0.1:1, 0:0.1:1, (x,y)->min(y/x, 1))


dataset = "iwildcam"
datasetname="iWildCam"
# Plot Domain shifted performance vs in-distribution performance 
# p1 = plot_relationship(results[dataset], "val_performance", "$dataset-val_performance", p=plot(title="DS-Val vs ID Performance", xlabel="ID-Val Performance", ylabel="DS-Val Performance"))
p2 = plot_relationship(results[dataset], "val_performance", "$dataset-test_performance", p=plot(title=datasetname, xlabel="ID-Val Performance", ylabel="DS-Test Performance"), show_yeqx=true, scale_yeqx=false)

xvals, xalgs = get_all_by_alg(results[dataset], "val_performance", requires="$dataset-test_performance")
yvals, yalgs = get_all_by_alg(results[dataset], "$dataset-test_performance", requires="val_performance")

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

# Experiment: Effect of Loss functions
irm = ["irm"]
coral = ["deepCORAL"]
dro = ["groupDRO"]
comparison_algs = OrderedDict("IRM" => irm,
                              "CORAL" => coral,
                              "GroupDRO" => dro)

hre_plot(all_results["iwildcam"], comparison_algs)
savefig("invariant_loss.png")

# Experiment: Leveraging unlabeled data
coral = ["deepCORAL-Coarse"] # domain invariance
afn = ["AFN"] # domain invariance
dann = ["DANN"] # domain invariance
noisystudent = ["NoisyStudent-extraunlabeled"] # self-training
psuedolabvels = ["PseudoLabels"] # Self-training
fixmatch = ["FixMatch"] # Self-training
swav = ["SwAV"] # Contrastive pretraining
comparison_algs = OrderedDict("CORAL" => coral,
                              "AFN" => afn,
                              "DANN" => dann,
                              "NoisyStudent" => noisystudent,
                              "PseudoLabels" => psuedolabvels,
                              "FixMatch" => fixmatch,
                              "SwAV" => swav)
hre_plot(all_results["iwildcam"], comparison_algs)
savefig("unlabeled_data.png")

# Experiment: Effect of pretraining algorithm
sup_algs = ["torchvision_vit_b_16_IMAGENET1K_V1", "torchvision_vit_l_16_IMAGENET1K_V1"]
swag_algs = ["torchvision_vit_b_16_IMAGENET1K_SWAG_LINEAR_V1", "torchvision_vit_l_16_IMAGENET1K_SWAG_LINEAR_V1", "torchvision_vit_h_14_IMAGENET1K_SWAG_LINEAR_V1"]
clip_algs = ["open_clip_vit_b_16_openai", "open_clip_vit_l_14_openai", "open_clip_vit_h_14_laion2b_s32b_b79k"]
mae_algs = ["mae_vit_b_16_DEFAULT", "mae_vit_l_16_DEFAULT", "mae_vit_h_14_DEFAULT"]

comparison_algs = OrderedDict("Supervised" => sup_algs,
                              "SWAG" => swag_algs,
                              "CLIP" => clip_algs,
                              "MAE" => mae_algs)

hre_plot(all_results["iwildcam"], comparison_algs)
savefig("pretraining.png")

# Experiment: Effect of model type
convnext = ["torchvision_convnext_large_IMAGENET1K_V1"]
efficientnet = ["torchvision_efficientnet_v2_l_IMAGENET1K_V1"]
maxvit_t = ["torchvision_maxvit_t_IMAGENET1K_V1"]
swin_v2 = ["torchvision_swin_v2_b_IMAGENET1K_V1"]
vit_b = ["torchvision_vit_b_16_IMAGENET1K_V1"]
vit_l = ["torchvision_vit_l_16_IMAGENET1K_V1"]
comparison_algs = OrderedDict("ConvNext" => convnext,
                                "EfficientNet-V2" => efficientnet,
                                "MaxViT" => maxvit_t,
                                "SWIN_V2" => swin_v2,
                                "ViT-B" => vit_b,
                                "ViT-L" => vit_l)
hre_plot(all_results["iwildcam"], comparison_algs)
savefig("model_type.png")

# Experiment: Effect of model size
vit_b = ["mae_vit_b_16_DEFAULT", "torchvision_vit_b_16_IMAGENET1K_SWAG_LINEAR_V1", "torchvision_vit_b_16_IMAGENET1K_V1", "open_clip_vit_b_16_openai"]
vit_l = ["torchvision_vit_l_16_IMAGENET1K_V1", "torchvision_vit_l_16_IMAGENET1K_SWAG_LINEAR_V1", "open_clip_vit_l_14_openai", "mae_vit_l_16_DEFAULT"]
vit_h = ["open_clip_vit_h_14_laion2b_s32b_b79k", "mae_vit_h_14_DEFAULT", "torchvision_vit_h_14_IMAGENET1K_SWAG_LINEAR_V1"]
comparison_algs = OrderedDict("ViT-B" => vit_b,
                              "ViT-L" => vit_l,
                              "ViT-H" => vit_h)

hre_plot(all_results["iwildcam"], comparison_algs)
savefig("model_size.png")

# Experiment: Effect of data augmentation
randaug = ["erm-augment"]
augmix = ["augmix"]

comparison_algs = OrderedDict("RandAugment" => randaug,
                              "AugMix" => augmix)
hre_plot(all_results["iwildcam"], comparison_algs, "val")
savefig("dataset_augmentation.png")

# Experiment: Effect of adversarial Training
pgd_training = ["pgd_training"]

comparison_algs = OrderedDict("PGD" => pgd_training)
hre_plot(all_results["iwildcam"], comparison_algs, "val")
savefig("adversarial_training.png")
