include("plotting_utils.jl")

## intialize some important names and defaults
datasets = ["camelyon17", "iwildcam", "fmow", "rxrx1"]
dataset_names = ["Camelyon17", "iWildCam", "fMoW", "RxRx1"]

performance_metrics(prefix="test") = ["$(prefix)_performance", "$(prefix)_robustness", "$(prefix)_security", "$(prefix)_calibration", "$(prefix)_ood_detection", "$(prefix)_hre_score"]
performance_metric_names = ["ID Performance", "DS Robustness", "Adv Robustness", "Calibration", "OOD Detection", "HRE score"]
ranges = [(0.65, 1), (0.7,1.0), (0,0.75), (0,1), (0,1), (0.3,1)]

baseline_algs = ["erm", "erm-v2"]


## Load the results
results_dir = "results/evaluation_results"
results = load_results(results_dir)

wetrain_results_dir = "/mnt/data/acorso/results/"
wetrain_results = load_results(wetrain_results_dir, self_trained=true)

# Put all the results together
all_results = deepcopy(results)
for (k, v) in wetrain_results["iwildcam-train"]
    all_results["iwildcam"][k] = v
end

## Make some plots 
function hre_plot(results, comparison_algs, prefix="test")
    plots = []
    for (i, (metric, mectric_name, range)) in enumerate(zip(performance_metrics(prefix), performance_metric_names, ranges))
        p = single_metric_comp(results, metric, mectric_name, baseline_algs, comparison_algs, ytick_label=i==1)
        plot!(xlims=range)

        push!(plots, p)
    end
    plot(plots..., layout=(1,6), size=(1200,100 + 25*length(comparison_algs)), bottom_margin=10Plots.mm, dpi=300)
end

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
