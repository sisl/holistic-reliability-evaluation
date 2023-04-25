include("plotting_utils.jl")

## intialize some important names and defaults
datasets = ["camelyon17", "iwildcam", "fmow", "rxrx1"]
dataset_names = ["Camelyon17", "iWildCam", "fMoW", "RxRx1"]

performance_metrics = ["test_performance", "test_robustness", "test_security", "test_calibration", "test_ood_detection", "test_hre_score"]
performance_metric_names = ["ID Performance", "DS Robustness", "Adv Robustness", "Calibration", "OOD Detection", "HRE score"]

baseline_algs = ["erm", "erm-v2"]


## Load the results
results_dir = "results/evaluation_results"
results = load_results(results_dir)

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

## Make some plots 

function hre_plot(results, comparison_algs)
    plots = []
    for (i, (metric, mectric_name)) in enumerate(zip(performance_metrics, performance_metric_names))
        p = single_metric_comp(results, metric, mectric_name, baseline_algs, comparison_algs, ytick_label=i==1)
        push!(plots, p)
    end
    plot(plots..., layout=(1,6), size=(1200,50*length(comparison_algs)), bottom_margin=10Plots.mm, dpi=300)
end

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

