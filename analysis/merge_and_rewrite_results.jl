include("plotting_utils.jl")

results_dir = "results/"
wilds_pretrained = load_results(string(results_dir, "wilds_pretrained"))
wilds_pretrained_calibrated = load_results(string(results_dir, "wilds_pretrained_calibrated"))
self_trained = load_results(string(results_dir, "self_trained"))
self_trained_calibrated = load_results(string(results_dir, "self_trained_calibrated"))

results_dir = "calibration_results/"
cal_wilds_pretrained = load_results(string(results_dir, "wilds_pretrained"))
cal_wilds_pretrained_calibrated = load_results(string(results_dir, "wilds_pretrained_calibrated"))
cal_self_trained = load_results(string(results_dir, "self_trained"))
cal_self_trained_calibrated = load_results(string(results_dir, "self_trained_calibrated"))


# update the results
function merge!(updated_vals, orig_vals)
    println("Missing datasets: ", setdiff(keys(orig_vals), keys(updated_vals)))
    for dataset in keys(updated_vals)
        println("Missing algorithms for ", dataset, " keys: ", setdiff(keys(orig_vals[dataset]), keys(updated_vals[dataset])))
        for alg in keys(updated_vals[dataset])
            println("Missing versions: ", setdiff(keys(orig_vals[dataset][alg]), keys(updated_vals[dataset][alg])))
            for ver in keys(updated_vals[dataset][alg])
                target_dir = string("results/", dataset, "/", alg, "/", ver)
                df = updated_vals[dataset][alg][ver][1]
                for col in names(df)
                    try
                        orig_vals[dataset][alg][ver][1][!, col] = df[!, col]
                    catch
                        println("Skipping column ", col, " for ", target_dir)
                    end
                end
                try
                    # Update the HRE scores
                    vp = orig_vals[dataset][alg][ver][1][1, "val_performance_norm"]
                    vr = orig_vals[dataset][alg][ver][1][1, "val_robustness"]
                    vs = orig_vals[dataset][alg][ver][1][1, "val_security"]
                    vc = orig_vals[dataset][alg][ver][1][1, "val_calibration"]
                    vood = orig_vals[dataset][alg][ver][1][1, "val_ood_detection"]
                    vhre = orig_vals[dataset][alg][ver][1][1, "val_hre_score"]

                    tp = orig_vals[dataset][alg][ver][1][2, "test_performance_norm"]
                    tr = orig_vals[dataset][alg][ver][1][2, "test_robustness"]
                    ts = orig_vals[dataset][alg][ver][1][2, "test_security"]
                    tc = orig_vals[dataset][alg][ver][1][2, "test_calibration"]
                    tood = orig_vals[dataset][alg][ver][1][2, "test_ood_detection"]
                    thre = orig_vals[dataset][alg][ver][1][2, "test_hre_score"]

                    orig_vals[dataset][alg][ver][1][1, "val_hre_score"] = vhre
                    orig_vals[dataset][alg][ver][1][2, "test_hre_score"] = thre
                catch 
                    println("Skipping: ", target_dir)
                end
            end
        end
    end
end

merge!(cal_wilds_pretrained, wilds_pretrained)
merge!(cal_wilds_pretrained_calibrated, wilds_pretrained_calibrated)
merge!(cal_self_trained, self_trained)
merge!(cal_self_trained_calibrated, self_trained_calibrated)

## Rewrite the results

using CSV, YAML
function write_results(dir, results)
    try mkdir(dir) catch end
    for dataset in keys(results)
        try mkdir(string(dir, "/", dataset)) catch end
        for alg in keys(results[dataset])
            try mkdir(string(dir, "/", dataset, "/", alg)) catch end
            for ver in keys(results[dataset][alg])
                target_dir = string(dir,"/", dataset, "/", alg, "/", ver)
                try mkdir(target_dir) catch end
                yaml = results[dataset][alg][ver][2]
                df = results[dataset][alg][ver][1]
                CSV.write(string(target_dir, "/metrics.csv"), df)
                YAML.write_file(string(target_dir, "/hparams.yaml"), yaml)
            end
        end
    end
end

try mkdir("updated_results") catch end
write_results("updated_results/wilds_pretrained", wilds_pretrained)
write_results("updated_results/wilds_pretrained_calibrated", wilds_pretrained_calibrated)
write_results("updated_results/self_trained", self_trained)
write_results("updated_results/self_trained_calibrated", self_trained_calibrated)

