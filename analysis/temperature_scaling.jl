include("plotting_utils.jl")

results_dir = "results/evaluation_results"
results = load_results(results_dir)

results_w_Tscale_dir = "holistic_reliability_evaluation/results_w_Tscale"
results_w_Tscale = load_results(results_w_Tscale_dir)



all_calibrations, algs = get_all_by_alg(results["iwildcam"], "test_calibration")

all_calibrations_w_T, algs_w_T = get_all_by_alg(results_w_Tscale["iwildcam"], "test_calibration")

alg = "erm"

algs_w_T

calibrations = vcat([all_calibrations[algs .== alg][1] for alg in algs_w_T]...)
calibrations_w_T = vcat([all_calibrations_w_T[algs_w_T .== alg][1] for alg in algs_w_T]...)

scatter(calibrations[1:length(calibrations_w_T)], calibrations_w_T)
plot!([0,1], [0,1], color=:black, linestyle=:dash, legend=false, xlabel="Original Calibration", ylabel="Temperature Scaled Calibration", title="Effect of Temperature Scaling on Calibration", dpi=300)
savefig("analysis/figures/temperature_scale_effect.png")