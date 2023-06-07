python holistic_reliability_evaluation/evaluate.py --wilds_pretrained=true --dataset=iwildcam --model_dir=/scratch/users/acorso/wilds_models --save_dir=results/wilds_pretrained
python holistic_reliability_evaluation/evaluate.py --dataset=iwildcam --model_dir=/mnt/data/acorso/results --save_dir=results/self_trained
python holistic_reliability_evaluation/evaluate.py --wilds_pretrained=true --dataset=camelyon17 --model_dir=/scratch/users/acorso/wilds_models --save_dir=results/wilds_pretrained
python holistic_reliability_evaluation/evaluate.py --dataset=camelyon17 --model_dir=/mnt/data/acorso/results --save_dir=results/self_trained
python holistic_reliability_evaluation/evaluate.py --wilds_pretrained=true --dataset=fmow --model_dir=/scratch/users/acorso/wilds_models --save_dir=results/wilds_pretrained
python holistic_reliability_evaluation/evaluate.py --dataset=fmow --model_dir=/mnt/data/acorso/results --save_dir=results/self_trained
python holistic_reliability_evaluation/evaluate.py --wilds_pretrained=true --dataset=rxrx1 --model_dir=/scratch/users/acorso/wilds_models --save_dir=results/wilds_pretrained
python holistic_reliability_evaluation/evaluate.py --dataset=rxrx1 --model_dir=/mnt/data/acorso/results --save_dir=results/self_trained

python holistic_reliability_evaluation/evaluate.py --wilds_pretrained=true --dataset=iwildcam --calibration_method=temperature_scaling --model_dir=/scratch/users/acorso/wilds_models --save_dir=results/wilds_pretrained_calibrated
python holistic_reliability_evaluation/evaluate.py --dataset=iwildcam --calibration_method=temperature_scaling --model_dir=/mnt/data/acorso/results --save_dir=results/self_trained_calibrated
python holistic_reliability_evaluation/evaluate.py --wilds_pretrained=true --dataset=camelyon17 --calibration_method=temperature_scaling --model_dir=/scratch/users/acorso/wilds_models --save_dir=results/wilds_pretrained_calibrated
python holistic_reliability_evaluation/evaluate.py --dataset=camelyon17 --calibration_method=temperature_scaling --model_dir=/mnt/data/acorso/results --save_dir=results/self_trained_calibrated
python holistic_reliability_evaluation/evaluate.py --wilds_pretrained=true --dataset=fmow --calibration_method=temperature_scaling --model_dir=/scratch/users/acorso/wilds_models --save_dir=results/wilds_pretrained_calibrated
python holistic_reliability_evaluation/evaluate.py --dataset=fmow --calibration_method=temperature_scaling --model_dir=/mnt/data/acorso/results --save_dir=results/self_trained_calibrated
python holistic_reliability_evaluation/evaluate.py --wilds_pretrained=true --dataset=rxrx1 --calibration_method=temperature_scaling --model_dir=/scratch/users/acorso/wilds_models --save_dir=results/wilds_pretrained_calibrated
python holistic_reliability_evaluation/evaluate.py --dataset=rxrx1 --calibration_method=temperature_scaling --model_dir=/mnt/data/acorso/results --save_dir=results/self_trained_calibrated
