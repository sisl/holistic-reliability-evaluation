
#run evaluation file to get model and dataset
exec(open('holistic_reliability_evaluation/evaluation.py').read())


#ensuring model weights are on GPU, was unable to run without doing this
model = model.cuda()

#run autoattack
from autoattack import AutoAttack
adversary = AutoAttack(model, norm='Linf', eps=8/255, version='rand')
adversary.run_standard_evaluation(input, y, bs=1)