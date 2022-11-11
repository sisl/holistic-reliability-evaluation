exec(open('holistic_reliability_evaluation/evaluation.py').read())

from autoattack import AutoAttack
adversary = AutoAttack(model, norm='Linf', eps=8/255, version='standard')
adversary.run_standard_evaluation(input, y)