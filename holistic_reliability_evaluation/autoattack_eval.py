exec(open('holistic_reliability_evaluation/evaluation.py').read())

print(input)
print(y)

input = input.cuda()
y = y.cuda()
model = model.cuda()

from autoattack import AutoAttack
adversary = AutoAttack(model, norm='Linf', eps=8/255, version='standard')
adversary.run_standard_evaluation(input, y)