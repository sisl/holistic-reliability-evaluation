import torch
from training.hre_model import ClassificationTask

checkpoint = "/home/acorso/results/erm/tune/erm/1yzjhqae/checkpoints/epoch=0-step=591.ckpt"
model = ClassificationTask.load_from_checkpoint(checkpoint_path=checkpoint)

# Sample a batch of validation data
batch = next(iter(model.val_dataloader()))

## play with adversarial robustness
model.security(batch["id"])
b = batch["id"]


model.to(torch.device('cuda'))

from autoattack import AutoAttack
adversary = AutoAttack(model.model, device=model.device)
adversary.fab.n_target_classes = model.n_classes-1
adversary.apgd_targeted.n_target_classes = model.n_classes-1
if model.n_classes < 4:
    adversary.attacks_to_run = ['apgd-ce', 'fab-t', 'square']

xadv, yadv = adversary.run_standard_evaluation(b[0].to(torch.device('cuda')), b[1].to(torch.device('cuda')), return_labels=True)
adv_acc = sum(b[1] == yadv.cpu()).item() / b[1].size(0)
adv_acc

## Produce OOD plots
import pytorch_ood as ood
import matplotlib.pyplot as plt
metrics = ood.utils.OODMetrics()

id_batch = batch["id"]
ood_batch = batch["ood"][0]
target = torch.ones(id_batch[0].shape[0])
metrics.update(model.ood_detector(id_batch[0]), -1 * target)
metrics.update(model.ood_detector(ood_batch[0]), 1 * target)

ood_metrics = metrics.compute()
ood_metrics["AUROC"]

id_scores = -1*metrics.buffer["scores"][metrics.buffer['labels'] == 0]
od_scores = -1*metrics.buffer["scores"][metrics.buffer['labels'] == 1]

fig, ax = plt.subplots()
ax.hist(id_scores, alpha=0.3, linewidth=1, edgecolor='black', label="ID (Cameylon17)")
ax.hist(od_scores, alpha=0.3, linewidth=1, edgecolor='black', label="OD (RxRx1)")
ax.set_title('OOD Detection')
ax.legend(loc="upper left")

plt.savefig("OOD_scores.png")