import os
import pickle as pkl
import torch as t
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from sklearn.covariance import EmpiricalCovariance

from sklearn.metrics import roc_curve, auc


# target_dir = '../log/colored_mnist/grad_ext/bias_0.005_noise_0.0/debug0/'
target_dir = '../log/colored_mnist/grad_ext/bias_0.01_noise_0.0/check_score0/'
# target_dir = '../log/colored_mnist/grad_ext/bias_0.05_noise_0.0/debug0/'

pkl_dir = target_dir +'out.pkl'
target_dir = target_dir+'fig/'
os.makedirs(target_dir,exist_ok=True)

def mahalanobis(x):
    cov_fn = EmpiricalCovariance()
    cov = cov_fn.fit(x.cpu().numpy())
    temp_precision = cov_fn.precision_
    temp_precision = t.from_numpy(temp_precision).float()
    mahal = -0.5 * t.mm(t.mm(x, temp_precision), x.t())
    return mahal.diag()


def plot_roc_curve(f, t, label):
    plt.plot(f, t, label=label)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    
with open(pkl_dir, 'rb') as fr:
    data = pkl.load(fr)

grad = data['grad']
feat = data['feat']
label = data['label']
gtlabel = data['gtlabel']
blabel = data['blabel']
loss = data['loss']

pos_major = t.where(blabel == label)
pos_minor = t.where(blabel != label)

targets = t.ones(len(label))
targets[pos_major] = 0
targets[pos_minor] = 1



auroc_grad = {}

grad_tot = []
feat_tot = []



grad_norm_1 = t.norm(grad, dim=1, p=1)
grad_norm_2 = t.norm(grad, dim=1, p=2)
grad_norm_inf = t.norm(grad, dim=1, p=float("inf"))
feat_norm = t.zeros(len(grad_norm_1))
maha_norm = t.zeros(len(grad_norm_1))
for cidx in t.unique(label):
    pos = t.where(cidx == label)
    feat_mean = t.mean(feat[pos], dim=0)
    center_feat = feat[pos] - feat_mean
    feat_norm[pos] = t.norm(center_feat,dim=1)
    maha_norm[pos] = mahalanobis(center_feat)






inv_score_grad_1 = 1./grad_norm_1 / t.sum(1./grad_norm_1)
inv_score_grad_2 = 1./grad_norm_2 / t.sum(1./grad_norm_2)
inv_score_grad_inf = 1./grad_norm_inf / t.sum(1./grad_norm_inf)
inv_score_feat = 1./feat_norm / t.sum(1./feat_norm)
inv_score_maha = 1./maha_norm / t.sum(1./maha_norm)
inv_score_loss = 1./loss / t.sum(1./loss)


score_grad_1 = 1./inv_score_grad_1 / t.sum(1./inv_score_grad_1)
score_grad_2 = 1./inv_score_grad_2 / t.sum(1./inv_score_grad_2)
score_grad_inf = 1./inv_score_grad_inf / t.sum(1./inv_score_grad_inf)
score_feat = 1./inv_score_feat / t.sum(1./inv_score_feat)
score_maha = 1./inv_score_maha / t.sum(1./inv_score_maha)
score_loss = 1./inv_score_loss / t.sum(1./inv_score_loss)


print('L-1 norm,',t.sum(score_grad_1[pos_major]),t.sum(score_grad_1[pos_minor])  )
print('L-2 norm,',t.sum(score_grad_2[pos_major]),t.sum(score_grad_2[pos_minor])  )
print('L-inf norm,',t.sum(score_grad_inf[pos_major]),t.sum(score_grad_inf[pos_minor])  )
print('Feat,',t.sum(score_feat[pos_major]),t.sum(score_feat[pos_minor])  )
print('Maha,',t.sum(score_maha[pos_major]),t.sum(score_maha[pos_minor])  )
print('Loss,',t.sum(score_loss[pos_major]),t.sum(score_loss[pos_minor])  )




sampling_prob_grad1 = t.zeros((10,10))
sampling_prob_grad2 = t.zeros((10,10))
sampling_prob_gradinf = t.zeros((10,10))
sampling_prob_feat = t.zeros((10,10))
sampling_prob_maha = t.zeros((10,10))
sampling_prob_loss = t.zeros((10,10))

for color in range(10):
    for digit in range(10):
        pos = t.where( (blabel == color) & (label == digit))
        sampling_prob_grad1[color,digit] = t.sum(score_grad_1[pos])
        sampling_prob_grad2[color,digit] = t.sum(score_grad_2[pos])
        sampling_prob_gradinf[color,digit] = t.sum(score_grad_inf[pos])
        sampling_prob_feat[color,digit] = t.sum(score_feat[pos])
        sampling_prob_maha[color,digit] = t.sum(score_maha[pos])
        sampling_prob_loss[color,digit] = t.sum(score_loss[pos])

sns.heatmap(sampling_prob_grad1, annot=True, fmt='.3f', cmap='YlGnBu')
plt.savefig(target_dir+'grad1.png')
plt.close()

sns.heatmap(sampling_prob_grad2, annot=True, fmt='.3f', cmap='YlGnBu')
plt.savefig(target_dir+'grad2.png')
plt.close()

sns.heatmap(sampling_prob_gradinf, annot=True, fmt='.3f', cmap='YlGnBu')
plt.savefig(target_dir+'gradinf.png')
plt.close()

sns.heatmap(sampling_prob_feat, annot=True, fmt='.3f', cmap='YlGnBu')
plt.savefig(target_dir+'feat.png')
plt.close()

sns.heatmap(sampling_prob_maha, annot=True, fmt='.3f', cmap='YlGnBu')
plt.savefig(target_dir+'maha.png')
plt.close()

sns.heatmap(sampling_prob_loss, annot=True, fmt='.3f', cmap='YlGnBu')
plt.savefig(target_dir+'loss.png')
plt.close()






















grad_norm_1 /= t.max(grad_norm_1)
grad_norm_2 /= t.max(grad_norm_2)
grad_norm_inf /= t.max(grad_norm_inf)
loss /= t.max(loss)
feat_norm /= t.max(feat_norm)
maha_norm /= t.max(maha_norm)

gradf1, gradt1, thresholds = roc_curve(targets, grad_norm_1)
gradf2, gradt2, thresholds = roc_curve(targets, grad_norm_2)
gradfinf, gradtinf, thresholds = roc_curve(targets, grad_norm_inf)
featf, featt, thresholds = roc_curve(targets, feat_norm)
mahaf, mahat, thresholds = roc_curve(targets, maha_norm)
lossf, losst, thresholds = roc_curve(targets, loss)

auroc_grad1 = auc(gradf1, gradt1)
auroc_grad2 = auc(gradf2, gradt2)
auroc_gradinf = auc(gradfinf, gradtinf)
auroc_feat = auc(featf, featt)
auroc_maha = auc(mahaf, mahat)
auroc_loss = auc(lossf, losst)


plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plot_roc_curve(gradf1, gradt1, 'grad_norm-1')
plot_roc_curve(gradf2, gradt2, 'grad_norm-2')
plot_roc_curve(gradfinf, gradtinf, 'grad_norm-inf')
plot_roc_curve(featf, featt, 'feat_norm')
plot_roc_curve(lossf, losst, 'Loss')
plot_roc_curve(mahaf, mahat, 'Mahalnobis')
plt.savefig(target_dir+'total.png')
plt.close()



np.savetxt(target_dir+'feat_major.txt', feat_norm[pos_major])
np.savetxt(target_dir+'feat_minor.txt', feat_norm[pos_minor])
np.savetxt(target_dir+'maha_major.txt', maha_norm[pos_major])
np.savetxt(target_dir+'maha_minor.txt', maha_norm[pos_minor])
np.savetxt(target_dir+'loss_major.txt', loss[pos_major])
np.savetxt(target_dir+'loss_minor.txt', loss[pos_minor])

print('AUROC [grad_1]: ',auroc_grad1)
print('AUROC [grad_2]: ',auroc_grad2)
print('AUROC [grad_inf]: ',auroc_gradinf)
print('AUROC [feat]: ',auroc_feat)
print('AUROC [maha]: ',auroc_maha)
print('AUROC [loss]: ',auroc_loss)

