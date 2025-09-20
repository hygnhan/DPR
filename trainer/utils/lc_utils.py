import numpy as np
import torch

import random
from typing import Dict, List


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from  2"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def obtain_groups(merged_labels: List, num_classes: int):
    """ Get index for each label group and index for minority samples in each label group.

    Args:
        merged_labels (List): a list of pairs the first entry is the target label
         and the second entry is the label from the biased classifier.
        num_classes (int): number of classes.

    Returns:
        grouped_target_index (dict): a dictionary with key as the target label and value as the index of the samples.
        grouped_minority_index (dict): a dictionary with key as
            the target label and value as the index of the minority samples.
    """
    grouped_target_index = {}
    grouped_minority_index = {}
    for k in range(0, num_classes):
        grouped_minority_index[k] = []
        grouped_target_index[k] = []
    for i in range(0, len(merged_labels)):
        # Minority samples are with different target and biased prediction.
        if merged_labels[i][0] != merged_labels[i][1]:
            grouped_minority_index[merged_labels[i][0].item()].append(i)
        grouped_target_index[merged_labels[i][0].item()].append(i)
    return grouped_minority_index, grouped_target_index


def group_mixUp(feature: torch.Tensor, bias_label: torch.Tensor, correction: torch.Tensor, label: torch.Tensor,
                num_classes: int, tau: float):
    """Calculate Group mixUp for a batch of samples.

    Args:
        feature (Tensor): a matrix with feature from N samples.
        bias_label (Tensor): prediction from the biased classifier.
        correction (Tensor): the logic correction matrix.
        label (Tensor): the target label of the samples.
        num_classes (int): number of classes.
        tau (float): mixUp parameter in Algorithm 1 in the paper.

    Returns: A dict with following fields
        mixed_feature (Tensor): Mixed feature.
        mixed_correction (Tensor): correction term for the mixUp samples.
        label_majority (Tensor): target label for the majority sample.
        label_minority (Tensor): target label of the minority sample.
        lam (float): the mix ratio.
    """
    merged_target = [(label[i], bias_label[i]) for i in range(0, len(label))]
    target_groups_a, target_groups_b = obtain_groups(merged_target, num_classes)
    return mixUp(target_groups_a, target_groups_b, feature, label, correction, tau)


def mixUp(grouped_minority_index: Dict, grouped_target_index: Dict, feature: torch.Tensor, label: torch.Tensor,
          correction: torch.Tensor, tau: float = 0.5):
    """Calculate mixUp for a batch of samples.

    Args:
        grouped_minority_index (dict): a dictionary with key as the target label and value as the index of the samples.
        grouped_target_index (dict): a dictionary with key as the target label and value as the index of the samples.
        feature (Tensor): a matrix with feature from N samples.
        label (Tensor): the target label of the samples.
        correction (Tensor): the logic correction matrix.
        tau (float): the ratio of the number of samples in each group.

    Returns: A dict with following fields
        mixed_feature (Tensor): Mixed feature.
        mixed_correction (Tensor): correction term for the mixUp samples.
        label_majority (Tensor): target label for the majority sample.
        label_minority (Tensor): target label of the minority sample.
        lam (float): the mix ratio.
    """
    # Get mixed up parameter.
    lam = np.random.uniform(1 - 2 * tau, 1 - tau)
    indices_all_groups = []
    random_indices_all_groups = []
    for k in grouped_target_index.keys():
        indices = grouped_target_index[k]
        indices_all_groups += indices

        if grouped_minority_index[k]:
            # Get minority index.
            draw_indices = torch.randint(len(grouped_minority_index[k]), size=(len(indices),))
            random_indices_all_groups += [grouped_minority_index[k][l] for l in draw_indices]
        else:
            # if no minority index, do regular mixup.
            random_indices_all_groups += random.sample(indices, len(indices))

    indices_all_groups = torch.tensor(indices_all_groups)
    random_indices_all_groups = torch.tensor(random_indices_all_groups)

    # Define return values.
    mixed_feature = None
    mixed_correction = None
    label_majority, label_minority = None, None
    if random_indices_all_groups.nelement() > 0:
        # Mix feature.
        mixed_feature = lam * feature[indices_all_groups] + (1 - lam) * feature[random_indices_all_groups]
        # Mix correction value.
        mixed_correction = lam * correction[indices_all_groups] + (1 - lam) * correction[random_indices_all_groups]
        label_majority = label[indices_all_groups]
        label_minority = label[random_indices_all_groups]

    return {"mixed_feature": mixed_feature, "mixed_correction": mixed_correction, "label_majority": label_majority,
            "label_minority": label_minority, "lam": lam}


class EMA:
    def __init__(self, label, num_classes=None, alpha=0.9):
        self.label = label.cuda()
        self.alpha = alpha
        self.parameter = torch.zeros(label.size(0), num_classes)
        self.updated = torch.zeros(label.size(0), num_classes)
        self.num_classes = num_classes
        self.max = torch.zeros(self.num_classes).cuda()

    def update(self, data, index, curve=None, iter_range=None, step=None):
        self.parameter = self.parameter.to(data.device)
        self.updated = self.updated.to(data.device)
        index = index.to(data.device)

        if curve is None:
            self.parameter[index] = self.alpha * self.parameter[index] + (1 - self.alpha * self.updated[index]) * data
        else:
            alpha = curve ** -(step / iter_range)
            self.parameter[index] = alpha * self.parameter[index] + (1 - alpha * self.updated[index]) * data
        self.updated[index] = 1

    def max_loss(self, label):
        label_index = torch.where(self.label == label)[0]
        return self.parameter[label_index].max()

    def min_loss(self, label):
        label_index = torch.where(self.label == label)[0]
        return self.parameter[label_index].min()


class EMA_squre:
    def __init__(self, num_classes=None, alpha=0.9, avg_type = 'mv'):
        self.alpha = alpha
        self.parameter = torch.zeros(num_classes, num_classes)
        self.global_count_ = torch.zeros(num_classes, num_classes)
        self.updated = torch.zeros(num_classes, num_classes)
        self.num_classes = num_classes
        self.max = torch.zeros(self.num_classes).cuda()
        self.avg_type = avg_type

    def update(self, data, y_list, a_list, curve=None, iter_range=None, step=None, bias=None, fix = None):
        self.parameter = self.parameter.to(data.device)
        self.updated = self.updated.to(data.device)
        # self.global_count_ = self.global_count_.to(data.device)
        y_list = y_list.to(data.device)
        a_list = a_list.to(data.device)


        count = torch.zeros(self.num_classes, self.num_classes).to(data.device)
        # parameter_temp = torch.zeros(self.num_classes, self.num_classes, self.num_classes).to(data.device)

        if self.avg_type == 'mv':
            if curve is None:
                for i, (y, a) in enumerate(zip(y_list, a_list)):
                    # parameter_temp[y,a] += data[i]
                    count[y,a] += 1
                    self.global_count_[y,a] += 1
                    self.parameter[y,a] = self.alpha * self.parameter[y,a] + (1 - self.alpha * self.updated[y,a]) * data[i,y]#parameter_temp[y,a]/count[y,a]
                    self.updated[y,a] = 1
            else:
                alpha = curve ** -(step / iter_range)
                for i, (y, a) in enumerate(zip(y_list, a_list)):
                    # parameter_temp[y,a] += data[i]
                    count[y,a] += 1
                    self.global_count_[y,a] += 1
                    self.parameter[y,a] = alpha * self.parameter[y,a] + (1 - alpha * self.updated[y,a]) * data[i,y]#parameter_temp[y,a]/count[y,a]
                    self.updated[y,a] = 1
        elif self.avg_type == 'mv_batch':
            self.parameter_temp = torch.zeros(self.num_classes, self.num_classes).to(data.device)
            for i, (y, a) in enumerate(zip(y_list, a_list)):
                count[y,a] += 1
                self.global_count_[y,a] += 1
                self.parameter_temp[y,a] += data[i,y]
            self.parameter = self.alpha * self.parameter + (1 - self.alpha) * self.parameter_temp / (count + 1e-4)
        elif self.avg_type == 'batch':
            self.parameter_temp = torch.zeros(self.num_classes, self.num_classes).to(data.device)
            for i, (y, a) in enumerate(zip(y_list, a_list)):
                count[y,a] += 1
                self.global_count_[y,a] += 1
                self.parameter_temp[y,a] += data[i,y]
            self.parameter = self.parameter_temp / (count + 1e-4)
        elif self.avg_type == 'epoch':
            for i, (y, a) in enumerate(zip(y_list, a_list)):
                count[y,a] += 1
                self.global_count_[y,a] += 1
                self.parameter[y,a] += data[i,y]
        else:
            raise NotImplementedError("This averaging type is not yet implemented!")

        if fix is not None:
            self.parameter = torch.ones(self.num_classes, self.num_classes) * 0.1#* 0.005/(self.num_classes-1)
            # for i in range(self.num_classes):
            #     self.parameter[i,i] = 0.995
            self.parameter = self.parameter.to(data.device)





    # def max_loss(self, label):
    #     label_index = torch.where(self.label == label)[0]
    #     return self.parameter[label_index].max()

    # def min_loss(self, label):
    #     label_index = torch.where(self.label == label)[0]
    #     return self.parameter[label_index].min()
