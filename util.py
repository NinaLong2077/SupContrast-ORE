from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,), label_to_race_mapping = None):
    """Computes the accuracy over the k top predictions for the specified values of k, and records accuracy by race"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        target_viewed = target.view(1, -1)
        target_expand_pred = target_viewed.expand_as(pred)

        correct = pred.eq(target_expand_pred)
        if label_to_race_mapping: 
          # print(label_to_race_mapping) 
          # print(correct)

          # Initialize a dictionary to store correct predictions for each race
          correct_by_race = {race: 0 for race in label_to_race_mapping.values()}
          total_by_race = {race: 0 for race in label_to_race_mapping.values()}

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
            if label_to_race_mapping: 
              # Iterate over each label in the batch
              for i in range(batch_size):
                  # Get the race corresponding to this label
                  # print(f"target: {target}")
                  # print(f"target[i]: {target[i]}")
                  race = label_to_race_mapping[target[i].item()]
                  # Increment the total count for this race
                  total_by_race[race] += 1
                  # If this label was predicted correctly
                  if correct[0][i]:
                      # Increment the count for this race
                      correct_by_race[race] += 1
    if label_to_race_mapping: 

      # Calculate accuracy for each race
      accuracy_by_race = {race: correct / total if total > 0 else 0 
                          for race, correct, total in zip(correct_by_race.keys(), correct_by_race.values(), total_by_race.values())}

      return res, accuracy_by_race
    return res




def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state
