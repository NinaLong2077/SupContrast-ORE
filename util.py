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


def initialize_race_dicts(label_to_race_mapping):
    """Initialize dictionaries to store correct predictions and total samples for each race."""
    correct_by_race = {}
    total_by_race = {}
    for race in label_to_race_mapping.values():
        correct_by_race[race] = 0
        total_by_race[race] = 0
    return correct_by_race, total_by_race

def print_tensor_as_grid(tensor, title):
    print(f"{title}:")
    print(np.array(tensor.cpu()))

def update_race_dicts(correct, target, label_to_race_mapping, correct_by_race, total_by_race):
    """
    Update dictionaries with data from the current batch.
    
    Args:
    - correct: A tensor indicating whether predictions were correct for the current batch.
    - target: A tensor containing true labels for the current batch.
    - label_to_race_mapping: A dictionary mapping labels to race categories.
    - correct_by_race: A dictionary to store the count of correct predictions for each race.
    - total_by_race: A dictionary to store the total count of samples for each race.
    """
    num_rows = target.size(0)
    num_cols = target.size(1)
    
    # Print the tensors as grids before the comparison
    # print_tensor_as_grid(correct, "Correct")
    # print_tensor_as_grid(target, "Target")
    
    for row_index in range(num_rows):
        for col_index in range(num_cols):
            true_label = target[row_index, col_index].item()
            race = label_to_race_mapping[true_label]
            
            # Increment the total count for this race
            total_by_race[race] += 1
            # print(f"After processing sample at ({row_index}, {col_index}), total count for {race} is now {total_by_race[race]}")
            
            is_correct_prediction = correct[row_index, col_index]
            
            # If this label was predicted correctly, increment the count for this race
            if is_correct_prediction:
                correct_by_race[race] += 1
                # print(f"Correct prediction for {race}, correct count is now {correct_by_race[race]}")

        # print("----")  # Separating lines for better visualization


def calculate_accuracy_by_race(correct_by_race, total_by_race):
    """
    Calculate accuracy for each race.
    
    Args:
    - correct_by_race: A dictionary storing the count of correct predictions for each race.
    - total_by_race: A dictionary storing the total count of samples for each race.
    
    Returns:
    - accuracy_by_race: A dictionary with accuracy values for each race.
    """
    accuracy_by_race = {}
    race_keys = correct_by_race.keys()
    correct_values = correct_by_race.values()
    total_values = total_by_race.values()
    
    zipped_data = list(zip(race_keys, correct_values, total_values))

    for race, correct, total in zipped_data:
        if total > 0:
            accuracy_by_race[race] = correct / total
        else:
            accuracy_by_race[race] = 0
    
    return accuracy_by_race


def accuracy(output, target, topk=(1,), label_to_race_mapping=None):
    """Computes the accuracy over the k top predictions for the specified values of k, and records accuracy by race"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        target_viewed = target.view(1, -1)
        target_expand_pred = target_viewed.expand_as(pred)

        correct = pred.eq(target_expand_pred)

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        if label_to_race_mapping: 
            correct_by_race, total_by_race = initialize_race_dicts(label_to_race_mapping)
            update_race_dicts(correct, target, label_to_race_mapping, correct_by_race, total_by_race)
            accuracy_by_race = calculate_accuracy_by_race(correct_by_race, total_by_race)
            return res, accuracy_by_race

        return res


"""
id_to_idx: {'m.0181j_': 0, 'm.01lb8z': 1, 'm.01vfm2v': 2, 'm.020skv': 3, 'm.0240pk': 4, 'm.025y4kr': 5, 'm.02ryn_3': 6, 'm.03bz141': 7, 'm.03bz8l9': 8, 'm.03c355h': 9, 'm.03gw8ss': 10, 'm.03tjn_': 11, 'm.0406m8s': 12, 'm.0415n32': 13, 'm.0437ps': 14, 'm.04n03h8': 15, 'm.04q43q': 16, 'm.04yy7s': 17, 'm.05km3r': 18, 'm.07h5rn': 19, 'm.09d6n9': 20, 'm.09fdg1': 21, 'm.09z1b2': 22, 'm.0bb8pbs': 23, 'm.0bh12n': 24, 'm.0cb1h4': 25, 'm.0cc99yf': 26, 'm.0cfxd5': 27, 'm.0chd66': 28, 'm.0czmj0': 29, 'm.0dd9hvn': 30, 'm.0ffgks': 31, 'm.0fxr_v': 32, 'm.0hn9wjt': 33, 'm.0jt66jg': 34, 'm.0jyvwv': 35, 'm.0ktdtr': 36, 'm.0n476ww': 37, 'm.0pcvrt5': 38, 'm.0zg9h1c': 39}
id_list: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
Class: ['m.0181j_', 'm.0181j_', 'm.0181j_', 'm.0181j_', 'm.0181j_', 'm.0181j_', 'm.0181j_', 'm.0181j_', 'm.0181j_', 'm.0181j_', 'm.0181j_', 'm.0181j_', 'm.01lb8z', 'm.01lb8z', 'm.01lb8z', 'm.01lb8z']
Races: ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']
Label-to-Race{0: '0', 1: '0'}
labels_to_races: {0: '0', 1: '0'}
accuracy_by_race: {'0': 0.25}

def accuracy(output, target, topk=(1,), label_to_race_mapping = None):
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


"""





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
