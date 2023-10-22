from __future__ import print_function

import sys
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn
import pandas as pd
from torchvision import transforms, datasets

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer
from networks.resnet_big import SupConResNet, LinearClassifier

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass
'''
python3 main_linear.py --batch_size 16 --num_workers 8 --learning_rate 0.5 --epochs 2 --cosine --dataset path --train_folder ./data/train --val_folder ./data/test/African --cur_race African --n_cls 500
'''

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='path',
                        choices=['path', 'cifar10', 'cifar100'], help='dataset')
    parser.add_argument('--n_cls', type=int, help='number of classes')
    parser.add_argument('--train_folder', type=str, help='path to train folder')
    parser.add_argument('--val_folder', type=str, help='path to val folder')
    parser.add_argument('--cur_race', type=str, choices=['Caucasian', 'African', 'Asian', 'Indian'], help="enter race currently testing")

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    opt = parser.parse_args()

    # set the path according to the environment
    # opt.data_folder = './datasets/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if opt.dataset == 'path':
        assert opt.train_folder is not None \
            and opt.val_folder is not None \
            and opt.n_cls is not None \
            and opt.cur_race is not None

    return opt

def set_loader(opt):
    #get train and val loader
    input_shape = [3, 128, 128]
    train_transform = transforms.Compose([
        transforms.Resize(int(input_shape[1] * 156 / 128)),
        transforms.RandomCrop(input_shape[1:]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
    ])  

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
    ])

    if opt.train_folder is not None:
        train_dataset = datasets.ImageFolder(
            root=opt.train_folder,
            transform=train_transform
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size, shuffle=True,
            num_workers=opt.num_workers, pin_memory=True)
    else:
        train_loader = None

    val_dataset = datasets.ImageFolder(
        root=opt.val_folder,
        transform=val_transform
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)

    return train_loader, val_loader



import os

def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    # Check if the checkpoint file exists
    if os.path.isfile(opt.ckpt):
        ckpt = torch.load(opt.ckpt, map_location='cpu')
        state_dict = ckpt['model']
    else:
        print(f"No checkpoint found at {opt.ckpt}")
        state_dict = None

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        elif state_dict is not None:  # Add this line
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        # Load the state dict only if it is not None
        if state_dict is not None:
            model.load_state_dict(state_dict)

    else:
        raise NotImplementedError('This code requires GPU')

    return model, classifier, criterion



def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def label_to_race(labels, id_to_idx, class_to_race):
    id_list = labels.tolist() 
    # print(f"id_list: {id_list}")

    # this is a list of id is the image from, 0 (e.g., 'm.0181j_'), and the remaining 4 images belong to class 1 (e.g., 'm.01lb8z').
    # id_list: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    
    class_labels = [class_label for idx in id_list for class_label, class_idx in id_to_idx.items() if class_idx == idx] # Get the list of corresponding ids that the images are from
    # id_list: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    # -> Class: ['m.0181j_', 'm.0181j_', 'm.0181j_', 'm.0181j_', 'm.0181j_', 'm.0181j_', 'm.0181j_', 'm.0181j_', 'm.0181j_', 'm.0181j_', 'm.0181j_', 'm.0181j_', 'm.01lb8z', 'm.01lb8z', 'm.01lb8z', 'm.01lb8z']
    races = [class_to_race[class_label] for class_label in class_labels] # Now map the ids to the race from the csv
    # print(f"Class: {class_labels}")
    # print(f"Races: {races}")
    # Create a dictionary mapping labels to their corresponding races
    label_to_race_mapping = dict(zip(id_list, races))
    # print(f"Label-to-Race{label_to_race_mapping}")

    return label_to_race_mapping

import csv
def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()
    id_to_idx = val_loader.dataset.class_to_idx # map of id to indices in the val set
    # print(f"id_to_idx: {id_to_idx}")

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # Read the CSV file and create a mapping from class labels to races
    class_to_race = {} 
    with open('./data/linear_prob_dataset_split.csv', 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            class_to_race[row['id']] = row['race']

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            labels_to_races = label_to_race(labels, id_to_idx, class_to_race)

            bsz = labels.shape[0]
            # print(f'Batch {idx} - Images: {images.shape}, Labels: {labels.shape}, Batch Size: {bsz}')

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)
            # print(f'labels: {labels}')

            # print(f'Output: {output}')

            # update metric
            losses.update(loss.item(), bsz)
            # acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            print(f"labels_to_races: {labels_to_races}")
            acc1, accuracy_by_race = accuracy(output, labels, topk=(1,), label_to_race_mapping = labels_to_races)
                # accuracy_by_race = {race: correct / total if total > 0 else 0 
                #         for race, correct, total in zip(correct_by_race.keys(), correct_by_race.values(), total_by_race.values())}

            top1.update(acc1[0][0], bsz)
            # print(acc1[0][0])
            print(f"accuracy_by_race: {accuracy_by_race}")

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      idx, len(val_loader), batch_time=batch_time,
                      loss=losses, top1=top1))
                for race, acc in accuracy_by_race.items():
                    print(f'Accuracy for {race}: {acc:.3f}')


    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    
    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy: {:.2f}'.format(
            epoch, time2 - time1, acc))

        # eval for one epoch
        loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
        if val_acc > best_acc:
            best_acc = val_acc

        # Save checkpoint after each epoch
        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, 'checkpoint_epoch_{}.pth'.format(epoch))

    print('best accuracy: {:.2f}'.format(best_acc))

if __name__ == '__main__':
    main()
