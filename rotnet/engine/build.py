# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午8:00
@file: build.py
@author: zj
@description: 
"""

import time
import copy
import torch

from rotnet.util.metrics import topk_accuracy


def train_model(model_name, model, criterion, optimizer, lr_scheduler, data_loaders, data_sizes,
                epoches=100, device=None):
    since = time.time()

    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    loss_dict = {'train': [], 'test': []}
    acc_dict = {'train': [], 'test': []}
    for epoch in range(epoches):
        print('{} - Epoch {}/{}'.format(model_name, epoch, epoches - 1))
        print('-' * 10)

        # Each epoch has a training and test phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_acc = 0.0

            # Iterate over data.
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # print(outputs.shape)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # compute top-k accuray
                    topk_list = topk_accuracy(outputs, labels, topk=(1,))
                    running_acc += topk_list[0]

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # print(f'loss: {running_loss}, acc: {running_acc}')
            if phase == 'train':
                lr_scheduler.step()

            epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_acc / len(data_loaders[phase])

            loss_dict[phase].append(epoch_loss)
            acc_dict[phase].append(epoch_acc)

            print('{} Loss: {:.4f} Top-1 Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

        # 每训练一轮就保存
        # util.save_model(model.cpu(), '../data/models/%s_%d.pth' % (model_name, epoch))
        # model = model.to(device)

    time_elapsed = time.time() - since
    print('Training {} complete in {:.0f}m {:.0f}s'.format(model_name, time_elapsed // 60, time_elapsed % 60))
    print('Best test Top-1 Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model, loss_dict, acc_dict
