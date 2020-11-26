# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import numpy as np
import random

import torch
# import torch.distributed.deprecated as dist
from cjltest.divide_data import partition_dataset, select_dataset
from cjltest.models import MnistCNN, AlexNetForCIFAR, LeNetForMNIST
from cjltest.utils_data import get_data_transform
from cjltest.utils_model import MySGD
from torch.autograd import Variable
from torch.multiprocessing import Process as TorchProcess
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import ResNetOnCifar10
import vgg

parser = argparse.ArgumentParser()
# 集群信息
parser.add_argument('--ps-ip', type=str, default='127.0.0.1')
parser.add_argument('--ps-port', type=str, default='29500')
parser.add_argument('--this-rank', type=int, default=1)
parser.add_argument('--workers', type=int, default=2)

# 模型与数据集
parser.add_argument('--data-dir', type=str, default='~/dataset')
parser.add_argument('--data-name', type=str, default='cifar10')
parser.add_argument('--model', type=str, default='LROnMnist')
parser.add_argument('--save-path', type=str, default='./')

# 参数信息
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--train-bsz', type=int, default=400)
parser.add_argument('--stale-threshold', type=int, default=0)
parser.add_argument('--ratio', type=float, default=5)
parser.add_argument('--isCompensate', type=bool, default=False)
parser.add_argument('--loops', type=int, default=10)

parser.add_argument('--byzantine', type=int, default=1)
parser.add_argument('--V', type=float, default=100)
parser.add_argument('--T', type=int, default=1)
parser.add_argument('--title', type=str, default='Threshold')
parser.add_argument('--method', type=str, default='TrimmedMean')

parser.add_argument('--alpha', type=float, default=0)
parser.add_argument('--beta', type=float, default=0)

args = parser.parse_args()

# default gaussian noise
def addNoise(model):
    for p_idx, param in enumerate(model.parameters()):
        param.data += args.V * torch.randn_like(param.data)
    return model

args = parser.parse_args()


# select top-k gradient changes
'''
def get_upload_topk(g_remain, g_new, ratio, isCompensate, dev):
    for idx, g_layer in enumerate(g_new):
        g_remain[idx] += g_layer

    g_remain_abs_vector = torch.empty(0).cuda(dev)
    g_remain_abs = []
    for idx, g_layer in enumerate(g_remain):
        g_remain_layer_abs = torch.abs(g_remain[idx])
        g_remain_abs.append(g_remain_layer_abs)
        g_remain_layer_abs_reshape = g_remain_layer_abs.reshape(torch.numel(g_remain_layer_abs))
        g_remain_abs_vector = torch.cat((g_remain_abs_vector, g_remain_layer_abs_reshape),dim=0)  # merge two vectors into one vector

    param_num = torch.numel(g_remain_abs_vector)
    k = int(param_num * ratio)
    k = k if k > 0 else 1
    top_k = torch.topk(g_remain_abs_vector, k)
    threshold = top_k[0][k - 1].item()
    # print(threshold)
    g_upload = []
    for idx, g_layer in enumerate(g_remain_abs):
        mask = g_layer >= threshold
        g_upload_layer = torch.zeros_like(g_layer).cuda(dev)
        g_upload_layer[mask] += g_remain[idx][mask]
        g_remain[idx][mask] = 0.
        g_upload.append(g_upload_layer)

    return g_remain, g_upload
'''

# select top-k gradient changes
def get_upload(g_remain, g_new, ratio, isCompensate, g_threshold, dev):
    g_change = []
    g_change_merge = torch.empty(0).cuda(dev)
    for idx, g_layer in enumerate(g_new):
        g_change_layer = g_layer - g_remain[idx]
        g_change_layer = torch.abs(g_change_layer)
        g_change.append(g_change_layer)

        g_change_layer_reshape = g_change_layer.reshape(torch.numel(g_change_layer))
        g_change_merge = torch.cat((g_change_merge, g_change_layer_reshape),dim=0)  # merge two vectors into one vector

    # threshold
    param_num = torch.numel(g_change_merge)
    threshold = ratio * g_threshold / np.sqrt(param_num)
    # print(threshold)
    g_change_new = []
    non_upload_num = 0
    for idx, g_layer in enumerate(g_change):
        mask = g_layer < threshold
        non_upload_num += torch.sum(mask)

        g_change_tmp = g_new[idx] - g_remain[idx]
        g_change_tmp[mask] = 0.0
        g_change_new.append(g_change_tmp)

        g_remain[idx] += g_change_tmp
    return g_remain, g_change_new, (param_num-int(non_upload_num))/param_num

def byzantine_func(g_change, dev):
    g_mem = []
    g_layer_vector = torch.empty(0).cuda(dev)
    for idx, g_layer in enumerate(g_change):
        g_mem.append(g_layer)
        g_layer_reshape = g_layer.reshape(torch.numel(g_layer))
        g_layer_vector = torch.cat((g_layer_vector, g_layer_reshape),dim=0)  # merge two vectors into one vector

    tot_num = len(torch.nonzero(g_layer_vector))
    tot_val = torch.sum(g_layer_vector)
    tot_val /= tot_num

    g_new = []
    for idx, g_layer in enumerate(g_mem):
        mask = g_layer >= 0
        g_new_layer = torch.zeros_like(g_layer).cuda(dev)
        g_new_layer[mask] = tot_val
        g_new.append(g_new_layer)
    return g_new

# input: gradient list
# output: element-wise median of all gradients
def median_defense(g_list, workers, dev):
    median_g = []
    for p_idx, g_layer in enumerate(g_list[0]):
        g_layer_list = []
        for w in workers:
            g_layer_list.append(g_list[w - 1][p_idx])
        data_dim = g_layer_list[0].dim()
        # 取中位数
        tensor = torch.zeros_like(g_layer.data).cuda(dev) + torch.median(torch.stack(g_layer_list, data_dim), data_dim)[0]
        median_g.append(tensor)
    return median_g

# output: element-wise trimmed mean of all gradients
def trimmed_mean(g_list, workers, trimK, dev):
    workers_num = len(workers)
    g_trimmed_mean = []
    for p_idx, g_layer in enumerate(g_list[0]):
        g_trimmed_mean_layer = torch.zeros_like(g_layer.data).cuda(dev)
        g_layer_list = []
        for w in workers:
            g_layer_list.append(g_list[w - 1][p_idx])
        data_dim = g_layer_list[0].dim()
        tensor_max = torch.min(torch.topk(torch.stack(g_layer_list, data_dim), trimK)[0], -1)[0]
        tensor_min = -torch.min(torch.topk(-torch.stack(g_layer_list, data_dim), trimK)[0], -1)[0]

        for w in workers:
            max_mask = g_list[w - 1][p_idx].data >= tensor_max
            min_mask = g_list[w - 1][p_idx].data <= tensor_min

            tmp_layer = g_list[w - 1][p_idx].data + torch.zeros_like(g_list[w - 1][p_idx].data).cuda(dev)
            tmp_layer[max_mask] = 0
            tmp_layer[min_mask] = 0

            g_list[w - 1][p_idx] = tmp_layer

            g_trimmed_mean_layer.data += g_list[w-1][p_idx].data / (workers_num - 2 * trimK)
        g_trimmed_mean.append(g_trimmed_mean_layer)
    return g_trimmed_mean

# output: the mean of the applicable gradients
def FABA(g_list, workers, byzantine, dev):
    workers_num = len(workers)
    for k in range(byzantine):
        g_zero = []
        for p_idx, g_layer in enumerate(g_list[0]):
            global_update_layer = torch.zeros_like(g_layer.data).cuda(dev)
            for w in workers:
                global_update_layer += g_list[w-1][p_idx]
            g_zero.append(global_update_layer / (workers_num - k))
        max_differ, max_idx = 0, 0
        for w in workers:
            # g_differ = [0] * len(g_zero)
            total_differ = 0
            for p_idx, _ in enumerate(g_zero):
                # g_differ[p_idx] = g_list[w-1][p_idx] - g_zero[p_idx]
                total_differ += torch.norm(g_list[w-1][p_idx] - g_zero[p_idx]).pow(2)
            # total_differ = torch.norm(g_differ)
            # print(max_differ)
            # print(total_differ)
            if max_differ < total_differ.data:
                max_differ, max_idx = total_differ.data, w
        for p_idx, g_layer in enumerate(g_list[max_idx - 1]):
            g_list[max_idx - 1][p_idx] = torch.zeros_like(g_layer.data).cuda(dev)
    g_zero = []
    print(max_idx)
    for p_idx, g_layer in enumerate(g_list[0]):
        global_update_layer = torch.zeros_like(g_layer.data).cuda(dev)
        for w in workers:
            global_update_layer += g_list[w-1][p_idx]
        g_zero.append(global_update_layer / (workers_num - byzantine))
    return g_zero

def Krum(g_list, workers, byzantine, dev):
    # dist = [[] for _ in workers]
    # for i in workers:
    #     for j in workers:
    #         if (i == j):
    #             dist[i - 1].append(0)
    #         elif (i > j):
    #             dist[i - 1].append(dist[j - 1][i - 1])
    #         else:
    #             total_differ = 0
    #             for p_idx, _ in enumerate(g_list[0]):
    #                 total_differ += torch.norm(g_list[i - 1][p_idx] - g_list[j - 1][p_idx]).pow(2)
    #             dist[i - 1].append(total_differ)

    # min_dist, min_idx = -1, 0

    # for i in workers:
    #     tot_dist = 0
    #     dist[i - 1].sort()
    #     tot_dist = sum(dist[i - 1][: -(2 + byzantine)])
    #     if min_dist == -1 or tot_dist < min_dist:
    #         min_dist, min_idx = tot_dist, i
    min_idx = random.randint(1, 1)
    print(min_idx)

    krum_g = []
    for g_layer in g_list[min_idx - 1]:
        krum_g_layer = g_layer + torch.zeros_like(g_layer).cuda(dev)
        krum_g.append(krum_g_layer)

    return krum_g


def mean(g_list, workers, dev):
    g_mean = []
    worker_num = len(workers)
    for p_idx, g_layer in enumerate(g_list[0]):
        global_update_layer = torch.zeros_like(g_layer.data).cuda(dev)
        for w in workers:
            global_update_layer += g_list[w-1][p_idx]
        g_mean.append(global_update_layer / worker_num)
    return g_mean

def test_model(rank, model, test_data, dev):
    correct = 0
    total = 0
    # model.eval()
    with torch.no_grad():
        for data, target in test_data:
            data, target = Variable(data).cuda(dev), Variable(target).cuda(dev)
            output = model(data)
            # get the index of the max log-probability
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            # pred = output.data.max(1)[1]
            # correct += pred.eq(target.data).sum().item()

    acc = format(correct / total, '.4%')
    # print('Rank {}: Test set: Accuracy: {}/{} ({})'
    #       .format(rank, correct, len(test_data.dataset), acc))
    return acc


# noinspection PyTypeChecker
def run(workers, models, save_path, train_data_list, test_data, iterations_epoch):
    dev = torch.device('cuda')
    cpu = torch.device('cpu')

    start_time = time.time()
    models[0] = models[0].cuda(dev)
    for i in workers:
        models[i] = models[i].cuda(dev)

    workers_num = len(workers)

    print('Model recved successfully!')
    optimizers_list = []
    for i in workers:
        optimizer = MySGD(models[i].parameters(), lr=args.lr)
        # if args.model in ['MnistCNN', 'AlexNet', 'ResNet18OnCifar10']:
        #     optimizer = MySGD(models[i].parameters(), lr=0.1)
        # elif args.model in ['VGG11']:
        #     optimizer = MySGD(models[i].parameters(), lr=0.1)
        # else:
        #     optimizer = MySGD(models[i].parameters(), lr=0.1)
        optimizers_list.append(optimizer)

    if args.model in ['MnistCNN', 'AlexNet']:
        criterion = torch.nn.NLLLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.model in ['AlexNet', 'ResNet18OnCifar10']:
        decay_period = 10000
    else:
        decay_period = 1000000

    print('Begin!')

    # the several workers in the front of the rank list
    byzantine_workers_list = [w + 1 for w in range(args.byzantine)]

    # cache g_old_num old gradients
    g_old_num = args.loops
    g_old_list = []
    for i in workers:
        worker_g_old_list = [[torch.zeros_like(param.data).cuda(dev) for param in model.parameters()] for _ in range(g_old_num)]
        g_old_list.append(worker_g_old_list)
    g_old_count = 0

    global_g = [torch.zeros_like(param.data).cuda(dev) for param in model.parameters()]

    # store (train loss, energy, iterations)
    # naming rules: title + model_name + number_of_workers
    trainloss_file = './result' \
        + args.title \
        + '_' + args.method \
        + '_' + args.model \
        + '_bsz' + str(args.train_bsz) \
        + '_lr' + str(args.lr) \
        + '_R' + str(int(args.ratio)) \
        + '_al' + str(int(args.alpha * 1000)) \
        + '_be' + str(int(args.beta * 1000)) \
        + '_W' + str(args.workers) + '.txt'
    
    if(os.path.isfile(trainloss_file)):
        os.remove(trainloss_file)
    f_trainloss = open(trainloss_file, 'a')

    train_data_iter_list = []
    for i in workers:
        train_data_iter_list.append(iter(train_data_list[i-1]))

    epoch_train_loss = 0.0
    global_clock = 0
    g_remain_list = []
    h_remain_list = [] 
    h_last_list = [] 
    ratio = args.ratio
    threshold = 0.

    g_change_list = []
    for i in workers:
        g_change_list.append([torch.zeros_like(param.data).cuda(dev) for param in models[0].parameters()])
        h_remain_list.append([torch.zeros_like(param.data).cuda(dev) for param in models[0].parameters()])
        h_last_list.append([torch.zeros_like(param.data).cuda(dev) for param in models[0].parameters()])
    
    for epoch in range(args.epochs):
        iteration_loss = 0.0

        # g_change_average = [torch.zeros_like(param.data).cuda(dev) for param in models[0].parameters()]
        global_clock += 1
        g_change_average_list = [[] for _ in range(workers_num)]
        for i in workers:
            try:
                data, target = next(train_data_iter_list[i-1])
            except StopIteration:
                train_data_iter_list[i-1] = iter(train_data_list[i - 1])
                data, target = next(train_data_iter_list[i-1])
            data, target = Variable(data).cuda(dev), Variable(target).cuda(dev)
            optimizers_list[i-1].zero_grad()
            output = models[i](data)
            loss = criterion(output, target)
            loss.backward()
            delta_ws = optimizers_list[i-1].get_delta_w()
            iteration_loss += loss.data.item()/workers_num

            # update old gradient list
            g_new = []
            for layer_g in delta_ws:
                layer_g_tmp = torch.zeros_like(layer_g).cuda(dev)
                layer_g_tmp += layer_g
                g_new.append(layer_g_tmp)
            g_old_list[i-1].append(g_new)   # cache new gradient
            g_old_list[i-1].pop(0)
            # count the number of gradient
            g_old_count = min(g_old_count+1, g_old_num)
            # g_old_count += 1
            # if g_old_count > g_old_num:
            #     g_old_count = g_old_num

            if global_clock == 1:
                g_remain = [torch.zeros_like(g_layer).cuda(dev)+g_layer for g_layer in delta_ws]
                g_remain_list.append(g_remain)
                # synchronous update
                # the gradient change in the first iteration is gradient itself
                for g_change_layer_idx, g_change_layer in enumerate(g_change_list[i - 1]):
                    g_change_layer.data += delta_ws[g_change_layer_idx].data

                    g_change_average_list[i - 1].append(g_change_layer.data)

                sparsification_ratio = 1.0
            else:
                update_new = []
                for layer_idx, layer_g in enumerate(delta_ws):
                    layer_update_new_tmp = torch.zeros_like(layer_g).cuda(dev)
                    for g_old in g_old_list[i-1]:
                        layer_update_new_tmp += g_old[layer_idx]
                    layer_update_new_tmp /= g_old_count
                    update_new.append(layer_update_new_tmp)
                # print(g_old_count)
                # g_remain, g_large_change, sparsification_ratio= get_upload(g_remain_list[i-1],update_new,ratio,args.isCompensate, threshold, dev)


                new_g_avg = [torch.zeros_like(g_layer) + g_layer for g_layer in update_new]
                for idx, g_layer in enumerate(update_new):
                    new_g_avg[idx] += args.alpha * (h_last_list[i - 1][idx] - h_remain_list[i - 1][idx])

                g_remain, g_large_change, sparsification_ratio = get_upload(g_remain_list[i - 1], new_g_avg, args.ratio, args.isCompensate, threshold, dev)
                g_remain_list[i - 1] = g_remain

                # if i in byzantine_workers_list:
                #     g_large_change = byzantine_func(g_large_change, dev)
                

                h_remain_list[i - 1] = h_last_list[i - 1]
                h_last_list[i - 1] = [torch.zeros_like(g_layer) for g_layer in update_new]
                for idx, g_layer in enumerate(update_new):
                    h_last_list[i - 1][idx] = h_remain_list[i - 1][idx] * args.beta
                    h_last_list[i - 1][idx] -= (update_new[idx] - g_remain[idx])


                for g_change_layer_idx, g_change_layer in enumerate(g_change_list[i - 1]):
                    g_change_layer.data += g_large_change[g_change_layer_idx].data
                    g_change_average_list[i - 1].append(g_change_layer)
                
                # if i in byzantine_workers_list:
                #     g_remain_list[i - 1] = g_change_list[i - 1]
                
                # for g_change_layer_idx, g_change_layer in enumerate(g_change_list[i - 1]):
                #     g_remain_list[i - 1][g_change_layer_idx] = g_change_layer + torch.zeros_like(g_change_layer).cuda(dev)

                # if i in byzantine_workers_list:
                #     g_change_layer.data += g_large_change[g_change_layer_idx].data
                #     g_change_layer.data += args.V * torch.randn_like(g_change_layer.data).data
                # else:
                #     g_change_layer.data += g_large_change[g_change_layer_idx].data
                
                # g_change_average_list[i - 1].append(g_change_layer.data)
                
                # if i in byzantine_workers_list:
                #     by_list = byzantine_func(g_change_list[i - 1], dev)
                #     for g_change_layer in by_list:
                #         g_change_average_list[i - 1].append(g_change_layer)
                # else:
                #     for g_change_layer in g_change_list[i - 1]:
                #         g_change_average_list[i - 1].append(g_change_layer)
        
        # non_byz_g = []
        # for p_idx, param in enumerate(models[0].parameters()):
        #     global_update_layer = torch.zeros_like(param.data).cuda(dev)
        #     for w in workers:
        #         if w not in byzantine_workers_list:
        #             global_update_layer += g_change_average_list[w - 1][p_idx]
        #     tensor = global_update_layer / (workers_num - args.byzantine)
        #     non_byz_g.append(tensor)
        
        # non_byz_g = byzantine_func(non_byz_g, dev)

        # for i in workers:
        #     if i in byzantine_workers_list:
        #         g_change_average_list[i - 1] = []
        #         for g_change_layer in non_byz_g:
        #             g_change_average_list[i - 1].append(g_change_layer + torch.zeros_like(g_change_layer).cuda(dev))
        
        # 同步操作
        if args.method == "Mean":
            g_median = mean(g_change_average_list, workers, dev)
        elif args.method == "TrimmedMean":
            # if args.T > 0 and args.T < workers_num/2:
            #     beta = args.T
            # else:
            #     beta = int((workers_num-1)/2)
            g_median = trimmed_mean(g_change_average_list, workers, args.byzantine, dev)
        elif args.method == "Median":
            g_median = median_defense(g_change_average_list, workers, dev)
        elif args.method == "FABA":
            g_median = FABA(g_change_average_list, workers, args.byzantine, dev)
        elif args.method == "Krum":
            g_median = Krum(g_change_average_list, workers, args.byzantine, dev)
        
        g_quare_sum = 0.0   # for threshold
        for p_idx, param in enumerate(models[0].parameters()):
            param.data -= g_median[p_idx].data
            # print(g_median[p_idx].data)
            for w in workers:
                list(models[w].parameters())[p_idx].data = param.data + torch.zeros_like(param.data).cuda(dev)

            g_quare_sum += torch.sum(g_median[p_idx].data * g_median[p_idx].data)

        g_quare_sum = torch.sqrt(g_quare_sum).cuda(dev)
        threshold = g_quare_sum.data.item()

        # epoch_train_loss += iteration_loss
        # epoch = int(iteration / iterations_epoch)
        current_time = time.time() - start_time
        test_acc = 0
        if epoch % 50 >= 45:
            test_acc = test_model(0, models[1], test_data, dev)
        print('Epoch {}, Time:{}, Loss:{}'.format(epoch, current_time, iteration_loss))
        f_trainloss.write(str(epoch) +
                            '\t' + str(current_time) +
                            '\t' + str(iteration_loss) + 
                            '\t' + str(sparsification_ratio) + 
                            # '\t' + str(test_loss) + 
                            '\t' + str(test_acc) +
                            '\n')
        f_trainloss.flush()
        # if (iteration+1) % iterations_epoch == 0:
            # 训练结束后进行test
            # test_loss, test_acc = test_model(0, model, test_data, criterion=criterion)
            # f_trainloss.write(str(args.this_rank) +
            #                   "\t" + str(epoch_train_loss / float(iterations_epoch)) +
            #                   "\t" + str(iteration_loss) +
            #                   "\t" + str(0) +
            #                   "\t" + str(epoch) +
            #                   "\t" + str(0) +
            #                   "\t" + str(iteration) +
            #                   "\t" + str(sparsification_ratio) +        # time
            #                   "\t" + str(global_clock) +        # time
            #                   '\n')
        
            # epoch_train_loss = 0.0
        # 在指定epochs (iterations) 减少缩放因子
        # if (epoch + 1) in [0, 250000]:
        #     ratio = ratio * 0.1
        #     print('--------------------------------')
        #     print(ratio)

            # for i in workers:
            #     models[i].train()
            #     if (epoch + 1) % decay_period == 0:
            #         for param_group in optimizers_list[i - 1].param_groups:
            #             param_group['lr'] *= 0.1
            #             print('LR Decreased! Now: {}'.format(param_group['lr']))

    f_trainloss.close()



def init_processes(workers,
                   models, save_path,
                   train_dataset_list, test_dataset,iterations_epoch,
                   fn, backend='tcp'):
    fn(workers, models, save_path, train_dataset_list, test_dataset, iterations_epoch)


if __name__ == '__main__':

    torch.manual_seed(1)
    workers_num = args.workers
    workers = [v+1 for v in range(workers_num)]
    models = []

    for i in range(workers_num + 1):
        if args.model == 'MnistCNN':
            model = MnistCNN()

            train_transform, test_transform = get_data_transform('mnist')

            train_dataset = datasets.MNIST(args.data_dir, train=True, download=False,
                                           transform=train_transform)
            test_dataset = datasets.MNIST(args.data_dir, train=False, download=False,
                                          transform=test_transform)
        elif args.model == 'LeNet':
            model = LeNetForMNIST()

            train_transform, test_transform = get_data_transform('mnist')

            train_dataset = datasets.MNIST(args.data_dir, train=True, download=False,
                                           transform=train_transform)
            test_dataset = datasets.MNIST(args.data_dir, train=False, download=False,
                                          transform=test_transform)
        elif args.model == 'LROnMnist':
            model = ResNetOnCifar10.LROnMnist()
            train_transform, test_transform = get_data_transform('mnist')

            train_dataset = datasets.MNIST(args.data_dir, train=True, download=True,
                                           transform=train_transform)
            test_dataset = datasets.MNIST(args.data_dir, train=False, download=True,
                                          transform=test_transform)
        elif args.model == 'LROnCifar10':
            model = ResNetOnCifar10.LROnCifar10()
            train_transform, test_transform = get_data_transform('cifar')

            train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=False,
                                           transform=train_transform)
            test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=False,
                                          transform=test_transform)
        elif args.model == 'AlexNet':

            train_transform, test_transform = get_data_transform('cifar')

            if args.data_name == 'cifar10':
                model = AlexNetForCIFAR()
                train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=False,
                                                 transform=train_transform)
                test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=False,
                                                transform=test_transform)
            else:
                model = AlexNetForCIFAR(num_classes=100)
                train_dataset = datasets.CIFAR100(args.data_dir, train=True, download=False,
                                                  transform=train_transform)
                test_dataset = datasets.CIFAR100(args.data_dir, train=False, download=False,
                                                 transform=test_transform)
        elif args.model == 'ResNet18OnCifar10':
            model = ResNetOnCifar10.ResNet18()

            train_transform, test_transform = get_data_transform('cifar')
            train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=True,
                                             transform=train_transform)
            test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True,
                                            transform=test_transform)
        elif args.model == 'ResNet34':
            model = models.resnet34(pretrained=False)

            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            test_transform = train_transform
            train_dataset = datasets.ImageFolder(args.data_dir, train=True, download=False,
                                             transform=train_transform)
            test_dataset = datasets.ImageFolder(args.data_dir, train=False, download=False,
                                            transform=test_transform)
        elif args.model == 'VGG11':
            model = vgg.vgg11()

            train_transform, test_transform = get_data_transform('cifar')
            train_dataset = datasets.CIFAR100(args.data_dir, train=True, download=False,
                                             transform=train_transform)
            test_dataset = datasets.CIFAR100(args.data_dir, train=False, download=False,
                                            transform=test_transform)
        else:
            print('Model must be {} or {}!'.format('MnistCNN', 'AlexNet'))
            sys.exit(-1)
        models.append(model)
    train_bsz = args.train_bsz
    train_bsz /= len(workers)
    train_bsz = int(train_bsz)

    train_data = partition_dataset(train_dataset, workers)
    train_data_list = []
    for i in workers:
        train_data_sub = select_dataset(workers, i, train_data, batch_size=train_bsz)
        train_data_list.append(train_data_sub)

    test_bsz = 400
    # 用所有的测试数据测试
    test_data = DataLoader(test_dataset, batch_size=test_bsz, shuffle = False)

    iterations_epoch = int(len(train_dataset) / args.train_bsz)

    save_path = str(args.save_path)
    save_path = save_path.rstrip('/')

    p = TorchProcess(target=init_processes, args=(workers,
                                                  models, save_path,
                                                  train_data_list, test_data,iterations_epoch,
                                                  run))
    p.start()
    p.join()
