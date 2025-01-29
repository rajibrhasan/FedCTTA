import os.path

import tqdm
import random
import numpy as np
import torch
import torch.nn as nn

from fling.component.client import get_client
from fling.component.server import get_server
from fling.component.group import get_group
from fling.dataset import get_dataset
from fling.utils.data_utils import data_sampling
from fling.utils import Logger, compile_config, client_sampling, VariableMonitor, LRScheduler

from fling.model import get_model
from torch.utils.data import DataLoader
from fling.utils.data_utils.sampling import NaiveDataset

import pandas as pd
from fling.utils.utils import VariableMonitor, SaveEmb
import pickle
import wandb
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import datetime
import time





def non_iid_continual(args, is_niid, client_number, corupt_number):
    if not is_niid:
        corupt_map = np.array([[i for i in range(corupt_number)] for _ in range(client_number)])
    else:
        corupt_map = np.load(args.other.partition_path)
    return corupt_map

def test_origin(net, test_dataloader):
    # Without Corupt Test
    monitor = VariableMonitor()
    net.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for _, data in enumerate(test_dataloader):
            preprocessed_data = {'x': data['input'].cuda(), 'y': data['class_id'].cuda()}
            batch_x, batch_y = preprocessed_data['x'], preprocessed_data['y']

            out = net(batch_x)
            y_pred = torch.argmax(out, dim=-1)
            loss = criterion(out, batch_y)
            monitor.append(
                {
                    'test_acc': torch.mean((y_pred == preprocessed_data['y']).float()).item(),
                    'test_loss': loss.item()
                },
                weight=preprocessed_data['y'].shape[0]
            )
    mean_monitor_variables = monitor.variable_mean()
    print(f'Origin test acc {mean_monitor_variables["test_acc"]}, test loss {mean_monitor_variables["test_loss"]}')

def init_tta_state(args, net, ckpt, logger, corrupt_dict, corrupt_test_sets, origin_test_sets, trainloader):

    group = get_group(args, logger)
    group.server = get_server(args, test_dataset=corrupt_dict)
    for i in range(args.client.client_num):
        group.append(
            get_client(args=args, client_id=i, train_dataset=corrupt_test_sets[i], test_dataset=origin_test_sets[i]))
        group.clients[i].init_weight(ckpt=ckpt)
    group.initialize()

    if args.client.name == 'fedthe_client':
        global_rep = init_train_feature(args, net, trainloader)
        for i in range(args.client.client_num):
            group.clients[i].init_globalrep(global_rep)
    elif args.client.name == 'fedactmad_client':
        clean_feature_mean, clean_feature_var = init_train_bn(net, trainloader)
        for i in range(args.client.client_num):
            group.clients[i].update_statistics(clean_feature_mean, clean_feature_var)

    return group

def init_train_bn(net, trainloader):
    chosen_layers = []
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            chosen_layers.append(m)

    n_chosen_layers = len(chosen_layers)
    hook_list = [SaveEmb() for _ in range(n_chosen_layers)]
    clean_feature_mean = []
    clean_feature_var = []
    for idx, data in enumerate(trainloader):
        hooks = [chosen_layers[i].register_forward_hook(hook_list[i]) for i in range(n_chosen_layers)]
        inputs = data['input'].cuda()
        with torch.no_grad():
            net.eval()
            _ = net(inputs)
            for yy in range(n_chosen_layers):
                hook_list[yy].statistics_update(), hook_list[yy].clear(), hooks[yy].remove()

    for i in range(n_chosen_layers):
        clean_feature_mean.append(hook_list[i].pop_mean()), clean_feature_var.append(hook_list[i].pop_var())
    return clean_feature_mean, clean_feature_var

def init_train_feature(args, net, trainloader):
    feature_bank = []
    with torch.no_grad():
        for idx, data in enumerate(trainloader):
            x, y = data['input'].to(args.learn.device), data['class_id'].to(args.learn.device)
            feature, out = net(x, mode='compute-feature-logit')
            if feature_bank == []:
                feature_bank = feature
            else:
                feature_bank = torch.concatenate((feature_bank, feature), dim=0)
    feature_mean = torch.mean(feature_bank, dim=0)
    return feature_mean



def compute_average_differences(group):
    model_count = len(group.clients)
    avg_differences = []
    
    # Compute pairwise average differences
    for i in range(model_count):
        for j in range(i + 1, model_count):
            total_diff = 0.0
            layer_count = 0
            
            # Iterate through all layers
            for name, param in group.clients[0].model.named_parameters():
                layer_params_i = group.clients[i].model.state_dict()[name]
                layer_params_j = group.clients[j].model.state_dict()[name]
                
                # Compute norm for the current layer
                diff = torch.norm(layer_params_i - layer_params_j).item()
                total_diff += diff
                layer_count += 1
            
            # Compute average difference
            avg_diff = total_diff / layer_count
            avg_differences.append((f"model_{i+1}vs_model{j+1}", avg_diff))
    
    return avg_differences

def compute_differences(features, f_name, mode = 'diff'):
    model_count = len(features)
    differences = []
    
    # Compute pairwise average differences
    for i in range(model_count):
        for j in range(i, model_count):
            if mode == 'diff':
                diff = torch.norm(features[i] - features[j]).item()
                differences.append((f"client_{i+1}_vs_client{j+1}", diff))
            elif mode == 'sim':
                sim = torch.nn.functional.cosine_similarity(features[i].unsqueeze(0), features[j].unsqueeze(0))
                differences.append((f"client_{i+1}_vs_client{j+1}", sim))

            else:
                raise NotImplementedError


    print(f"Average Differences Between {f_name}")
    for model_pair, diff in differences:
        wandb.log({f"{model_pair}": diff})
    
def FedTTA_Pipeline(args: dict, seed: int = 0) -> None:
    start_time = time.time()
    
    # Compile the input arguments first.
    args = compile_config(args, seed)

    # Construct logger.
    logger = Logger(args.other.logging_path)

    # Load origin train dataset `origin_train_set`, origin test dataset `origin_test_set`
    dataset_name = args.data.dataset
    pos = args.data.dataset.rfind('_')
    if pos != -1:
        args.data.dataset = args.data.dataset[:pos]
    print(dataset_name, args.data.dataset)

    wandb.login(key="f22f65f9d8ca626c5d8a36d80eee6506d14501c3")
    folder_name_split = args.other.logging_path.split('/')
    wandb.init(project=f"fed_tta_{dataset_name}_{folder_name_split[-2]}_{folder_name_split[-3]}", dir="output", name=folder_name_split[-1])
    
    origin_test_set = get_dataset(args, train=False)
    origin_train_set = get_dataset(args, train=True)
    train_dataloader = DataLoader(origin_train_set, batch_size=args.learn.batch_size, shuffle=True)
    test_dataloader = DataLoader(origin_test_set, batch_size=args.learn.batch_size, shuffle=False)


    # Corrupted Type & Level Test Data
    '''
    corrupt_dict[ corrupt_type ][ corrupt_severity ]
    '''
    args.data.dataset = dataset_name
    corrupt_dict = {}
    common_corrupt = args.data.corruption
    common_serverity = args.data.level
    for corrupt in common_corrupt:
        corrupt_dict[corrupt] = {}
        for serv in common_serverity:
            args.data.corruption = corrupt
            args.data.level = serv
            test_set = get_dataset(args, train=False)
            corrupt_dict[corrupt][serv] = test_set
    args.data.corruption = common_corrupt
    args.data.level = common_serverity

    # Split dataset into clients.
    origin_test_sets = data_sampling(origin_test_set, args, seed, train=False)
    '''
    corrupt_test_sets[ client_id ][ corrupt_type ][ corrupt_severity ]
    '''
    corrupt_test_sets = [{} for _ in range(args.client.client_num)]
    for corrupt in common_corrupt:
        for cidx in range(args.client.client_num):
            corrupt_test_sets[cidx][corrupt] = {}
        for serv in common_serverity:
            test_sets = data_sampling(corrupt_dict[corrupt][serv], args, seed, train=False)
            for cidx in range(args.client.client_num):
                corrupt_test_sets[cidx][corrupt][serv] = test_sets[cidx]

    # load pre-trained net
    ckpt = torch.load(args.other.model_path)
    net = get_model(args)
    if 'tiny' in args.data.dataset:
        net.avgpool = nn.AdaptiveAvgPool2d(1)
        num_features = net.fc.in_features
        net.fc = nn.Linear(num_features, 200)
    print(net)

    print(ckpt.keys())


    if args.other.pre_trained == 'wideresnet' and args.data.class_number == 10:
        # ckpt = ckpt['state_dict']
        net.load_state_dict(ckpt)
    elif args.other.pre_trained == 'wideresnet28' and args.data.class_number == 100:
        ckpt = ckpt['model_state_dict']
        net.load_state_dict(ckpt)
    elif args.other.pre_trained == 'cifarresnext' and args.data.class_number == 100:
        ckpt = {key.replace("module.", ""): value for key, value in ckpt.items()}
        net.load_state_dict(ckpt)
    elif args.other.pre_trained == 'resnet' and args.data.class_number == 100:
        del ckpt['mu']
        del ckpt['sigma']
        net.load_state_dict(ckpt)
    else:
        net.load_state_dict(ckpt)

    net.cuda()

    test_origin(net, test_dataloader)

    # Initialize group, clients and server.
    group = init_tta_state(args, net, ckpt, logger, corrupt_dict, corrupt_test_sets, origin_test_sets, train_dataloader)

    # Training loop
    test_monitor_list = [VariableMonitor() for _ in range(len(args.data.corruption))]
    adapt_monitor_list = [VariableMonitor() for _ in range(len(args.data.corruption))]
    fed_adapt_monitor_list = [VariableMonitor() for _ in range(len(args.data.corruption))]

    # calculate loop
    all_loop = int(len(corrupt_test_sets[0][args.data.corruption[0]][args.data.level[0]]) /
                    args.other.ttt_batch)
    avg_loop = all_loop // args.other.loop
    last_add = all_loop % args.other.loop
    global_eps = [avg_loop for _ in range(args.other.loop-1)] + [avg_loop+last_add]

    print(args.client.sample_rate)
    print(args.other.is_continue)
    print(args.other.online)

    cnt = 0

    for level in args.data.level:
        for lp in tqdm.tqdm(range(args.other.loop)):
            for cidx in range(len(args.data.corruption)):
                # determine the corruption
                logger.logging('Starting Federated Test-Time Adaptation round: ')

                corupt_map = non_iid_continual(args=args, is_niid=args.other.niid, client_number=args.client.client_num,
                                               corupt_number=len(args.data.corruption))
                participated_clients = client_sampling(range(args.client.client_num), args.client.sample_rate)

                # Random sample participated clients in each communication round.
                if not args.other.is_continue:
                    group = init_tta_state(args, net, ckpt, logger, corrupt_dict, corrupt_test_sets, origin_test_sets, train_dataloader)
                    print('Here in the is continue')

                for i in range(global_eps[lp]):
                    cnt += 1
                    global_feature_indicator = []
                    global_mean = []

                    # Update each batch
                    if not args.other.online:
                        group = init_tta_state(args, net, ckpt, logger, corrupt_dict, corrupt_test_sets, origin_test_sets)
                        print('Here in the online')

                    for j in participated_clients:
                        # Collect test data
                        corupt = args.data.corruption[corupt_map[j][cidx]]
                
                        indexs = corrupt_test_sets[j][corupt][level].indexes[0:  args.other.ttt_batch]
                        dataset = corrupt_test_sets[j][corupt][level].tot_data
                        corrupt_test_sets[j][corupt][level].indexes = corrupt_test_sets[j][corupt][level].indexes[args.other.ttt_batch : ]
                        inference_data = NaiveDataset(tot_data=dataset, indexes=indexs)

                        # Test Before Adaptation
                        test_monitor, feature_indicator = group.clients[j].test_source(test_data=inference_data)
                        test_monitor_list[cidx].append(test_monitor)
                        # logger.logging(
                        #     f'Client {j} Corupt {corupt}: Old Test Acc {test_monitor["test_acc"]}, Old Test Loss {test_monitor["test_loss"]}'
                        # )
                       

                        #  Client Test Along with Adaptation
                        if args.other.method == 'bn':
                            test_mean, adapt_monitor = group.clients[j].adapt(test_data=inference_data)
                            global_mean.append(test_mean)
                        else:
                            adapt_monitor = group.clients[j].adapt(test_data=inference_data)
                        adapt_monitor_list[cidx].append(adapt_monitor)
                        # fed_adapt_monitor_list[cidx].append(adapt_monitor)

                        if args.method.data_used != "original":
                            if args.method.data_used == 'random':
                                test_data = {}
                                test_data['input'] = torch.rand((64, 3, 32, 32))
                                test_data['class_id'] = torch.randint(low = 0, high=100, size = (64,))
                                
                            elif args.method.data_used =='cifar':
                                for _, data in enumerate(test_dataloader):
                                    test_data = data
                                    break
                            else:
                                raise NotImplementedError
                                
                            feature_indicator = group.clients[j].get_logits(test_data)

                        global_feature_indicator.append(feature_indicator)

                    # Aggregate parameters in each client.
                    if args.other.is_average and cnt % 10 == 0 :
                        logger.logging('-' * 10 + ' Average ' + '-' * 10)
                        if args.other.method == 'bn':
                            if args.method.name == 'ours':
                                print('Ours')
                                group.aggregate_bn_ours(i, global_mean, global_feature_indicator)
                            else:
                                print(args.method.name)
                                group.aggregate_bn(i, global_mean, global_feature_indicator)
                        else:
                            if args.method.name == 'ours':
                                print('Ours')
                                group.aggregate_grad_ours(i, global_feature_indicator)
                  
                            else:
                                print(args.method.name)
                                group.aggregate_grad(i, global_feature_indicator)

                    for j in participated_clients:
                        if 'ft' in args.method.name:
                            adapt_monitor = group.clients[j].adapt(test_data=inference_data)
                        else:

                            adapt_monitor = group.clients[j].inference()
                        fed_adapt_monitor_list[cidx].append(adapt_monitor)
                    
                    # tsne_plot(group, participated_clients)
                    # Compute and log average differences
                    # compute_average_differences(group)
                   

                    # if args.method.name == 'ours':
                    #     diff_mode = 'diff'

                    #     if args.method.feat_sim == 'feature':
                    #         compute_differences(global_feature_indicator, 'Feature Mean', diff_mode)

                    #     elif args.method.feat_sim == 'pvec':
                    #         compute_differences(global_feature_indicator, 'Principal Vector', diff_mode)
                        
                    #     elif args.method.feat_sim == 'output':
                    #         compute_differences(global_feature_indicator, 'Output', diff_mode)
                        
                    #     elif args.method.feat_sim == 'model':
                    #         flattened_weights_list = []
                    #         for client in group.clients:
                    #             # Flatten and concatenate all parameters, detaching them from the computation graph
                    #             client.model.requires_grad_(True)
                    #             flattened_weights = torch.cat([param.view(-1).detach() for param in client.model.parameters() if param.requires_grad])
                    #             flattened_weights_list.append(flattened_weights)
                            
                    #         compute_differences(flattened_weights_list, 'Model Weight', diff_mode)
                        
                    #     elif args.method.feat_sim == 'gradient':
                    #         flattened_gradients_list = []
                    #         for client in group.clients:
                    #             flattened_gradients = torch.cat([param.grad.view(-1) for param in client.model.parameters() if param.grad is not None])
                    #             flattened_gradients_list.append(flattened_gradients)
                    #         compute_differences(flattened_gradients_list, 'Gradient', diff_mode)

                    
                    # tsne_plot(group, participated_clients, args.other.logging_path)
                    num_round = cidx * global_eps[lp] + i

                    logger.logging(
                        f'{num_round} / 750 | Coruption Type: {corupt}{level} | Adapt: {adapt_monitor_list[cidx].variable_mean()["test_acc"]: 0.3f} | Fed Adapt: {fed_adapt_monitor_list[cidx].variable_mean()["test_acc"]: 0.3f}'
                    )
    if args.group.name == 'adapt_group' or args.group.name == 'fedamp_group' or args.group.name == 'fedgraph_group':
        with open(os.path.join(args.other.logging_path, 'collaboration.pkl'), 'wb') as f:
            pickle.dump(group.collaboration_graph, f)
    # Print & Save the outcome
    data_record = np.array([[0. for _ in range(len(args.data.corruption))] for _ in range(3)])
    for cidx in range(len(args.data.corruption)):
        corupt = args.data.corruption[cidx]
        # Origin
        mean_test_variables = test_monitor_list[cidx].variable_mean()
        data_record[0][cidx] = mean_test_variables["test_acc"]
        logger.logging(
            f'Corruption Type: {corupt}{args.data.level} | Old Test Acc:  {mean_test_variables["test_acc"]: 0.3f} | Old Test Loss {mean_test_variables["test_loss"]: 0.3f}')

        mean_adapt_variables = adapt_monitor_list[cidx].variable_mean()
        data_record[1][cidx] = mean_adapt_variables["test_acc"]
        logger.logging(
            f'Corruption Type: {corupt}{args.data.level} | Adapt Acc: {mean_adapt_variables["test_acc"]: 0.3f} | Adapt Loss {mean_adapt_variables["test_loss"]: 0.3f}')

        mean_fed_variables = fed_adapt_monitor_list[cidx].variable_mean()
        data_record[2][cidx] = mean_fed_variables["test_acc"]
        logger.logging(
            f'Corruption Type: {corupt}{args.data.level} | Fed Acc:  {mean_fed_variables["test_acc"]:0.3f} | Fed Loss: {mean_fed_variables["test_loss"]: 0.3f}')

    dfData = {
        '序号': ['Before', 'Adapt', 'Fed'],
    }
    data_record_mean = np.mean(data_record, axis=1)
    for cidx in range(len(args.data.corruption)):
        corupt = args.data.corruption[cidx]
        dfData[corupt] = data_record[:, cidx]
    dfData['Avg'] = data_record_mean
    df = pd.DataFrame(dfData)
    print(dfData)
    df.to_excel(os.path.join(args.other.logging_path, 'outcome.xlsx'), index=False)

    logger.logging(f"Test acc: {data_record_mean[0] * 100 : 0.2f}")
    logger.logging(f"Adapt acc: {data_record_mean[1] * 100 : 0.2f}")
    logger.logging(f"Fed acc: {data_record_mean[2] * 100 : 0.2f}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Convert to hours, minutes, and seconds
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    logger.logging(f"Time elapsed: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    # /teamspace/studios/this_studio/FedCTTA/fling/pipeline/fltta_model_pipeline.