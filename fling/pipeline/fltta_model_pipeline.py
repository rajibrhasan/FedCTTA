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


def non_iid_continual(is_niid, client_number, corupt_number):
    if not is_niid:
        corupt_map = np.array([[i for i in range(corupt_number)] for _ in range(client_number)])
    else:
        # corupt_map = np.array([[-1, ] * corupt_number] * client_number)
        # for j in range(client_number):
        #     for i in range(corupt_number):
        #         temp = random.randint(0, corupt_number - 1)
        #         while temp in corupt_map[j]:
        #             temp = (temp + 1) % corupt_number
        #         corupt_map[j][i] = temp
        corupt_map = np.array([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],

            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],

            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],

            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        ])
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

    return group
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


def FedTTA_Pipeline(args: dict, seed: int = 0) -> None:
    # Compile the input arguments first.
    args = compile_config(args, seed)

    # Construct logger.
    logger = Logger(args.other.logging_path)

    # Load origin train dataset `origin_train_set`, origin test dataset `origin_test_set`
    dataset_name = args.data.dataset
    args.data.dataset = args.data.dataset.split('_')[0]
    print(dataset_name, args.data.dataset)
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
    print(net)
    if args.other.pre_trained == 'wideresnet28' and args.data.class_number == 10:
        ckpt = ckpt['state_dict']
        net.load_state_dict(ckpt)
    elif args.other.pre_trained == 'wideresnet28' and args.data.class_number == 100:
        ckpt = ckpt['model_state_dict']
        net.load_state_dict(ckpt)
    elif args.other.pre_trained == 'cifarresnext' and args.data.class_number == 100:
        ckpt = {key.replace("module.", ""): value for key, value in ckpt.items()}
        net.load_state_dict(ckpt)
    else:
        net.load_state_dict(ckpt)

    net.cuda()

    # test_origin(net, test_dataloader)

    # Initialize group, clients and server.
    group = init_tta_state(args, net, ckpt, logger, corrupt_dict, corrupt_test_sets, origin_test_sets, train_dataloader)

    # Training loop
    test_monitor_list = [VariableMonitor() for _ in range(len(args.data.corruption))]
    adapt_monitor_list = [VariableMonitor() for _ in range(len(args.data.corruption))]
    fed_adapt_monitor_list = [VariableMonitor() for _ in range(len(args.data.corruption))]

    for level in args.data.level:
        for cidx in range(len(args.data.corruption)):
            # determine the corruption
            logger.logging('Starting Federated Test-Time Adaptation round: ')

            corupt_map = non_iid_continual(is_niid=args.other.niid, client_number=args.client.client_num,
                                           corupt_number=len(args.data.corruption))
            participated_clients = client_sampling(range(args.client.client_num), args.client.sample_rate)

            # Random sample participated clients in each communication round.
            if not args.other.is_continue:
                group = init_tta_state(args, net, ckpt, logger, corrupt_dict, corrupt_test_sets, origin_test_sets, train_dataloader)

            global_eps = int(
                len(corrupt_test_sets[0][args.data.corruption[corupt_map[0][cidx]]][level]) / args.other.ttt_batch)

            for i in range(global_eps):
                global_feature_indicator = []
                global_mean = []

                # Update each batch
                if not args.other.online:
                    group = init_tta_state(args, net, ckpt, logger, corrupt_dict, corrupt_test_sets, origin_test_sets)

                for j in tqdm.tqdm(participated_clients):
                    # Collect test data
                    corupt = args.data.corruption[corupt_map[j][cidx]]

                    indexs = corrupt_test_sets[j][corupt][level].indexes[0:  args.other.ttt_batch]
                    dataset = corrupt_test_sets[j][corupt][level].tot_data
                    corrupt_test_sets[j][corupt][level].indexes = corrupt_test_sets[j][corupt][level].indexes[args.other.ttt_batch : ]
                    inference_data = NaiveDataset(tot_data=dataset, indexes=indexs)

                    # Test Before Adaptation
                    test_monitor, feature_indicator = group.clients[j].test_source(test_data=inference_data)
                    test_monitor_list[cidx].append(test_monitor)
                    logger.logging(
                        f'Client {j} Corupt {corupt}: Old Test Acc {test_monitor["test_acc"]}, Old Test Loss {test_monitor["test_loss"]}'
                    )
                    global_feature_indicator.append(feature_indicator)

                    #  Client Test Along with Adaptation
                    if args.other.method == 'bn':
                        test_mean, adapt_monitor = group.clients[j].adapt(test_data=inference_data)
                        global_mean.append(test_mean)
                    else:
                        adapt_monitor = group.clients[j].adapt(test_data=inference_data)
                    adapt_monitor_list[cidx].append(adapt_monitor)

                # Aggregate parameters in each client.
                if args.other.is_average:
                    logger.logging('-' * 10 + ' Average ' + '-' * 10)
                    if args.other.method == 'bn':
                        group.aggregate_bn(i, global_mean, global_feature_indicator)
                    else:
                        group.aggregate_grad(i, global_feature_indicator)

                for j in tqdm.tqdm(participated_clients):
                    # adapt_monitor = group.clients[j].adapt()
                    adapt_monitor = group.clients[j].inference()
                    fed_adapt_monitor_list[cidx].append(adapt_monitor)

                logger.logging(
                    f'Coruption Type {corupt}, level {level}, Old Test Acc {adapt_monitor_list[cidx].variable_mean()["test_acc"]}\n,'
                    f'Fed Adapt Test Acc {fed_adapt_monitor_list[cidx].variable_mean()["test_acc"]}'
                )

    # Print & Save the outcome
    data_record = np.array([[0. for _ in range(len(args.data.corruption))] for _ in range(3)])
    for cidx in range(len(args.data.corruption)):
        corupt = args.data.corruption[cidx]
        # Origin
        mean_test_variables = test_monitor_list[cidx].variable_mean()
        data_record[0][cidx] = mean_test_variables["test_acc"]
        logger.logging(
            f'Corruption Type {corupt}, level {args.data.level}: Old Test Acc {mean_test_variables["test_acc"]}, Old Test Loss {mean_test_variables["test_loss"]}')

        mean_adapt_variables = adapt_monitor_list[cidx].variable_mean()
        data_record[1][cidx] = mean_adapt_variables["test_acc"]
        logger.logging(
            f'Corruption Type {corupt}, level {args.data.level}: Adapt Acc {mean_adapt_variables["test_acc"]}, Adapt Loss {mean_adapt_variables["test_loss"]}')

        mean_fed_variables = fed_adapt_monitor_list[cidx].variable_mean()
        data_record[2][cidx] = mean_fed_variables["test_acc"]
        logger.logging(
            f'Corruption Type {corupt}, level {args.data.level}: Fed Acc {mean_fed_variables["test_acc"]}, Fed Loss {mean_fed_variables["test_loss"]}')

    dfData = {
        '序号': ['Before', 'Adapt', 'Fed'],
    }

    for cidx in range(len(args.data.corruption)):
        corupt = args.data.corruption[cidx]
        dfData[corupt] = data_record[:, cidx]
    df = pd.DataFrame(dfData)
    df.to_excel(os.path.join(args.other.logging_path, 'outcome.xlsx'), index=False)


