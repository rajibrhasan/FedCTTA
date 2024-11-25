import yaml

import argparse
import torch
import torch.nn.functional as F
import random
from copy import deepcopy
from client import Client
from robustbench.data import load_cifar10c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from sklearn.decomposition import PCA
from methods.fed_avg import FedAvg
from yacs.config import CfgNode as CfgNode
from fed_utils import split_indices_into_batches, get_available_corruptions, get_dataset, create_schedule_iid, create_schedule_niid, cosine_similarity
from conf import cfg, load_cfg_fom_args
from tqdm import tqdm
import numpy as np
import logging

logger = logging.getLogger(__name__)

def main(severity, device):
    print(f"===============================Severity: {severity} || IID : {cfg.MISC.IID}===============================")
    max_use_count = cfg.CORRUPTION.NUM_EX // cfg.MISC.BATCH_SIZE 
    
    dataset = get_dataset(cfg, severity, cfg.CORRUPTION.DATASET)
    clients = []
    global_model = load_model(cfg.MODEL.ARCH, cfg.MISC.CKPT_DIR, cfg.CORRUPTION.DATASET, ThreatModel.corruptions)

    for i in range(cfg.MISC.NUM_CLIENTS):
        clients.append(Client(f'client_{i}', deepcopy(global_model), cfg, device))

    if cfg.MISC.IID:
        print('IID')
        client_schedule = create_schedule_iid(cfg.MISC.NUM_CLIENTS, cfg.MISC.NUM_STEPS, cfg.CORRUPTION.TYPE, cfg.MISC.TEMPORAL_H)
    else:
        print('Non-IID')
        client_schedule = create_schedule_niid(cfg.MISC.NUM_CLIENTS, cfg.MISC.NUM_STEPS, cfg.CORRUPTION.TYPE, cfg.MISC.TEMPORAL_H, cfg.MISC.SPATIAL_H)
    
    # logger.info('Client schedule: \n')
    # logger.info(client_schedule)
    
    for t in tqdm(range(cfg.MISC.NUM_STEPS)):
        w_locals = []
        for idx, client in enumerate(clients):
            selected_domain = dataset[client_schedule[idx][t]]
            cur_idx = selected_domain['indices'][selected_domain['use_count']]
            x = selected_domain['all_x'][cur_idx]
            y = selected_domain['all_y'][cur_idx]
            client.domain_list.append(client_schedule[idx][t])
            client.adapt(x, y)
            w_locals.append(deepcopy(client.get_state_dict()))
            selected_domain['use_count'] += 1
        
        if cfg.MODEL.ADAPTATION == 'fedavg':
            w_avg = FedAvg(w_locals)
            for client in clients:
                client.set_state_dict(deepcopy(w_avg))

        elif cfg.MODEL.ADAPTATION == 'roid':
            similarity_mat = torch.zeros((len(clients), len(clients)))
            
            if cfg.MISC.USE_BN:
                bn_params_list = [client.extract_bn_weights_and_biases() for client in clients]
                with torch.no_grad():
                    for i, bn_params1 in enumerate(bn_params_list):
                        for j, bn_params2 in enumerate(bn_params_list):
                            similarity = cosine_similarity(bn_params1, bn_params2)
                            similarity_mat[i,j] = similarity
            else:
                ema_prob_list = [client.class_probs_ema for client in clients]
                with torch.no_grad():
                    for i, ema_prob1 in enumerate(ema_prob_list):
                        for j, ema_prob2 in enumerate(ema_prob_list):
                            similarity = F.cosine_similarity(ema_prob1.reshape(-1, 1), ema_prob2.reshape(-1,1))
                            similarity_mat[i,j] = similarity.item()

            
            scaled_similarity = np.array(similarity_mat / cfg.MISC.TEMP)
            # Apply softmax to normalize the similarity values for aggregation
            exp_scaled_similarity = np.exp(scaled_similarity - np.max(scaled_similarity, axis=1, keepdims=True))  # Subtract max for numerical stability
            # exp_scaled_similarity = np.exp(scaled_similarity)  # Subtract max for numerical stability
            normalized_similarity = exp_scaled_similarity / np.sum(exp_scaled_similarity, axis=1, keepdims=True)
            if t  % 10 == 0:
                print(normalized_similarity)


            # if t % 10 == 0:
            #     print(similarity_mat)
            
            for i in range(len(clients)):
                ww = FedAvg(w_locals, normalized_similarity[i])
                clients[i].set_state_dict(deepcopy(ww))


    acc = 0
    for client in clients:
        client_acc = sum(client.correct_preds) / sum(client.total_preds)*100
        acc += client_acc
        print(f'{client.name} accuracy: {client_acc: 0.3f}')

    print(f'Global accuracy: {acc/len(clients) : 0.3f}')
    logger.info(f'Global accuracy: {acc/len(clients) : 0.3f}')

if __name__ == '__main__':
    load_cfg_fom_args("CIFAR-10C Evaluation")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(cfg)
    for severity in cfg.CORRUPTION.SEVERITY:
        main(severity, device)
