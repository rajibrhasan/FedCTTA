import copy

import matplotlib.pyplot as plt

from fling.utils import get_params_number
from fling.utils.compress_utils import *
from fling.utils.registry_utils import GROUP_REGISTRY
from fling.utils import VariableMonitor
from fling.component.client import ClientTemplate
import torch
from functools import reduce
import numpy as np
import torch.nn.functional as F
from sklearn.cluster import KMeans
from fling.model.stattention import ST_block, SpatialAttention
from fling.component.group import ParameterServerGroup

@GROUP_REGISTRY.register('adapt_group')
class TTAServerGroup(ParameterServerGroup):
    r"""
    Overview:
        Base implementation of the group in federated learning.
    """

    def __init__(self, args: dict, logger: VariableMonitor):
        r"""
        Overview:
            Implementation of the group in FedTTA
        """
        super(TTAServerGroup, self).__init__(args, logger)
        self.history_feature = []
        self.history_weight = [[] for _ in range(args.client.client_num)]
        self.indicator = torch.tensor([])
        self.time_slide = 10
        self.collaboration_graph = []


    def initialize(self) -> None:
        r"""
        Overview:
            In this function, several things will be done:
            1) Set ``fed_key`` in each client is determined, determine which parameters shoul be included for federated
        learning.
            2) ``glob_dict`` in the server is determined, which is exactly a state dict with all keys in ``fed_keys``.
            3) Each client local model will be updated by ``glob_dict``.
        Returns:
            - None
        """
        # Step 1.
        if self.args.group.aggregation_parameters.name == 'all':
            fed_keys = self.clients[0].model.state_dict().keys()
        elif self.args.group.aggregation_parameters.name == 'contain':
            keywords = self.args.group.aggregation_parameters.keywords
            fed_keys = []
            for kw in keywords:
                for k in self.clients[0].model.state_dict():
                    if kw in k:
                        fed_keys.append(k)
            fed_keys = list(set(fed_keys))
        elif self.args.group.aggregation_parameters.name == 'except':
            keywords = self.args.group.aggregation_parameters.keywords
            fed_keys = []
            for kw in keywords:
                for k in self.clients[0].model.state_dict():
                    if kw in k:
                        fed_keys.append(k)
            fed_keys = list(set(self.clients[0].model.state_dict().keys()) - set(fed_keys))
        elif self.args.group.aggregation_parameters.name == 'include':
            keywords = self.args.group.aggregation_parameters.keywords
            fed_keys = []
            for name, param in self.clients[0].model.named_parameters():
                if keywords in name:
                    fed_keys.append(name)
            fed_keys = list(set(fed_keys))
        else:
            raise ValueError(f'Unrecognized aggregation_parameters.name: {self.args.group.aggregation_parameters.name}')

        # Step 2.
        self.logger.logging(f'Weights for federated training: {fed_keys}')
        glob_dict = {k: self.clients[0].model.state_dict()[k] for k in fed_keys}
        self.server.glob_dict = glob_dict
        self.set_fed_keys()

        # Step 3.
        if not self.args.other.resume:
            self.sync()

        # Logging model information.
        self.logger.logging(str(self.clients[0].model))
        self.logger.logging('All clients initialized.')
        self.logger.logging(
            'Parameter number in each model: {:.2f}M'.format(get_params_number(self.clients[0].model) / 1e6)
        )

    def st_agg_grad(self, time_att=None, space_att=None, wotime=False):
        client_num = self.args.client.client_num
        # time_att = torch.mean(time_att, dim=1)
        weight_list = []
        for cidx in range(client_num):
            w_time = copy.deepcopy(self.clients[cidx].model.state_dict())
            # if not wotime:
            #     T_all = time_att.shape[2]
            #     for k in w_time.keys():
            #         for tidx in range(T_all):
            #             if tidx == 0:
            #                 w_time[k] = time_att[cidx, -1, tidx] * self.history_weight[cidx][-T_all + tidx][k].cuda()
            #             else:
            #                 w_time[k] += time_att[cidx, -1, tidx] * self.history_weight[cidx][-T_all + tidx][k].cuda()
            # else:
            for k in w_time.keys():
                w_time[k] = w_time[k].cuda()
            weight_list.append(w_time)

        # print(space_att)
        for cidx in range(client_num):
            w_space = copy.deepcopy(self.clients[cidx].model.state_dict())
            S_all = space_att.shape[0]
            for k in w_space.keys():
                for sidx in range(S_all):
                    if sidx == 0:
                        w_space[k] = space_att[cidx, sidx] * weight_list[sidx][k]
                    else:
                        w_space[k] += space_att[cidx, sidx] * weight_list[sidx][k]

            # for k in w_space.keys():
            #     w_space[k] = 0.1 * w_space[k] + 0.9 * weight_list[cidx][k]
            
            self.clients[cidx].model.load_state_dict(w_space)

    def aggregate_grad_ours(self, train_round, feature_indicator):

        # client_num = self.args.client.client_num

        # feature_indicator = torch.stack(feature_indicator, dim=0)
        # if self.indicator.shape[0] == 0:
        #     self.indicator = feature_indicator.unsqueeze(1)
        # else:
        #     self.indicator = torch.cat([self.indicator, feature_indicator.unsqueeze(1)], dim=1)

        # self.time_slide = self.args.other.time_slide
        # if self.indicator.shape[1] < self.time_slide:
        #     feature_input = self.indicator[:, :, :]
        # else:
        #     feature_input = self.indicator[:, self.indicator.shape[1] - self.time_slide:, :]

        # print(feature_input.shape)
        
        # feature_input = feature_input.view(len(feature_input), -1)
        # # Normalize the data to unit vectors (important for cosine similarity)
        # normalized_data = F.normalize(feature_input, p=2, dim=1)

        # # Compute cosine similarity (20x20 matrix)
        # cosine_similarity_matrix = torch.matmul(normalized_data, normalized_data.T)

       
        # # Apply temperature scaling
        # temperature = 1  # Adjust temperature (smaller = sharper distribution)
        # scaled_similarity = cosine_similarity_matrix / temperature

        # # Apply softmax row-wise
        # softmax_similarity = F.softmax(scaled_similarity, dim=1)

        indicator = torch.stack(feature_indicator, dim = 0)

        if self.args.method.metric == 'euclid':
            pairwise_diff = indicator.unsqueeze(0) - indicator.unsqueeze(1)
            threshold = torch.mean(pairwise_diff)
            mask = (pairwise_diff <= threshold).all(dim=-1)  # Shape: [n, n]
            # `mask` is True where elements are within the threshold, False otherwise
            # print(mask)
            # Apply mask to pairwise_norm
            pairwise_norm = -1 * torch.norm(pairwise_diff, dim=-1)  # Shape: [n, n]
            # pairwise_norm = pairwise_norm.masked_fill(~mask, float('-inf'))  # Mask elements > threshold

            # Compute softmax on the masked pairwise_norm
            space_att = torch.softmax(pairwise_norm, dim=-1)  # Shape: [n, n]
        
        elif self.args.method.metric == 'cosine':
            normalized_indicator = F.normalize(indicator, p=2, dim=1)
            # Compute cosine similarity by multiplying the tensor with its transpose
            similarity_matrix = torch.mm(normalized_indicator, normalized_indicator.T)
            similarity_matrix = torch.exp(similarity_matrix)
            space_att = torch.softmax(similarity_matrix, dim = -1)

        elif self.args.method.metric == 'kl_div':
            P = F.softmax(indicator, dim=-1)  # Shape: (N, D)

            # Compute log probabilities for KL divergence
            log_P = torch.log(P + 1e-10)  # Avoid log(0)

            # Compute pairwise KL divergence using matrix multiplication
            # KL(P || Q) = sum(P * (log P - log Q))
            # We use broadcasting to compute log_Q for all pairs efficiently
            log_Q = log_P.unsqueeze(0)  # Shape: (1, N, D)
            kl_divergence = (P.unsqueeze(1) * (log_P.unsqueeze(1) - log_Q)).sum(dim=-1)  # Shape: (N, N)

            # Convert KL divergence to similarity score
            similarity_matrix = torch.exp(-kl_divergence)
            space_att = torch.softmax(similarity_matrix, dim = -1)

        elif self.args.method.metric == 'cross_entropy':
            P = F.softmax(logits, dim=-1)  # Shape: (N, D)
            log_P = torch.log(P + 1e-10)   # Log probabilities for numerical stability

            # Compute pairwise cross-entropy loss
            # CE(P || Q) = - sum(P * log(Q)), computed efficiently with broadcasting
            log_Q = log_P.unsqueeze(0)  # Shape: (1, N, D)
            ce_loss = -(P.unsqueeze(1) * log_Q).sum(dim=-1)  # Shape: (N, N)

            # Convert CE loss to similarity score (lower CE -> higher similarity)
            similarity_matrix = torch.exp(-ce_loss) 
            space_att = torch.softmax(similarity_matrix, dim = -1)


        self.st_agg_grad(space_att, space_att)
        self.collaboration_graph.append(space_att)


    def aggregate_grad(self, train_round, feature_indicator):
        client_num = self.args.client.client_num
        # for cidx in range(client_num):
        #     self.history_weight[cidx].append(self.clients[cidx].model.state_dict())

            

        if self.args.group.aggregation_method == 'st':
            time_att, space_att = self.ST_attention(feature_indicator)
            self.st_agg_grad(time_att, space_att)

        elif self.args.group.aggregation_method == 'avg':
            total_samples = float(sum([client.sample_num for client in self.clients]))
            w_avg = copy.deepcopy(self.clients[0].model.state_dict())
            for key in w_avg.keys():
                for cidx in range(client_num):
                    if cidx == 0:
                        continue
                    else:
                        w_avg[key] += self.clients[cidx].model.state_dict()[key]
                w_avg[key] = torch.div(w_avg[key], client_num)
            for client in self.clients:
                client.model.load_state_dict(w_avg)

        elif self.args.group.aggregation_method == 'sim':
            time_att, space_att = self.ST_similarity(feature_indicator)
            self.st_agg_grad(time_att, space_att)

        elif self.args.group.aggregation_method == 'wotime':
            time_att, space_att = self.ST_attention(feature_indicator, wotime=True)
            self.st_agg_grad(time_att, space_att, wotime=True)

        elif self.args.group.aggregation_method == 'base':
            space_att = self.S_similarity(feature_indicator)
            self.st_agg_grad(space_att=space_att, wotime=True)

        if self.args.group.aggregation_method != 'avg':
            self.collaboration_graph.append(space_att)

    def st_agg_bn(self, time_att=None, space_att=None, global_mean=None, wotime=False):
        n_chosen_layer = len(global_mean[0])
        client_num = self.args.client.client_num
        sum_mean = [[[] for _ in range(n_chosen_layer)] for _ in range(client_num)]
        sum_var = [[[] for _ in range(n_chosen_layer)] for _ in range(client_num)]
        for chosen_layer in range(n_chosen_layer):
            if self.history_feature[0].shape[1] < self.time_slide:
                feature_input = self.history_feature[chosen_layer][:, :, :]
            else:
                T_all = self.history_feature[chosen_layer].shape[1]
                feature_input = self.history_feature[chosen_layer][:, T_all - self.time_slide:, :]
            N, T, D = feature_input.shape
            heads = 1
            if not wotime:
                out = torch.matmul(time_att, feature_input.view(N, T, heads, D).permute(0, 2, 1, 3)).permute(0, 2, 1, 3).contiguous().view(N, T, heads * D)
                out = torch.matmul(space_att, out.view(N, T, heads, D).permute(1, 2, 0, 3)).permute(1, 2, 0, 3).contiguous().view(N, T, heads * D)
                # out = torch.matmul(time_att, feature_input.view(N, T, heads, D).permute(0, 2, 1, 3)).permute(0, 2, 1, 3).contiguous().view(N, T, heads, D)
                # out = torch.mean(out, dim=2)
                # out = torch.matmul(space_att, out.view(N, T, heads, D).permute(1, 2, 0, 3)).permute(1, 2, 0, 3).contiguous().view(N, T, heads, D)
                # out = torch.mean(out, dim=2)
            else:
                out = torch.matmul(space_att, feature_input.view(N, T, heads, D).permute(1, 2, 0, 3)).permute(1, 2, 0, 3).contiguous().view(N, T, heads * D)
                # out = torch.matmul(space_att, feature_input.view(N, T, heads, D).permute(1, 2, 0, 3)).permute(1, 2, 0, 3).contiguous().view(N, T, heads, D)
                # out = torch.mean(out, dim=2)

            half = out.shape[2] // 2
            for cidx in range(client_num):
                sum_mean[cidx][chosen_layer] = out[cidx][T - 1][:half]
                sum_var[cidx][chosen_layer] = out[cidx][T - 1][half:]
        return sum_mean, sum_var

    def aggregate_bn_ours(self, train_round, global_mean, feature_indicator):
        # Store feature mean and variance
        n_chosen_layer = len(global_mean[0])
        client_num = self.args.client.client_num
        if len(self.history_feature) == 0:
            self.history_feature = [[] for _ in range(n_chosen_layer)]
            for chosen_layer in range(n_chosen_layer):
                feature_t = []
                for cidx in range(client_num):
                    feature_t.append(global_mean[cidx][chosen_layer])
                feature_t = torch.stack(feature_t, dim=0)
                self.history_feature[chosen_layer] = feature_t.unsqueeze(1)
        else:
            for chosen_layer in range(n_chosen_layer):
                feature_t = []
                for cidx in range(client_num):
                    feature_t.append(global_mean[cidx][chosen_layer])
                feature_t = torch.stack(feature_t, dim=0)
                self.history_feature[chosen_layer] = torch.cat([self.history_feature[chosen_layer], feature_t.unsqueeze(1)], dim=1)

        # calculate aggregation rate & aggregate model weight
        sum_mean = [[[] for _ in range(n_chosen_layer)] for _ in range(client_num)]
        sum_var = [[[] for _ in range(n_chosen_layer)] for _ in range(client_num)]
       
        indicator = torch.stack(feature_indicator, dim = 0)

        if self.args.method.metric == 'euclid':
            pairwise_diff = indicator.unsqueeze(0) - indicator.unsqueeze(1)
            threshold = torch.mean(pairwise_diff)
            mask = (pairwise_diff <= threshold).all(dim=-1)  # Shape: [n, n]
            # `mask` is True where elements are within the threshold, False otherwise
            # print(mask)
            # Apply mask to pairwise_norm
            pairwise_norm = -1 * torch.norm(pairwise_diff, dim=-1)  # Shape: [n, n]
            # pairwise_norm = pairwise_norm.masked_fill(~mask, float('-inf'))  # Mask elements > threshold

            # Compute softmax on the masked pairwise_norm
            space_att = torch.softmax(pairwise_norm, dim=-1)  # Shape: [n, n]
        
        elif self.args.method.metric == 'cosine':
            # normalized_indicator = F.normalize(indicator, p=2, dim=1)
            # Compute cosine similarity by multiplying the tensor with its transpose
            similarity_matrix = torch.mm(indicator, indicator.T)
            space_att = torch.softmax(similarity_matrix, dim = -1)

        # print(space_att)

        self.collaboration_graph.append(space_att)
        sum_mean, sum_var = self.st_agg_bn(space_att=space_att, global_mean=global_mean, wotime=True)

        for cidx in range(client_num):
            self.clients[cidx].update_bnstatistics(sum_mean[cidx], sum_var[cidx])

    def aggregate_bn(self, train_round, global_mean, feature_indicator):
        # Store feature mean and variance
        n_chosen_layer = len(global_mean[0])
        client_num = self.args.client.client_num
        if len(self.history_feature) == 0:
            self.history_feature = [[] for _ in range(n_chosen_layer)]
            for chosen_layer in range(n_chosen_layer):
                feature_t = []
                for cidx in range(client_num):
                    feature_t.append(global_mean[cidx][chosen_layer])
                feature_t = torch.stack(feature_t, dim=0)
                self.history_feature[chosen_layer] = feature_t.unsqueeze(1)
        else:
            for chosen_layer in range(n_chosen_layer):
                feature_t = []
                for cidx in range(client_num):
                    feature_t.append(global_mean[cidx][chosen_layer])
                feature_t = torch.stack(feature_t, dim=0)
                self.history_feature[chosen_layer] = torch.cat([self.history_feature[chosen_layer], feature_t.unsqueeze(1)], dim=1)

        # calculate aggregation rate & aggregate model weight
        sum_mean = [[[] for _ in range(n_chosen_layer)] for _ in range(client_num)]
        sum_var = [[[] for _ in range(n_chosen_layer)] for _ in range(client_num)]
        if self.args.group.aggregation_method == 'st':
            time_att, space_att = self.ST_attention(feature_indicator)
            sum_mean, sum_var = self.st_agg_bn(time_att, space_att, global_mean)

        elif self.args.group.aggregation_method == 'avg':
            total_samples = float(sum([client.sample_num for client in self.clients]))
            for chosen_layer in range(len(global_mean[0])):
                half = len(global_mean[0][chosen_layer]) // 2
                for idx in range(len(global_mean)):
                    if idx == 0:
                        sum_mean[0][chosen_layer] = global_mean[idx][chosen_layer][:half] * self.clients[idx].sample_num
                        sum_var[0][chosen_layer] = global_mean[idx][chosen_layer][half:] * self.clients[idx].sample_num
                    else:
                        sum_mean[0][chosen_layer] += global_mean[idx][chosen_layer][:half] * self.clients[idx].sample_num
                        sum_var[0][chosen_layer] += global_mean[idx][chosen_layer][half:] * self.clients[idx].sample_num
            for idx in range(len(global_mean)):
                for chosen_layer in range(n_chosen_layer):
                    if idx == 0:
                        sum_mean[idx][chosen_layer] /= total_samples
                        sum_var[idx][chosen_layer] /= total_samples
                    else:
                        sum_mean[idx][chosen_layer] = sum_mean[0][chosen_layer]
                        sum_var[idx][chosen_layer] = sum_var[0][chosen_layer]

        elif self.args.group.aggregation_method == 'sim':
            time_att, space_att = self.ST_similarity(feature_indicator)
            sum_mean, sum_var = self.st_agg_bn(time_att, space_att, global_mean)
        
        elif self.args.group.aggregation_method == 'wotime':
            time_att, space_att = self.ST_attention(feature_indicator, wotime=True)
            sum_mean, sum_var = self.st_agg_bn(time_att, space_att, global_mean, wotime=True)

        elif self.args.group.aggregation_method == 'base':
            space_att= self.S_similarity(feature_indicator)
            sum_mean, sum_var = self.st_agg_bn(space_att=space_att, global_mean=global_mean, wotime=True)

        # print(space_att)
        for cidx in range(client_num):
            self.clients[cidx].update_bnstatistics(sum_mean[cidx], sum_var[cidx])

    def ST_attention(self, feature_indicator, wotime=False):
        '''
        :param feature_indicator: global_mean[ cidx ][ chosen_layer ][ D(mean) ]
        :return: weight1, weight2
        '''

        feature_indicator = torch.stack(feature_indicator, dim=0)
        if self.indicator.shape[0] == 0:
            self.indicator = feature_indicator.unsqueeze(1)
        else:
            self.indicator = torch.cat([self.indicator, feature_indicator.unsqueeze(1)], dim=1)

        # print(self.indicator.shape)

        # Get Aggregate Weights with Trainable Modules
        self.time_slide = self.args.other.time_slide
        if self.indicator.shape[1] < self.time_slide:
            feature_input = self.indicator[:, :, :]
        else:
            feature_input = self.indicator[:, self.indicator.shape[1]-self.time_slide:, :]
        ST_model = ST_block(args=self.args, dim=feature_input.shape[2])
        ST_model.cuda()
        opt = torch.optim.Adam(ST_model.parameters(), lr=self.args.other.st_lr)
        loss_min = 1000000
        epoch_num = self.args.other.st_epoch
        for epoch in range(epoch_num):
            # print('Epoch {}'.format(epoch))
            ST_model.train()
            logits, mask_logits, aug_logits, t_sim, s_sim = ST_model(feature_input, wotime=wotime)
            loss_reg = F.mse_loss(feature_input, logits)
            loss_consist = F.mse_loss(logits, mask_logits)
            loss_robust = F.mse_loss(logits, aug_logits)

            loss = (loss_reg + self.args.other.robust_weight * loss_robust)

            if loss.item() < loss_min:
                time_att = t_sim
                space_att = s_sim
                loss_min = loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

            # print('loss = {:.4f} + {:.4f} + {:.4f} = {:.4f}(min {:.4f})'.format(loss_reg.item(), loss_consist.item(),
            #                                                                     loss_robust.item(), loss.item(),
            #                                                                     loss_min))
        torch.cuda.empty_cache()
        return time_att, space_att

    # def ST_attention(self, feature_indicator, wotime=False):
    #     '''
    #     :param feature_indicator: global_mean[ cidx ][ chosen_layer ][ D(mean) ]
    #     :return: weight1, weight2
    #     '''
    #
    #     feature_indicator = torch.stack(feature_indicator, dim=0)
    #     if self.indicator.shape[0] == 0:
    #         self.indicator = feature_indicator.unsqueeze(1)
    #     else:
    #         self.indicator = torch.cat([self.indicator, feature_indicator.unsqueeze(1)], dim=1)
    #
    #     # Get Aggregate Weights with Trainable Modules
    #     self.time_slide = self.args.other.time_slide
    #
    #     if self.indicator.shape[1] < self.time_slide:
    #         feature_input = self.indicator[:, :, :]
    #     else:
    #         feature_input = self.indicator[:, self.indicator.shape[1]-self.time_slide:, :]
    #     ST_model = ST_block(args=self.args, dim=feature_input.shape[2])
    #     ST_model.cuda()
    #     opt = torch.optim.Adam(ST_model.parameters(), lr=self.args.other.st_lr)
    #     loss_min = 1000000
    #     epoch_num = self.args.other.st_epoch
    #     for epoch in range(epoch_num):
    #         index = 0
    #         while (index+self.time_slide <= self.indicator.shape[1]) or (index == 0):
    #             if index == 0 and (index + self.time_slide) > self.indicator.shape[1]-1:
    #                 feature_input, label = self.indicator[:, :, :], None
    #             elif (index + self.time_slide) > self.indicator.shape[1]-1:
    #                 feature_input, label = self.indicator[:, self.indicator.shape[1]-self.time_slide:, :], None
    #             else:
    #                 feature_input, label = self.indicator[:, index:index+self.time_slide, :], self.indicator[:, index+self.time_slide, :]
    #             index += 1
    #             print('Epoch {}, index {}'.format(epoch, index))
    #             ST_model.train()
    #             logits, mask_logits, aug_logits, t_sim, s_sim = ST_model(feature_input, wotime=wotime)
    #             loss_reg = F.mse_loss(feature_input, logits)
    #             loss_consist = F.mse_loss(logits, mask_logits)
    #             loss_robust = F.mse_loss(logits, aug_logits)
    #
    #             if label is not None:
    #                 loss_reg += F.mse_loss(logits[:, -1, :], label)
    #             loss = (loss_reg + self.args.other.robust_weight * loss_robust)
    #
    #             if loss.item() < loss_min:
    #                 time_att = t_sim
    #                 space_att = s_sim
    #                 loss_min = loss.item()
    #
    #             opt.zero_grad()
    #             loss.backward()
    #             opt.step()
    #
    #             print('loss = {:.4f} + {:.4f} + {:.4f} = {:.4f}(min {:.4f})'.format(loss_reg.item(), loss_consist.item(),
    #                                                                                 loss_robust.item(), loss.item(),
    #                                                                                 loss_min))
    #     torch.cuda.empty_cache()
    #     return time_att, space_att

    def ST_similarity(self, feature_indicator):
        feature_indicator = torch.stack(feature_indicator, dim=0)
        if self.indicator.shape[0] == 0:
            self.indicator = feature_indicator.unsqueeze(1)
        else:
            self.indicator = torch.cat([self.indicator, feature_indicator.unsqueeze(1)], dim=1)

        self.time_slide = self.args.other.time_slide
        if self.indicator.shape[1] < self.time_slide:
            feature_input = self.indicator[:, :, :]
        else:
            feature_input = self.indicator[:, self.indicator.shape[1] - self.time_slide:, :]
        ST_model = ST_block(args=self.args, dim=feature_input.shape[2])
        ST_model.cuda()
        logits, mask_logits, aug_logits, t_sim, s_sim = ST_model(feature_input)

        time_att = t_sim
        space_att = s_sim

        return time_att, space_att

    def S_similarity(self, feature_indicator):
        feature_indicator = torch.stack(feature_indicator, dim=0)
        if self.indicator.shape[0] == 0:
            self.indicator = feature_indicator.unsqueeze(1)
        else:
            self.indicator = torch.cat([self.indicator, feature_indicator.unsqueeze(1)], dim=1)
        self.time_slide = self.args.other.time_slide
        if self.indicator.shape[1] < self.time_slide:
            feature_input = self.indicator[:, :, :]
        else:
            feature_input = self.indicator[:, self.indicator.shape[1] - self.time_slide:, :]
        SA = SpatialAttention(dim=feature_input.shape[2], heads=1)
        SA.requires_grad_(False)
        SA.cuda()
        logits, logits_raw, s_sim = SA(feature_input)
        return s_sim


