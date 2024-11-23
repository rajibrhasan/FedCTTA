import numpy as np
from copy import deepcopy

import torch 
from torch import nn, optim
import torch.nn.functional as F
from sklearn.decomposition import PCA
from fed_utils import ema_update_model
from losses import symmetric_cross_entropy, softmax_entropy_ema, softmax_entropy

class Client(object):
    def __init__(self, name, model, cfg, device):
        self.cfg = cfg
        self.name = name 
        self.src_model = deepcopy(model)
        self.model = deepcopy(model)

        self.configure_model()
        self.params, param_names = self.collect_params()
        self.optimizer = self.setup_optimizer() if len(self.params) > 0 else None

        self.correct_preds = []
        self.total_preds = []
        self.domain_list = []

        self.momentum_probs = cfg.MISC.MOMENTUM_PROBS
        self.temperature = cfg.MISC.TEMP

        self.device = device
        self.class_probs_ema = 1 / self.cfg.MODEL.NUM_CLASSES * torch.ones(self.cfg.MODEL.NUM_CLASSES).to(self.device)
        self.pvec = None
        self.local_features = None
        self.weights = None


    def adapt(self, x, y):
        self.x = x
        self.y = y
        self.model.to(self.device)
        feats, outputs = self.model(self.x.to(self.device))

        if self.cfg.MODEL.ADAPTATION == 'roid':
            weights_div = 1 - F.cosine_similarity(self.class_probs_ema.unsqueeze(dim=0), outputs.softmax(1), dim=1)
            weights_div = (weights_div - weights_div.min()) / (weights_div.max() - weights_div.min())
            # calculate certainty based weight
            weights_cert = - softmax_entropy(outputs)
            print(weights_cert)
            weights_cert = (weights_cert - weights_cert.min()) / (weights_cert.max() - weights_cert.min())
            print(weights_cert)

            # calculate the final weights
            weights = torch.exp(weights_div * weights_cert / self.cfg.MISC.TEMP)
            self.class_probs_ema = self.momentum_probs * self.class_probs_ema + (1 - self.momentum_probs) * outputs.softmax(1).mean(0)

            self.weights = weights.mean(0).item()
        
        if self.cfg.MODEL.ADAPTATION != 'source':
            loss = softmax_entropy(outputs).mean(0)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.model.to('cpu')
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == self.y.to(self.device)).sum().item()
        self.correct_preds.append(correct)
        self.total_preds.append(len(self.y))

    def setup_optimizer(self):
        """Set up optimizer for tent adaptation.
        For best results, try tuning the learning rate and batch size.
        """
        if self.cfg.OPTIM.METHOD == 'Adam':
            return optim.Adam(self.params,
                        lr=self.cfg.OPTIM.LR,
                        betas=(self.cfg.OPTIM.BETA, 0.999),
                        weight_decay=self.cfg.OPTIM.WD)
        
        elif self.cfg.OPTIM.METHOD == 'SGD':
            return optim.SGD(self.params,
                    lr=self.cfg.OPTIM.LR,
                    momentum=self.cfg.OPTIM.MOMENTUM,
                    dampening=self.cfg.OPTIM.DAMPENING,
                    weight_decay=self.cfg.OPTIM.WD,
                    nesterov=self.cfg.OPTIM.NESTEROV)
        else:
            raise NotImplementedError(f"Unknown optimizer: {self.cfg.optim_method}")
    
    def configure_model(self):
        """Configure model."""
        # self.model.train()
        self.model.train()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        # disable grad, to (re-)enable only what we update
        self.model.requires_grad_(False)
        # enable all trainable
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
                m.requires_grad_(True)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)
                
    def collect_params(self):
        """Collect the affine scale + shift parameters from normalization layers.
        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias'] and p.requires_grad:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def extract_bn_weights_and_biases(self):
        bn_params = {}
        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                gamma = layer.weight.data.cpu()  # Scale (weight)
                beta = layer.bias.data.cpu()    # Offset (bias)
                weights = torch.cat((gamma, beta), dim =0)
                bn_params[name] = weights
        return bn_params

    def get_state_dict(self):
        return self.model.state_dict()
    
    def set_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
    
    def get_model(self):
        return self.model
    
   

 
   
   