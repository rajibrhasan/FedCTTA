import numpy as np
from copy import deepcopy

import torch 
from torch import nn, optim
import torch.nn.functional as F
from sklearn.decomposition import PCA
from fed_utils import ema_update_model
from losses import symmetric_cross_entropy, softmax_entropy_ema, softmax_entropy, Entropy, SymmetricCrossEntropy, SoftLikelihoodRatio
from transforms_cotta import get_tta_transforms

@torch.no_grad()
def update_model_probs(x_ema, x, momentum=0.9):
    return momentum * x_ema + (1 - momentum) * x

class Client(object):
    def __init__(self, name, model, cfg, device):
        self.cfg = cfg
        self.name = name 
        self.model = deepcopy(model)

        self.img_size = (32, 32) if "cifar" in cfg.CORRUPTION.DATASET else (224, 224)
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.use_weighting = cfg.ROID.USE_WEIGHTING
        self.use_prior_correction = cfg.ROID.USE_PRIOR_CORRECTION
        self.use_consistency = cfg.ROID.USE_CONSISTENCY
        self.momentum_src = cfg.ROID.MOMENTUM_SRC
        self.momentum_probs = cfg.ROID.MOMENTUM_PROBS
        self.temperature = cfg.ROID.TEMPERATURE
        self.batch_size = cfg.MISC.BATCH_SIZE
        self.tta_transform = get_tta_transforms(self.img_size, padding_mode="reflect", cotta_augs=False)

        # setup loss functions
        self.slr = SoftLikelihoodRatio()
        self.symmetric_cross_entropy = SymmetricCrossEntropy()
        self.softmax_entropy = Entropy()  # not used as loss

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

        self.src_model = deepcopy(self.model).cpu()
        for param in self.src_model.parameters():
            param.detach_()


    def adapt(self, x, y):
        self.x = x
        self.y = y
        self.model.to(self.device)

        outputs = self.model(x.to(self.device))

        if self.use_weighting:
            with torch.no_grad():
                # calculate diversity based weight
                weights_div = 1 - F.cosine_similarity(self.class_probs_ema.unsqueeze(dim=0), outputs.softmax(1), dim=1)
                weights_div = (weights_div - weights_div.min()) / (weights_div.max() - weights_div.min())
                mask = weights_div < weights_div.mean()

                # calculate certainty based weight
                weights_cert = - self.softmax_entropy(logits=outputs)
                weights_cert = (weights_cert - weights_cert.min()) / (weights_cert.max() - weights_cert.min())

                # calculate the final weights
                weights = torch.exp(weights_div * weights_cert / self.temperature)
                weights[mask] = 0.

                self.class_probs_ema = update_model_probs(x_ema=self.class_probs_ema, x=outputs.softmax(1).mean(0), momentum=self.momentum_probs)

        # calculate the soft likelihood ratio loss
        loss_out = self.slr(logits=outputs)

        # weight the loss
        if self.use_weighting:
            loss_out = loss_out * weights
            loss_out = loss_out[~mask]
        loss = loss_out.sum() / self.batch_size

        # calculate the consistency loss
        if self.use_consistency:
            outputs_aug = self.model(self.tta_transform(x[~mask]))
            loss += (self.symmetric_cross_entropy(x=outputs_aug, x_ema=outputs[~mask]) * weights[~mask]).sum() / self.batch_size
        
        self.model = ema_update_model(
            model_to_update=self.model,
            model_to_merge=self.src_model,
            momentum=self.momentum_src,
            device=self.device
        )

        with torch.no_grad():
            if self.use_prior_correction:
                prior = outputs.softmax(1).mean(0)
                smooth = max(1 / outputs.shape[0], 1 / outputs.shape[1]) / torch.max(prior)
                smoothed_prior = (prior + smooth) / (1 + smooth * outputs.shape[1])
                outputs *= smoothed_prior

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
        self.model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
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
    
   

 
   
   