from easydict import EasyDict
import datetime
import os
import time
import random

now = int(time.time())

common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                      'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                      'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
exp_args = dict(
    data=dict(dataset='cifar10_test', data_path='./data/CIFAR10', sample_method=dict(name='iid', train_num=50000, test_num=500),
              corruption=['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                      'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                      'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'],
              level=[5], class_number=10),
    learn=dict(
        device='cuda:0', local_eps=1, global_eps=1, batch_size=64, optimizer=dict(name='sgd', lr=0.00001, momentum=0.9)
    ),
    model=dict(
        name='resnet8',
        input_channel=3,
        class_number=10,
    ),
    client=dict(name='fedpl_client', client_num=20),
    server=dict(name='base_server'),
    group=dict(name='adapt_group', aggregation_method='st',
               aggregation_parameters=dict(
                   name='all',
               )),
    other=dict(test_freq=3, logging_path='./logging/0126_cifar10_resnet_loop5_seed1',
               model_path='./pretrain/resnet8_cifar10.ckpt',
               partition_path='4area.npy',
               online=True,
               adap_iter=1,
               ttt_batch=10,

               is_continue=True,
               niid=True,

               is_average=True,
               method='adapt',
               pre_trained='resnet8',
               resume=True,

               time_slide=10,
               st_lr=1e-4,
               st_epoch=100,
               robust_weight=0.5,
                st='both',

                st_head=1,
               loop=1,
               alpha = 0.9
               ),
    method=dict(name = "ours", #Ffedtsa or ours
                feat_sim = "output", #output or feature
                data_used = "random",
                metric = 'euclid'
            ),
)

exp_args = EasyDict(exp_args)
seed = 3
iid_text = "niid" if exp_args.other.niid else "iid"
file_name = f"agg_freq_{exp_args.method.name}_{exp_args.method.data_used}_{exp_args.method.feat_sim}_{exp_args.other.method}_{exp_args.client.name}_lp_{exp_args.other.loop}_seed{seed}_{now}"
exp_args.other.logging_path = os.path.join('logging', exp_args.data.dataset, "tta_"+exp_args.other.method, iid_text, file_name )
print(exp_args.other.logging_path)

if __name__ == '__main__':
    from fling.pipeline import FedTTA_Pipeline
    FedTTA_Pipeline(exp_args, seed=seed)
   