from easydict import EasyDict

exp_args = dict(
    data=dict(
        dataset='cifar10',
        data_path='./data/CIFAR10',
        sample_method=dict(name='iid')
    ),
    learn=dict(
        device='cuda:0',
        local_eps=8,
        global_eps=400,
        batch_size=32,
        optimizer=dict(name='sgd', lr=0.001, momentum=0.9),
    ),
    model=dict(
        name='cnn',
        input_channel=3,
        linear_hidden_dims=[256],
        class_number=10,
    ),
    client=dict(name='base_client', client_num=1),
    server=dict(name='base_server'),
    group=dict(
        name='base_group',
        aggregation_method='avg',
    ),
    other=dict(test_freq=3, logging_path='./logging/cifar10_pretrain_cnn')
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling.pipeline import generic_model_pipeline
    generic_model_pipeline(exp_args, seed=0)