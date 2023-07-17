from copy import deepcopy
import numpy as np
from typing import Iterable

from torch.utils import data


class NaiveDataset(data.Dataset):

    def __init__(self, tot_data: Iterable, indexes: Iterable):
        self.tot_data = tot_data
        self.indexes = indexes

    def __getitem__(self, item: int) -> object:
        return self.tot_data[self.indexes[item]]

    def __len__(self) -> int:
        return len(self.indexes)


def iid_sampling(dataset: Iterable, client_number: int, sample_num: int, seed: int) -> list:
    # if sample_num is not specified, then the dataset is divided equally among each client
    num_indices = len(dataset)
    if sample_num == 0:
        sample_num = num_indices // client_number

    random_state = np.random.RandomState(seed)

    dict_users, all_index = {}, [i for i in range(len(dataset))]
    for i in range(client_number):
        dict_users[i] = random_state.choice(all_index, sample_num, replace=False)

    return [NaiveDataset(tot_data=dataset, indexes=dict_users[i]) for i in range(len(dict_users))]


def pathological_sampling(dataset: Iterable, client_number: int, sample_num: int, seed: int, alpha: int) -> list:
    num_indices = len(dataset)
    labels = np.array([dataset[i]['class_id'] for i in range(num_indices)])
    num_classes = len(np.unique(labels))
    idxs_classes = [[] for _ in range(num_classes)]

    # If sample_num is not specified, then the dataset is divided equally among each client
    if sample_num == 0:
        sample_num = num_indices // client_number

    # Get samples for each class.
    for i in range(num_indices):
        idxs_classes[labels[i]].append(i)

    client_indexes = [[] for _ in range(client_number)]
    random_state = np.random.RandomState(seed)

    class_idxs = [i for i in range(num_classes)]
    for i in range(client_number):
        # Randomly select some classes to sample from.
        class_idx = random_state.choice(class_idxs, alpha, replace=False)
        for j in class_idx:
            # Calculate number of samples for each class.
            select_num = int(sample_num / alpha)
            # Sample a required number of samples.
            # If the number of samples in ``idx_classes[j]`` is more or equal than the required number,
            # set the argument ``replace=False``. Otherwise, set ``replace=True``
            selected = random_state.choice(idxs_classes[j], select_num, replace=(select_num > len(idxs_classes[j])))
            client_indexes[i] += list(selected)
        client_indexes[i] = np.array(client_indexes[i])
    return [NaiveDataset(tot_data=dataset, indexes=client_indexes[i]) for i in range(client_number)]


def dirichlet_sampling(dataset: Iterable, client_number: int, sample_num: int, seed: int, alpha: float) -> list:
    num_indices = len(dataset)
    labels = np.array([dataset[i]['class_id'] for i in range(num_indices)])
    num_classes = len(np.unique(labels))
    idxs_classes = [[] for _ in range(num_classes)]

    # If ``sample_num`` is not specified, the dataset is divided equally among each client
    if sample_num == 0:
        sample_num = num_indices // client_number

    # Get samples for each class.
    for i in range(num_indices):
        idxs_classes[labels[i]].append(i)

    client_indexes = [[] for _ in range(client_number)]
    random_state = np.random.RandomState(seed)
    q = random_state.dirichlet(np.repeat(alpha, num_classes), client_number)

    for i in range(client_number):
        num_samples_of_client = 0
        # Partition class-wise samples.
        for j in range(num_classes):
            # Make sure that each client have exactly ``sample_num`` samples.
            # For the last class, the number of samples is exactly the remaining sample number.
            select_num = int(sample_num * q[i][j] + 0.5) if j < num_classes - 1 else sample_num - num_samples_of_client
            # Record current sampled number.
            num_samples_of_client += select_num
            # Sample a required number of samples.
            # If the number of samples in ``idx_classes[j]`` is more or equal than the required number,
            # set the argument ``replace=False``. Otherwise, set ``replace=True``
            selected = random_state.choice(idxs_classes[j], select_num, replace=(select_num > len(idxs_classes[j])))
            client_indexes[i] += list(selected)
        client_indexes[i] = np.array(client_indexes[i])
    return [NaiveDataset(tot_data=dataset, indexes=client_indexes[i]) for i in range(client_number)]


sampling_methods = {
    'iid': iid_sampling,
    'dirichlet': dirichlet_sampling,
    'pathological': pathological_sampling,
}


def data_sampling(dataset: Iterable, args: dict, seed: int, train: bool = True) -> object:
    sampling_config = deepcopy(args.data.sample_method)
    train_num, test_num = sampling_config.pop('train_num'), sampling_config.pop('test_num')
    sample_num = train_num if train else test_num
    sampling_name = sampling_config.pop('name')
    try:
        sampling_func = sampling_methods[sampling_name]
    except KeyError:
        raise ValueError(f'Unrecognized sampling method: {args.data.sample_method.name}')
    return sampling_func(dataset, args.client.client_num, sample_num, seed, **sampling_config)
