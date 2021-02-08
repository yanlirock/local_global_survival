from torch.utils.data import Dataset
import numpy as np
import h5py


def deepsurvival_hf5_reader(file=None):
    """
    read and extract survival train/test data from deepsurvival hdf5 files
    :param file:
    :return:
    """
    if not file.endswith('h5'):
        raise TypeError("function 'deepsurvival_hf5_reader' can only "
                        "read '.h5 files")
    f = h5py.File(file, 'r')
    train_data = f['train']
    test_data = f['train']
    x_train = train_data['x'][:]
    time_train = train_data['t'][:]
    event_train = train_data['e'][:]
    x_test = test_data['x'][:]
    time_test = test_data['t'][:]
    event_test = test_data['e'][:]
    return x_train, time_train, event_train, x_test, time_test, event_test


class SurvData(Dataset):
    def __init__(self, x: np.ndarray, target, event=None):
        self.x = x
        self.length = x.shape[0]
        if type(target) == tuple and event is None:
            y, event = target
            self.y = y
            self.event = event
        elif type(target) == np.ndarray:
            if target.ndim == 2 and event is None:
                self.y = target[:, 0]
                self.event = target[:, 1]
            elif target.ndim == 1 and event is not None:
                self.y = target
                self.event = event
            else:
                raise ValueError("Please check the status of the input "
                                 "target time and event indicator")
        else:
            raise NotImplementedError(
                "Currently only support 'tuple' and 'np.ndarray' "
                "for target time and event indicator")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        event = self.event[index]
        return x, y, event


def collate_fn(data):
    # separate feature, target time, and event indicator
    x, y, event = zip(*data)
    return np.asarray(x), np.asarray(y), np.asarray(event)


