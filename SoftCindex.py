import torch
import numpy as np
from nonparametric import CensoringDistributionEstimator


def compute_weights(event_indicator, event_time, weights):
    n = event_time.shape[0]
    wweights = np.repeat(weights, n).reshape(n, n)
    weightsj = np.repeat(event_time, n).reshape(n, n)
    weightsk = np.tile(event_time, n).reshape(n, n)
    weightsI = (weightsj == weightsk) * 0.5 + (weightsj < weightsk) - np.eye(
        n) * 0.5
    censored_id = np.repeat(event_indicator, n).reshape(n, n)
    wweights = censored_id * weightsI * wweights
    wweights = wweights / np.sum(wweights)
    return wweights


class SoftCindexCensoredLoss(torch.nn.Module):
    """
    soft_Concordance index for right-censored data
    """

    def __init__(self,
                 sigma=0.1,
                 Cindex_type='Harrell',
                 train_event_indicator=None,
                 train_event_time=None,
                 dtype=None
                 ):
        """
        :param sigma: float, used to control smoothness in sigmoid function
        :param tied_tol: The tolerance value for considering ties.
                If the absolute difference between risk scores is smaller
                or equal than `tied_tol`, risk scores are considered tied.
        :param Cindex_type: the type of C-index {'Harrell','ipcw'}
        """
        super(SoftCindexCensoredLoss, self).__init__()
        self.sigma = sigma
        self.type = Cindex_type
        self.dtype = dtype
        if self.type not in {"Harrell", "ipcw"}:
            raise NotImplementedError("currently we only support 'Harrell' "
                                      "or 'ipcw' ")

        if self.type == "ipcw":
            if train_event_indicator is None:
                raise ValueError(
                    "When using 'ipcw', you need to provide survival "
                    "time and event status of training data")
            else:
                survival_train = np.zeros(train_event_indicator.shape[0],
                                          dtype={'names': ('event', 'time'),
                                                 'formats': ('bool', 'float')})
                survival_train['event'] = train_event_indicator.astype(bool)
                survival_train['time'] = train_event_time
                self.cens = CensoringDistributionEstimator()
                self.cens.fit(survival_train)
        else:
            self.cens = None

    def forward(self, event_indicator, event_time, estimate):
        """
        :param event_indicator: array-like, shape = (n_samples,) {0, 1}
            array denotes whether an event occurred
        :param event_time: array-like, shape = (n_samples,)
            the time of an event or time of censoring
        :param estimate: Estimated time/risk of experiencing an event
        :return:
        """
        if self.type == "Harrell":
            weights = np.ones_like(event_time)
        elif self.type == "ipcw":
            survival_test = np.zeros(event_indicator.shape[0],
                                      dtype={'names': ('event', 'time'),
                                             'formats': ('bool', 'float')})
            survival_test['event'] = event_indicator.astype(bool)
            survival_test['time'] = event_time
            ipcw_test = self.cens.predict_ipcw(survival_test)
            weights = np.square(ipcw_test)

        weights_out = compute_weights(event_indicator, event_time, weights)
        tf_weights_out = torch.from_numpy(weights_out).type(self.dtype)
        n = estimate.shape[0]
        etak = estimate.view(-1, 1).expand(n, n)
        etaj = estimate.view(1, -1).expand(n, n)
        etaMat = torch.sigmoid((etaj - etak) / self.sigma) * tf_weights_out
        cindex = torch.sum(etaMat)

        return cindex
