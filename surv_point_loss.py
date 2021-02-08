import torch


class SurvivalPointLoss(torch.nn.Module):
    """
    the loss function for pointwise comparision
    """
    def __init__(self, sigma=0.1, measure='mse', dtype=None):
        super(SurvivalPointLoss, self).__init__()
        self.sigma = sigma
        self.measure = measure
        self.dtype = dtype

    def forward(self, event_indicator, event_time, estimate):
        t_event_time = torch.from_numpy(event_time).type(self.dtype)
        # this are weights for uncensored instances
        uc_weights = torch.from_numpy(event_indicator).type(self.dtype)
        # this are weights for censored instances
        num_sample = event_indicator.shape[0]
        c_event_ind = torch.ones(num_sample, requires_grad=True
                                 ).type(self.dtype) - uc_weights
        # the difference is counted only if the predicted time is smaller
        # than censored time
        is_less = torch.sigmoid((t_event_time - estimate) / self.sigma)
        c_weights = c_event_ind * is_less

        all_weights = uc_weights+c_weights
        mu = torch.mean(t_event_time)

        if self.measure == 'mse':
            v = torch.sum(((t_event_time - mu) * all_weights)**2) + 1e-5
            loss = torch.sum(((t_event_time - estimate) * all_weights)**2)/v
        elif self.measure == 'mae':
            v = torch.sum(torch.abs(t_event_time-mu)*all_weights) + 1e-5
            loss = torch.sum(torch.abs(t_event_time-estimate)*all_weights)/v
        else:
            raise NotImplementedError("Currently only support "
                                      "'mse', and 'mae' losses.")
        return loss
