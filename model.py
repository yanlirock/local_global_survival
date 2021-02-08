import torch
import numpy as np
from SoftCindex import SoftCindexCensoredLoss
from surv_point_loss import SurvivalPointLoss


class DeepCindex(torch.nn.Module):
    """
    Deep Cindex model for right-censored data
    """
    def __init__(self,
                 model,
                 sigma=0.01,
                 Cindex_type='Harrell',
                 event_train=None,
                 time_train=None
                 ):
        super(DeepCindex, self).__init__()
        self.model = model
        self.loss_layer = SoftCindexCensoredLoss(
            sigma=sigma,
            Cindex_type=Cindex_type,
            train_event_indicator=event_train,
            train_event_time=time_train)

    def forward(self, x, event_time, event_indicator):
        estimate = self.model(x)
        estimate = estimate.view(-1)
        cindex = self.loss_layer(event_indicator, event_time, estimate)
        return -1*cindex, estimate, cindex, cindex


class DeepAsymmetric(torch.nn.Module):
    """
    Deep learning model for right-censored data with asymmetric pointwise loss
    """
    def __init__(self,
                 model,
                 sigma=0.01,
                 measure='mse',
                 dtype=None
                 ):
        super(DeepAsymmetric, self).__init__()
        self.model = model
        self.loss_layer = SurvivalPointLoss(
            sigma=sigma, measure=measure, dtype=dtype)

    def forward(self, x, event_time, event_indicator):
        estimate = self.model(x)
        estimate = estimate.view(-1)
        loss = self.loss_layer(event_indicator, event_time, estimate)
        return loss, estimate, loss, loss


class CombinedLossSurvModel(torch.nn.Module):
    """
    Deep learning model for right-censored data with multiple loss function
    """

    def __init__(self,
                 model,
                 sigma=0.01,
                 Cindex_type='Harrell',
                 event_train=None,
                 time_train=None,
                 measure='mse',
                 dtype=None,
                 alpha=1
                 ):
        super(CombinedLossSurvModel, self).__init__()
        self.model = model
        self.cindex_loss = SoftCindexCensoredLoss(
            sigma=sigma,
            Cindex_type=Cindex_type,
            train_event_indicator=event_train,
            train_event_time=time_train, dtype=dtype)
        self.pointwise_loss = SurvivalPointLoss(
            sigma=sigma, measure=measure, dtype=dtype)
        self.alpha = alpha
        self.measure = measure

    def forward(self, x, event_time, event_indicator):
        estimate = self.model(x)
        estimate = estimate.view(-1)
        cindex = -1 * self.cindex_loss(event_indicator, event_time, estimate)
        loss = self.pointwise_loss(event_indicator, event_time, estimate)

        total_loss = self.alpha * loss + (1 - self.alpha) * cindex

        return total_loss, estimate, cindex, loss
