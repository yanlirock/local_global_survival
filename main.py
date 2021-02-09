import os
import time
import torch
import torchtuples as tt
import numpy as np
import easydict
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import StratifiedKFold

from model import DeepCindex, DeepAsymmetric, CombinedLossSurvModel
from surv_point_loss import SurvivalPointLoss
from metrics import concordance_index_censored
from surv_data import SurvData, collate_fn, deepsurvival_hf5_reader
from torch.utils.data import DataLoader

path = os.getcwd()+'/data/'
file  = 'nwtco_all.csv'
file_name = path+file
data = pd.read_csv(file_name)

args = easydict.EasyDict({
    "batch_size": 1024,
    "cuda": False,
    "lr": 0.05,
    "seed": 1111,
    "reduce_rate": 0.95,
    "epochs": 200,
    "clip": 5.0,
    "log_interval":10,
})

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.FloatTensor
if torch.cuda.is_available() and args.cuda:
    dtype = torch.cuda.FloatTensor

np.random.seed(1234)
_ = torch.manual_seed(123)

X = data.iloc[:, :-2]
X_normalize = preprocessing.scale(X)
time_all = data.iloc[:, -2].values
event_all = data.iloc[:, -1].values

skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(X_normalize, event_all)

for train_index, test_index in skf.split(X_normalize, event_all):
    x_train, x_test = X_normalize[train_index], X_normalize[test_index]
    time_train, time_test = time_all[train_index], time_all[test_index]
    event_train, event_test = event_all[train_index], event_all[test_index]

    # initialize model

    in_features = x_train.shape[1]
    num_nodes = [32,32]
    out_features = 1
    batch_norm = True
    dropout = 0.1
    output_bias = False

    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                                  dropout, output_bias=output_bias)


    """
    model  = DeepCindex(net, sigma=0.01, Cindex_type="Harrell", 
                        event_train= event_all,
                        time_train = time_train).to(device)

    model = DeepAsymmetric(net, sigma=0.01,measure='mae',dtype=dtype).to(device)
    """
    model  = CombinedLossSurvModel(net, sigma=0.01, Cindex_type="ipcw", 
                        event_train= event_all,alpha=alpha,
                        time_train = time_all, measure='mae',dtype=dtype).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=args.reduce_rate)


    # load data
    train_data = SurvData(x_train, time_train, event_train)
    train_load = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    train_loader = iter(cycle(train_load))

    test_data = SurvData(x_test, time_test, event_test)
    test_load = DataLoader(
        dataset=test_data,
        batch_size=x_test.shape[0],
        shuffle=True,
        collate_fn=collate_fn
    )
    test_loader = iter(cycle(test_load))

    train_iter = len(train_data) // args.batch_size +1
    test_iter = 1

    # Start model training and evaulation
    cindex_epoch = []
    mae_epoch = []
    for epoch in range(args.epochs):
        start_time = time.time()

        for i_iter in range(train_iter):
            model.train()
            x, y, event = next(train_loader)
            x = torch.from_numpy(x).to(device).type(dtype)
            loss, _,_,_= model(x, y, event)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            if i_iter % args.log_interval == 0 and i_iter > 0:
                elapsed = time.time() - start_time
                cur_loss = loss.item()
                print('epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                      'current loss {:5.4f} '.format(
                          epoch, i_iter, train_iter, elapsed * 1000 / args.log_interval,cur_loss))
                start_time = time.time()

        event_indicator = np.array([])
        event_time = np.array([])
        estimate = np.array([])
        for i_test in range(test_iter):
            model.eval()
            x_test, y_test, event_test = next(test_loader)
            x_test = torch.from_numpy(x_test).to(device).type(dtype)
            _, estimate_y, _, _ = model(x_test, y_test, event_test)
            event_indicator = np.hstack((event_indicator, event_test))
            event_time x= np.hstack((event_time,y_test))
            estimate = np.hstack((estimate,estimate_y.cpu().detach().numpy()))

        test_cindex = concordance_index_censored(event_indicator.astype(bool), event_time, -1*estimate)
        cindex,_,_,_,_ = test_cindex
        mae = sum(abs(estimate-event_time)*event_indicator)/sum(event_indicator)
        print(f"Cindex for test data is: {cindex}, MAE for uncensored instance is: {mae}")
        
